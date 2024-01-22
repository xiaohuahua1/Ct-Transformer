from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from args import *
from data_process import *
from TCN import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q:[B,L, D]
        B, L_K, E = K.shape
        _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 这里扩展出来一维L_Q，表示每一个query都有L_K个对应的key，且每个key是长度为E的向量 k_expand的维度是[B, H, L_Q, L_K, E]
        # K_expand = K.unsqueeze(-3).expand(B, L_Q, L_K, E) 
        K_expand = K.unsqueeze(-3).expand(B, L_Q, L_K, E) 
        # index_sample的维度是[L_Q, sample_k]
        # 该函数的作用是生成一个形状为 (L_Q, sample_k) 的张量，其中的每个元素都是从 [0, L_K) 范围（左闭右开）内按离散均匀分布随机抽取的整数。
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q 为每一个query采样sampe_k个key的index
        # K_sample的维度是[B,L_Q, sample_k, E]
        # torch.arange(L_Q).unsqueeze(1)这句话相当于在L_Q的这个维度之后加了一个维度
        K_sample = K_expand[:, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # # Q.unsqueeze(-2)后的维度是[B, H, L_Q, 1, D], K_sample.transpose(-2, -1)后的维度是[B, H, L_Q, E, sample_k]，此处D与E的长度应该相同
        # # torch.matmul的计算应该可以这样理解，由于D=E(源码中应该是64)，所以将[1, D]的张量与[D, sample_k]的张量做矩阵相乘，然后删除1的维度则最终
        # # Q_K_sample的维度就是[B,L_Q, sample_k], 含义为每一个query与采样下来的sample_k个key内积后的attention结果
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2) # 这里的squeeze需要加上-2如果不加的话，在运行batch_size=1时会出现问题

        # # find the Top_k query with sparisty measurement
        # # torch.div(Q_K_sample.sum(-1), L_K)计算后的维度是[B, H, L_Q], Q_K_sample.max(-1)[0]计算后的维度是[B, H, L_Q]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # # M_top的维度是[B,n_top]
        M_top = M.topk(n_top, sorted=False)[1]

        # # use the reduced Q to calculate Q_K
        # # Q_reduce的维度是[B,n_top, D]
        Q_reduce = Q[torch.arange(B)[:,None], M_top, :] # factor*ln(L_q)
        # # Q_k的维度是[B,n_top, L_K]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B,L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            # contex:[batch,len,hidden]
            contex = V_sum.unsqueeze(-2).expand(B,L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex
    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):

        B,L_V, D = V.shape
        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        # context_in:[batch,len,hidden]
        context_in[torch.arange(B)[:, None],index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B,L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)


    def forward(self, queries, keys, values, attn_mask=None):
        # q:[batch,len,d_q]
        # k:[batch,len,d_k]
        # v:[batch,len,d_v]
        B, L_Q,D = queries.shape
        _, L_K, _ = keys.shape

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # # add scale factor
        scale = self.scale or 1. / np.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # # get the context
        context = self._get_initial_context(values, L_Q)
        # # update the context with selected top_k queries
        # context:[batch,len,hidden]
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn
    
# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model//n_heads) #如果d_model=512并且采用默认n_heads=8时，d_keys=64
#         d_values = d_values or (d_model//n_heads)

#         self.inner_attention = attention # FullAttention or ProbAttention
#         # https://pytorch.org/docs/master/generated/torch.nn.Linear.html?highlight=nn%20linear#torch.nn.Linear
#         # 由官方对nn.Linear的介绍可知，全连接只针对最后一维特征进行全连接
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads) 
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#         self.mix = mix #Informer类和InformerStack类中是False
        
#     def forward(self, queries, keys, values, attn_mask = None):
#         B, L, _ = queries.shape 
#         _, S, _ = keys.shape #这里S与L的维度应该相同才对
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask
#         )
#         if self.mix:
#             out = out.transpose(2,1).contiguous()
#         out = out.view(B, L, -1) #out的维度应该是[batch_size, seq_len, d_values*n_heads]

#         return self.out_projection(out), attn #out_projection前向过程结束后张量维度应该是[batch_size, seq_len, d_model]
    
class AttentionLayer(nn.Module):
    def __init__(self, attention,n_head, d_model, dropout):
        super(AttentionLayer, self).__init__()
            
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        
        self.v_layer = nn.Linear(self.d_model, self.d_v, bias = False)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q, bias = False) 
                                       for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias = False) 
                                       for _ in range(self.n_head)])
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])
        # self.attention = ProbAttention(False, 5, attention_dropout=0.1,output_attention=True)
        self.attention = attention
        self.w_h = nn.Linear(self.d_v, self.d_model, bias = False)
        
        self.init_weights()
            
    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
#                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            else:
                torch.nn.init.zeros_(p)
        
    def forward(self, q, k, v, mask = None):
        
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            #print('qs layer: {}'.format(qs.shape))
            head, attn = self.attention(qs, ks, vs, mask)
            # print('head layer: {}'.format(head.shape))
            #print('attn layer: {}'.format(attn.shape))
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        # head:[batch,len,n_head,d_k]    
        head = torch.stack(heads, dim = 2) if self.n_head > 1 else heads[0]
        # print('concat heads: {}'.format(head.shape))
        #print('heads {}: {}'.format(0, head[0,0,Ellipsis]))
        attn = torch.stack(attns, dim = 2)
        #print('concat attn: {}'.format(attn.shape))
        # outputs:[batch,len,d_k]
        outputs = torch.mean(head, dim = 2) if self.n_head > 1 else head
        # print('outputs mean: {}'.format(outputs.shape))
        #print('outputs mean {}: {}'.format(0, outputs[0,0,Ellipsis]))
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        
        return outputs, attn
    
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention #AttentionLayer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1) #d_ff依赖self.args.d_ff含义是Dimension of fcn (defaults to 2048)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model) #LayerNorm取同一样本的不同通道进行归一化
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
	
    # attention layer->skip layer操作->LayerNorm->MLP(conv1d)->skip layer操作->LayerNorm
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            mask = attn_mask
        ) #new_x:[batch_size, seq_len, d_model]
        # 这里有一个skip layer的操作
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # 两层的1维卷积操作
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        
        return self.norm2(x+y), attn
    
class ConvLayer(nn.Module):
    # c_in的维度应该与d_model=512相同
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x:[batch_size, seq_len, d_model]
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        # x = self.maxPool(x) #经过maxPool操作后，x:[batch_size, d_model, seq_len/2]
        x = x.transpose(1,2) #第一次经过conv_layer时，返回结果的维度是[batch_size, seq_len/2, d_model]
        return x

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers) # EncoderLayer List
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x:[batch_size, seq_len, d_model]
        attns = [] #记录每层attention的结果
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask) #x:[batch_size, seq_len, d_model]
                x = conv_layer(x) #进行Self-attention distilling来减小内存的占用 x:[batch_size, seq_len/2, d_model]第一次经过conv_layer时
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                # x:[batch_size, seq_len, d_model]
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns



    
if __name__ == '__main__':
    batch = 128
    length = 20
    d_model = 512
    d_ff = 2048
    n_head = 8
    e_layer = 2
    dropout = 0.1
    activation = 'gelu'
    distil = True

    input = torch.randn([batch,length,d_model])

    encoder = Encoder([EncoderLayer(AttentionLayer(ProbAttention(False, 5, attention_dropout=dropout,output_attention=True),
                                                   n_head,d_model,dropout),
                                                   d_model,d_ff
                                                   ) for i in range(e_layer)],
                        [ConvLayer(d_model) for l in range(e_layer - 1)],
                        norm_layer=torch.nn.LayerNorm(d_model)
                                                   )
    
    x,attns = encoder(input)
    # print(x.shape)

    # attentionLayer =  AttentionLayer(n_head,hidden,0.1)
    # attn = ProbAttention(False, 5, attention_dropout=0.1,output_attention=False)
    # attn(input,input,input)
    # outputs:[batch,len,hidden]
    # attn:[batch,len,n_head,len]?
    # outputs, attn = attentionLayer(input,input,input)
    # print(attn.shape)




    
if __name__ == '__main__':
    batch = 128
    length = 20
    d_model = 512
    d_ff = 2048
    n_head = 8
    e_layer = 2
    dropout = 0.1
    activation = 'gelu'
    distil = True

    input = torch.randn([batch,length,d_model])

    encoder = Encoder([EncoderLayer(AttentionLayer(ProbAttention(False, 5, attention_dropout=dropout,output_attention=True),
                                                   n_head,d_model,dropout),
                                                   d_model,d_ff
                                                   ) for i in range(e_layer)],
                        [ConvLayer(d_model) for l in range(e_layer - 1)],
                        norm_layer=torch.nn.LayerNorm(d_model)
                                                   )
    
    x,attns = encoder(input)
    # print(x.shape)

    # attentionLayer =  AttentionLayer(n_head,hidden,0.1)
    # attn = ProbAttention(False, 5, attention_dropout=0.1,output_attention=False)
    # attn(input,input,input)
    # outputs:[batch,len,hidden]
    # attn:[batch,len,n_head,len]?
    # outputs, attn = attentionLayer(input,input,input)
    # print(attn.shape)
