from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from args import *
from data_process import *
from TCN import *
from myModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,k=5):
        super(context_embedding,self).__init__()
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)

    def forward(self,x):
        x = self.causal_convolution(x)
        return F.tanh(x)
    
# Self Attention Class
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers=8, kernel_size=5, mask_next=True, mask_diag=False):
        super().__init__()

        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag

        h = headers

        # Query, Key and Value Transformations
        padding = (kernel_size - 1)
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)

        self.toqueries = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True)
        self.tovalues = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)  # No convolution operated

        # Heads unifier
        self.unifyheads = nn.Linear(k * h, k)

    def forward(self, x):
        # Extraction dimensions
        b, t, k = x.size()  # batch_size, number_of_timesteps, number_of_time_series

        # Checking Embedding dimension
        assert self.k == k, 'Number of time series ' + str(k) + ' didn t much the number of k ' + str(
            self.k) + ' in the initiaalization of the attention layer.'
        h = self.headers

        #  Transpose to see the different time series as different channels
        x = x.transpose(1, 2)
        x_padded = self.padding_opertor(x)

        # Query, Key and Value Transformations
        queries = self.toqueries(x_padded).view(b, k, h, t)
        keys = self.tokeys(x_padded).view(b, k, h, t)
        values = self.tovalues(x).view(b, k, h, t)

        # Transposition to return the canonical format
        queries = queries.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        queries = queries.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        values = values.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        values = values.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        keys = keys.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        keys = keys.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        # Weights
        queries = queries / (k ** (.25))
        keys = keys / (k ** (.25))

        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        weights = torch.bmm(queries, keys.transpose(1, 2))

        ## Mask the upper & diag of the attention matrix
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(t, t, offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
            else:
                indices = torch.triu_indices(t, t, offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')

        # Softmax
        weights = F.softmax(weights, dim=2)

        # Output
        output = torch.bmm(weights, values)
        output = output.view(b, h, t, k)
        output = output.transpose(1, 2).contiguous().view(b, t, k * h)

        return self.unifyheads(output)  # shape (b,t,k)


# Conv Transforme Block
class ConvTransformerBLock(nn.Module):
    def __init__(self, k, headers, kernel_size=5, mask_next=True, mask_diag=False, dropout_proba=0.2):
        super().__init__()

        # Self attention
        self.attention = SelfAttentionConv(k, headers, kernel_size, mask_next, mask_diag)

        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
        # Dropout funtcion  & Relu:
        self.dropout = nn.Dropout(p=dropout_proba)
        self.activation = nn.ReLU()

    def forward(self, x, train=False):
        # Self attention + Residual
        x = self.attention(x) + x

        # Dropout attention
        if train:
            x = self.dropout(x)

        # First Normalization
        x = self.norm1(x)

        # Feed Froward network + residual
        x = self.feedforward(x) + x

        # Second Normalization
        x = self.norm2(x)

        return x


# Forcasting Conv Transformer :
class ForcastConvTransformer(nn.Module):
    def __init__(self, k, headers, depth, seq_length, kernel_size=5, mask_next=True, mask_diag=False, dropout_proba=0.2,
                 num_tokens=None):
        super().__init__()
        # Embedding
        self.tokens_in_count = False
        if num_tokens:
            self.tokens_in_count = True
            self.token_embedding = nn.Embedding(num_tokens, k)  # （369, 1）= (nb_ts, k)

        # Embedding the position
        self.position_embedding = nn.Embedding(seq_length, k)   # (500, 1) = (windows_size, k)

        # Number of kind of time series
        self.k = k  # 没有协变量的情况下，k=1
        self.seq_length = seq_length    # seq_length即窗口大小, 数据准备的时候切割好了

        # Transformer blocks
        tblocks = []
        # log sparse 稀疏策略： 采用多层ConvTrans层堆叠的方式
        for t in range(depth):
            tblocks.append(ConvTransformerBLock(k, headers, kernel_size, mask_next, mask_diag, dropout_proba))
        self.TransformerBlocks = nn.Sequential(*tblocks)

        # Transformation from k dimension to numClasses
        self.topreSigma = nn.Linear(k, 1)
        self.tomu = nn.Linear(k, 1)
        self.plus = nn.Softplus()

    def forward(self, x, tokens=None):
        b, t, k = x.size()

        # checking that the given batch had same number of time series as the BLock had
        assert k == self.k, 'The k :' + str(
            self.k) + ' number of timeseries given in the initialization is different than what given in the x :' + str(
            k)
        assert t == self.seq_length, 'The lenght of the timeseries given t ' + str(
            t) + ' miss much with the lenght sequence given in the Tranformers initialisation self.seq_length: ' + str(
            self.seq_length)

        # Position embedding
        pos = torch.arange(t)
        self.pos_emb = self.position_embedding(pos).expand(b, t, k)

        # Checking token embedding
        assert self.tokens_in_count == (not (tokens is None)), 'self.tokens_in_count = ' + str(
            self.tokens_in_count) + ' should be equal to (not (tokens is None)) = ' + str((not (tokens is None)))
        if not (tokens is None):
            ## checking that the number of tockens corresponde to the number of batch elements
            assert tokens.size(0) == b
            self.tok_emb = self.token_embedding(tokens)
            self.tok_emb = self.tok_emb.expand(t, b, k).transpose(0, 1)

        # Adding Pos Embedding and token Embedding to the variable
        if not (tokens is None):
            x = self.pos_emb + self.tok_emb + x
        else:
            x = self.pos_emb + x

        # Transformer :
        x = self.TransformerBlocks(x)
        mu = self.tomu(x)
        presigma = self.topreSigma(x)
        sigma = self.plus(presigma)

        return mu, sigma
    

class convMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super(convMultiHeadAttention, self).__init__()
            
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head

        self.kernel_size = 5
        padding = (self.kernel_size - 1)
        self.padding_opertor = nn.ConstantPad1d((padding, 0), 0)

        self.dropout = nn.Dropout(p=dropout)
        
        # self.v_layer = nn.Linear(self.d_model, self.d_v, bias = False)
        # self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q, bias = False) 
        #                                for _ in range(self.n_head)])
        # self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias = False) 
        #                                for _ in range(self.n_head)])
        # self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])

        self.v_layer = nn.Conv1d(self.d_model, self.d_v, kernel_size=1, padding=0, bias=False)
        self.q_layers = nn.ModuleList([nn.Conv1d(self.d_model, self.d_q, self.kernel_size, padding=0, bias=True) 
                                       for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Conv1d(self.d_model, self.d_k, self.kernel_size, padding=0, bias=True)
                                       for _ in range(self.n_head)])
        self.v_layers = nn.ModuleList([self.v_layer for _ in range(self.n_head)])

        self.attention = ScaledDotProductAttention()
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.padding_opertor(q)
        k = self.padding_opertor(k)
        #v = self.padding_opertor(v)

        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            
            qs = qs.transpose(1, 2)
            ks = ks.transpose(1, 2)
            vs = vs.transpose(1, 2)
            head, attn = self.attention(qs, ks, vs, mask)
            #print('head layer: {}'.format(head.shape))
            #print('attn layer: {}'.format(attn.shape))
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
            
        head = torch.stack(heads, dim = 2) if self.n_head > 1 else heads[0]
        #print('concat heads: {}'.format(head.shape))
        #print('heads {}: {}'.format(0, head[0,0,Ellipsis]))
        attn = torch.stack(attns, dim = 2)
        #print('concat attn: {}'.format(attn.shape))
        
        outputs = torch.mean(head, dim = 2) if self.n_head > 1 else head
        #print('outputs mean: {}'.format(outputs.shape))
        #print('outputs mean {}: {}'.format(0, outputs[0,0,Ellipsis]))
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        
        return outputs, attn

    
class convattentionLayer(nn.Module):
    def __init__(self, args):
        super(convattentionLayer, self).__init__()
        self.n_heads = args.n_heads
        self.hidden_layer_size = args.hidden_layer_size
        self.dropout_rate = args.dropout_rate
        self.d_ff = args.d_ff

        self.self_attn_layer = convMultiHeadAttention(n_head = self.n_heads, d_model = self.hidden_layer_size,dropout = self.dropout_rate)
        # self.self_attn_layer = AttentionLayer(ProbAttention(False, 5, attention_dropout=self.dropout_rate,output_attention=True),
        #                                            self.n_heads,self.d_model,self.dropout_rate)
        
        self.post_attn_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.hidden_layer_size, bias=False))
        
        self.post_attn_gate_add_norm1 = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def forward(self, input):
        x,attn = self.self_attn_layer(input,input,input)
        x = self.post_attn_gate_add_norm(x, input)

        x_fc = self.fc(x)
        x_fc = self.post_attn_gate_add_norm1(x_fc,x)
        return x_fc

class convAttention(nn.Module):
    def __init__(self, args):
        super(convAttention, self).__init__()
        self.n_layers = args.n_layers
        self.layers = nn.ModuleList([convattentionLayer(args) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == '__main__':

    batch_size = 128
    hidden = 256
    length = 20

    args = CtTransformer_args_parser()
    attn_layer = convAttention(args)
    input = torch.randn([batch_size,length,hidden])
    out = attn_layer(input)
    
