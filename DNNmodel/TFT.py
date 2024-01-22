from unicodedata import bidirectional
from torch import nn
import torch
import math
import torch.nn.functional as F
from args import *
from data_process import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GLU
class GatedLinearUnit(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate,
                 activation = None):
        
        super(GatedLinearUnit, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)
            
        self.W4 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        self.W5 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        
        if self.activation_name:
            self.activation = getattr(nn, self.activation_name)()
            
        self.sigmoid = nn.Sigmoid()
            
        self.init_weights()
            
    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' not in n:
                torch.nn.init.xavier_uniform_(p)
#                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif 'bias' in n:
                torch.nn.init.zeros_(p)
            
    def forward(self, x):
        
        if self.dropout_rate:
            x = self.dropout(x)
            
        if self.activation_name:
            output = self.sigmoid(self.W4(x)) * self.activation(self.W5(x))
        else:
            output = self.sigmoid(self.W4(x)) * self.W5(x)
            
        return output


class GateAddNormNetwork(nn.Module):
    def __init__(self, input_size,
                 hidden_layer_size,
                 dropout_rate,
                 activation = None):
        
        super(GateAddNormNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        
        self.GLU = GatedLinearUnit(self.input_size, 
                                   self.hidden_layer_size, 
                                   self.dropout_rate,
                                   activation = self.activation_name)
        
        self.LayerNorm = nn.LayerNorm(self.hidden_layer_size)
        
    def forward(self, x, skip):
        
        output = self.LayerNorm(self.GLU(x) + skip)
            
        return output



class GatedResidualNetwork(nn.Module):
    def __init__(self,
                 hidden_layer_size,
                 input_size = None,
                 output_size = None, 
                 dropout_rate = None, 
                 additional_context = None,
                 return_gate = False):
        
        super(GatedResidualNetwork, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size if input_size else self.hidden_layer_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
        self.return_gate = return_gate
        
        self.W1 = torch.nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.W2 = torch.nn.Linear(self.input_size, self.hidden_layer_size)
        
        if self.additional_context:
            self.W3 = torch.nn.Linear(self.additional_context, self.hidden_layer_size, bias = False)
            

        if self.output_size:
            self.skip_linear = torch.nn.Linear(self.input_size, self.output_size)
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.output_size,
                                                   self.dropout_rate)
        else:
            self.glu_add_norm = GateAddNormNetwork(self.hidden_layer_size,
                                                   self.hidden_layer_size,
                                                   self.dropout_rate)
            
        self.init_weights()
            
    def init_weights(self):
        for name, p in self.named_parameters():
            if ('W2' in name or 'W3' in name) and 'bias' not in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif ('skip_linear' in name or 'W1' in name) and 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
#                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif 'bias' in name:
                torch.nn.init.zeros_(p)
            
    def forward(self, x):
        
        if self.additional_context:
            x, context = x
            #x_forward = self.W2(x)
            #context_forward = self.W3(context)
            #print(self.W3(context).shape)
            n2 = F.elu(self.W2(x) + self.W3(context))
        else:
            n2 = F.elu(self.W2(x))
        
        #print('n2 shape {}'.format(n2.shape))
            
        n1 = self.W1(n2)
        
        #print('n1 shape {}'.format(n1.shape))
            
        if self.output_size:
            output = self.glu_add_norm(n1, self.skip_linear(x))
        else:
            output = self.glu_add_norm(n1, x)
            
        #print('output shape {}'.format(output.shape))
        
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout = 0, scale = True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim = 2)
        self.scale = scale
            
    def forward(self, q, k, v, mask = None):
        #print('---Inputs----')
        #print('q: {}'.format(q[0]))
        #print('k: {}'.format(k[0]))
        #print('v: {}'.format(v[0]))
        
        attn = torch.bmm(q, k.permute(0,2,1))
        #print('first bmm')
        #print(attn.shape)
        #print('attn: {}'.format(attn[0]))
        
        if self.scale:
            dimention = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimention
        #    print('attn_scaled: {}'.format(attn[0]))
            
        if mask is not None:
            #fill = torch.tensor(-1e9).to(DEVICE)
            #zero = torch.tensor(0).to(DEVICE)
            attn = attn.masked_fill(mask == 0, -1e9)
        #    print('attn_masked: {}'.format(attn[0]))
            
        attn = self.softmax(attn)
        #print('attn_softmax: {}'.format(attn[0]))
        attn = self.dropout(attn)
        
        output = torch.bmm(attn, v)
        
        return output, attn

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super(InterpretableMultiHeadAttention, self).__init__()
            
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
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            vs = self.v_layers[i](v)
            #print('qs layer: {}'.format(qs.shape))
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

class VariableSelectionNetwork(nn.Module):
    def __init__(self, hidden_layer_size,
                 dropout_rate,
                 output_size,
                 input_size = None, 
                 additional_context = None):
        super(VariableSelectionNetwork, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
            
        self.flattened_grn = GatedResidualNetwork(self.hidden_layer_size,
                                                   input_size = self.input_size,
                                                   output_size = self.output_size,
                                                   dropout_rate = self.dropout_rate,
                                                   additional_context=self.additional_context)
        
        self.per_feature_grn = nn.ModuleList([GatedResidualNetwork(self.hidden_layer_size,
                                                                   dropout_rate=self.dropout_rate)
                                                      for i in range(self.output_size)])
    def forward(self, x):
        # embedding:[batch,encoder_steps,hidden,group]
        embedding = x

        time_steps = embedding.shape[1]
        # flatten:[batch,encoder_steps,hidden*group]
        flatten = embedding.view(-1, time_steps, self.hidden_layer_size * self.output_size)
        # mlp_outputs:[batch,encoder_steps,output_size(group)]
        mlp_outputs = self.flattened_grn(flatten)
        sparse_weights = F.softmax(mlp_outputs, dim = -1)
        # sparse_weights:[batch,encoder_steps,1,output_size(group)]
        sparse_weights = sparse_weights.unsqueeze(2)
        
        trans_emb_list = []
        for i in range(self.output_size):
            # e:[batch,encoder_steps,hidden]
            e = self.per_feature_grn[i](embedding[:,:,:,i])
            trans_emb_list.append(e)
        # transformed_embedding:[batch,encoder_steps,hidden,group]
        transformed_embedding = torch.stack(trans_emb_list, axis=-1)
        # combined:[batch,encoder_steps,hidden,group]
        combined = sparse_weights * transformed_embedding
        # temporal_ctx:[batch,encoder_steps,hidden]
        temporal_ctx = torch.sum(combined, dim = -1)

        return temporal_ctx, sparse_weights



class NormalizedQuantileLossCalculator():
    """Computes the combined quantile loss for prespecified quantiles.
    Attributes:
      quantiles: Quantiles to compute losses
    """

    def __init__(self, quantiles, output_size):
        """Initializes computer with quantiles for loss calculations.
            Args:
            quantiles: Quantiles to use for computations.
        """
        self.quantiles = quantiles
        self.output_size = output_size
        
    # Loss functions.
    def apply(self, y, y_pred, quantile):
        """ Computes quantile loss for pytorch.
            Standard quantile loss as defined in the "Training Procedure" section of
            the main TFT paper
            Args:
            y: Targets
            y_pred: Predictions
            quantile: Quantile to use for loss calculations (between 0 & 1)
            Returns:
            Tensor for quantile loss.
        """

        # Checks quantile
        if quantile < 0 or quantile > 1:
            raise ValueError(
                'Illegal quantile value={}! Values should be between 0 and 1.'.format(quantile))

        prediction_underflow = y - y_pred
#         print('prediction_underflow')
#         print(prediction_underflow.shape)
        weighted_errors = quantile * torch.max(prediction_underflow, torch.zeros_like(prediction_underflow)) + \
                (1. - quantile) * torch.max(-prediction_underflow, torch.zeros_like(prediction_underflow))
        
        quantile_loss = torch.mean(weighted_errors)
        normaliser = torch.mean(torch.abs(quantile_loss))
        return 2 * quantile_loss / normaliser



    # regular_inputs = all_inputs[:, :, :self.num_regular_variables].to(torch.float)
    # unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(regular_inputs, categorical_inputs)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, args):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_layer_size = args.hidden_layer_size
        self.num_variables = args.input_size
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.num_non_static_historical_inputs = args.input_size
        self.num_non_static_future_inputs = args.input_size
        self.output_size = 1
        self.num_encoder_steps = args.num_encoder_steps
        self.quantiles = [0.1, 0.5, 0.9]


        self.build_embeddings()
        self.build_static_context_networks()
        self.build_variable_selection_networks()
        self.build_lstm()
        self.build_post_lstm_gate_add_norm()
        self.build_static_enrichment()
        self.build_temporal_self_attention()
        self.build_position_wise_feed_forward()
        self.build_output_feed_forward()
        self.init_weights()


    def init_weights(self):
        for name, p in self.named_parameters():
            if ('lstm' in name and 'ih' in name) and 'bias' not in name:
                #print(name)
                #print(p.shape)
                torch.nn.init.xavier_uniform_(p)
#                 torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='sigmoid')
            elif ('lstm' in name and 'hh' in name) and 'bias' not in name:
        
                 torch.nn.init.orthogonal_(p)
            
            elif 'lstm' in name and 'bias' in name:
                #print(name)
                #print(p.shape)
                torch.nn.init.zeros_(p)

    def build_embeddings(self):
        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1, self.hidden_layer_size) for i in range(self.num_variables)])

    def build_static_context_networks(self):
        self.static_context_variable_selection_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)
        
        self.static_context_enrichment_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)

        self.static_context_state_h_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)
        
        self.static_context_state_c_grn = GatedResidualNetwork(self.hidden_layer_size,dropout_rate=self.dropout_rate)

    def build_variable_selection_networks(self):
        # self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,input_size = self.hidden_layer_size *self.num_non_static_historical_inputs,
        #                                                         output_size = self.num_non_static_historical_inputs,
        #                                                         dropout_rate = self.dropout_rate,
        #                                                         additional_context=self.hidden_layer_size)

        self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,input_size = self.hidden_layer_size *self.num_non_static_historical_inputs,
                                                                output_size = self.num_non_static_historical_inputs,dropout_rate = self.dropout_rate)
        
        self.temporal_future_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,input_size = self.hidden_layer_size *self.num_non_static_future_inputs,
                                                            output_size = self.num_non_static_future_inputs, dropout_rate = self.dropout_rate)

    def build_lstm(self):
        self.historical_lstm = nn.LSTM(input_size = self.hidden_layer_size,hidden_size = self.hidden_layer_size,batch_first = True,bidirectional=True)

        self.future_lstm = nn.LSTM(input_size = self.hidden_layer_size,hidden_size = self.hidden_layer_size,batch_first = True,bidirectional=True)

        self.lstm_change = nn.Linear(2*self.hidden_layer_size,self.hidden_layer_size)

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(self.hidden_layer_size,dropout_rate = self.dropout_rate)

    def build_temporal_self_attention(self):
        self.self_attn_layer = InterpretableMultiHeadAttention(n_head = self.num_heads, d_model = self.hidden_layer_size,dropout = self.dropout_rate)
        
        self.post_attn_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(self.hidden_layer_size, dropout_rate = self.dropout_rate)
        
        self.post_tfd_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def build_output_feed_forward(self):
        self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size * len(self.quantiles))

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.
        Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1]
        bs = self_attn_inputs.shape[0]
        mask = torch.cumsum(torch.eye(len_s), 0)
        mask = mask.repeat(bs,1,1).to(torch.float32)
        # mask:[batch,len,len]
        # [1,0,0,0]
        return mask.to(device)

    def get_tft_embeddings(self, input):

        known_regular_inputs = [self.regular_var_embeddings[i](input[:,:,i:i + 1]) for i in range(self.num_variables)]
        # len(known_regular_inputs):group
        # known_regular_inputs[0]:[batch,len,hidden]

        # known_combined_layer = torch.stack(known_regular_inputs,axis=-1)
        # known_combined_layer:[batch,len,hidden,group]
        known_combined_layer = torch.stack(known_regular_inputs,axis=-1)
        # print(known_combined_layer.shape)
        
        return known_combined_layer


    def forward(self, input):
        # input:[batch,len,group]
        input = input.to(device)
        batch_size = input.shape[0]
        # input_embedding:[batch,len,hidden,group]
        input_embedding = self.get_tft_embeddings(input)
        # historical_inputs :[batch,encoder_steps,hidden,group]
        historical_inputs = input_embedding[:,:self.num_encoder_steps,:]
        # future_inputs:[batch,len - encoder_steps,hidden,group]
        future_inputs = input_embedding[:,self.num_encoder_steps:,:]
        # historical_features:[batch,encoder_steps,hidden]
        # historical_flags:[batch,encoder_steps,1,group]
        historical_features, historical_flags = self.temporal_historical_vsn(historical_inputs)     

        # future_features:[batch,len - encoder_steps,hidden]
        # future_flags:[batch,len - encoder_steps,1,group]
        future_features, future_flags  = self.temporal_future_vsn(future_inputs)

        h_0 = torch.randn(2, batch_size, self.hidden_layer_size).to(device)
        c_0 = torch.randn(2, batch_size, self.hidden_layer_size).to(device)

        # history_lstm:[batch,encoder_steps,hidden]
        # future_lstm:[batch,len - encoder_steps,hidden]
        history_lstm, (state_h, state_c) = self.historical_lstm(historical_features,(h_0,c_0))
        future_lstm, _ = self.future_lstm(future_features,(state_h,state_c))

        history_lstm = self.lstm_change(history_lstm)
        future_lstm = self.lstm_change(future_lstm)
        # input_embeddings:[batch,len,hidden]
        # lstm_layer:[batch,len,hidden]
        input_embeddings = torch.cat((historical_features, future_features), axis=1)
        lstm_layer = torch.cat((history_lstm, future_lstm), axis=1)
        # temporal_feature_layer:[batch,len,hidden]
        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(lstm_layer, input_embeddings)
        # enriched:[batch,len,hidden]
        enriched = self.static_enrichment(temporal_feature_layer)
        # x:[batch,len.hidden]
        # self_att:[batch,len,n_head,len]
        x, self_att = self.self_attn_layer(enriched, enriched, enriched,mask = self.get_decoder_mask(enriched))
        # x:[batch,len.hidden]
        x = self.post_attn_gate_add_norm(x, enriched)
        # decoder:[batch,len.hidden]
        decoder = self.GRN_positionwise(x)
        
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)
        # outputs:[batch,len - encoder_steps,quantiles]
        outputs = self.output_feed_forward(transformer_layer[:, self.num_encoder_steps:, :])

        return outputs

        


        




# if __name__ == '__main__':
#     Inum = 10
#     smooth = False

#     fold_data = "..\\..\\results"

#     net = ["ER"]
#     d = [10]
#     R = [2,2.5]

#     args = TFT_args_parser()
#     Dtr,Dva,Dtr_all,Dva_all = get_Train_range(args,fold_data,net,d,R,Inum,smooth)
#     model =TemporalFusionTransformer(args).to(device)

#     quantiles = [0.1, 0.5, 0.9]
#     loss_function = QuantileLoss(quantiles).to(device)

#     for (seq, label) in Dtr:
        
#         output = model(seq)
#         output = output[:,:,:].view(-1,3)
#         output = output[:,1:2]
#         print(output.shape)
#         # label = label.flatten()
        
#         # loss = loss_function(output,label)
        






    








