from unicodedata import bidirectional
from torch import nn
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F
from args import *
from data_process import *
from myModel_unit import *
from myModel_patch import *
from loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictHead(nn.Module):
    def __init__(self, d_model, patch_len, output_size,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len*output_size)
        self.patch_len = patch_len

    def forward(self, x):
        # x:[batch,patch_num,hidden]
        batch,patch_num,hidden = x.shape
        # x:[batch,patch_num,patch_len*output_size]
        x = self.linear( self.dropout(x) ) 
        # x:[batch,patch_num,patch_len,output_size]
        x = x.view(batch,patch_num,self.patch_len,-1)
        # x:[batch,patch_num*patch_len,output_size]
        outputs = outputs.view(batch,patch_num*self.patch_len,-1)
        
        return x
    
class noPatchHead(nn.Module):
    def __init__(self, d_model, output_size,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x:[batch,len,hidden]
        batch,patch_num,hidden = x.shape
        # x:[batch,len,output_size]
        x = self.linear( self.dropout(x) ) 
        
        return x

class NoPatch(nn.Module):
    def __init__(self, args,head_type = "prediction"):
        super(NoPatch, self).__init__()
        self.hidden_layer_size = args.hidden_layer_size
        self.d_model = args.hidden_layer_size
        self.input_size = args.input_size
        self.dropout_rate = args.dropout_rate
        self.d_ff = args.d_ff
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.output_size = args.output_size
        self.num_encoder_steps = args.num_encoder_steps
        self.patchLen = args.patchLen
        self.head_type = head_type
        # self.quantiles = [0.1, 0.5, 0.9]

        self.attn_layer = Attention(args)

        self.build_embeddings()
        self.build_static_context_networks()
        self.build_variable_selection_networks()
        self.build_lstm()
        self.build_post_lstm_gate_add_norm()
        self.build_static_enrichment()
        # self.build_temporal_self_attention()
        self.build_position_wise_feed_forward()
        self.build_output_head()
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
        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1, self.hidden_layer_size) for i in range(self.input_size*self.patchLen)])

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

        self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,input_size = self.hidden_layer_size *self.input_size,
                                                                output_size = self.input_size,dropout_rate = self.dropout_rate)


    def build_lstm(self):
        self.historical_lstm = nn.GRU(input_size = self.hidden_layer_size,hidden_size = self.hidden_layer_size,batch_first = True,bidirectional=True)



        self.lstm_change = nn.Linear(2*self.hidden_layer_size,self.hidden_layer_size)

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(self.hidden_layer_size,dropout_rate = self.dropout_rate)


    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(self.hidden_layer_size, dropout_rate = self.dropout_rate)
        
        self.post_tfd_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    # def build_output_feed_forward(self):
    #     # self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size * len(self.quantiles)*self.patchLen)
    #     self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size*self.patchLen)

    def build_output_head(self):
        self.head = noPatchHead(self.d_model,self.output_size,self.dropout_rate)

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

        known_regular_inputs = [self.regular_var_embeddings[i](input[:,:,i:i + 1]) for i in range(self.input_size)]
        known_combined_layer = torch.stack(known_regular_inputs,axis=-1)
        # print(known_combined_layer.shape)
        
        return known_combined_layer

    def forward(self,input):
        # input:[batch,plen,input_size]
        input = input.to(device)
        batch_size = input.shape[0]
        
        input_embedding = self.get_tft_embeddings(input)

        features, flags = self.temporal_historical_vsn(input_embedding) 

        h_0 = torch.randn(2, batch_size, self.hidden_layer_size).to(device)

        history_lstm, state_h = self.historical_lstm(features,h_0)

        history_lstm = self.lstm_change(history_lstm)

        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(history_lstm, features)
        enriched = self.static_enrichment(temporal_feature_layer)

        x = self.attn_layer(enriched)

        decoder = self.GRN_positionwise(x)
        # transformer_layer:[batch,patch_num,hidden]
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)

        outputs = self.head(transformer_layer)

        if self.head_type == "prediction":
            outputs = outputs[:,self.num_encoder_steps:,:]

        if self.output_size == 1:
            outputs = outputs.squeeze(2)

        return outputs
    
class NoGRU(nn.Module):
    def __init__(self, args,head_type = "prediction"):
        super(NoGRU, self).__init__()
        self.hidden_layer_size = args.hidden_layer_size
        self.d_model = args.hidden_layer_size
        self.input_size = args.input_size
        self.dropout_rate = args.dropout_rate
        self.d_ff = args.d_ff
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.output_size = args.output_size
        self.num_encoder_steps = args.num_encoder_steps
        self.patchLen = args.patchLen
        self.head_type = head_type
        # self.quantiles = [0.1, 0.5, 0.9]

        self.attn_layer = Attention(args)

        self.build_embeddings()
        self.build_static_context_networks()
        self.build_variable_selection_networks()
        self.build_lstm()
        self.build_post_lstm_gate_add_norm()
        self.build_static_enrichment()
        # self.build_temporal_self_attention()
        self.build_position_wise_feed_forward()
        self.build_output_head()
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
        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1, self.hidden_layer_size) for i in range(self.input_size*self.patchLen)])

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

        self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,input_size = self.hidden_layer_size *self.input_size*self.patchLen,
                                                                output_size = self.input_size*self.patchLen,dropout_rate = self.dropout_rate)


    def build_lstm(self):
        self.historical_lstm = nn.LSTM(input_size = self.hidden_layer_size,hidden_size = self.hidden_layer_size,batch_first = True,bidirectional=False)

        # self.lstm_change = nn.Linear(2*self.hidden_layer_size,self.hidden_layer_size)

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(self.hidden_layer_size,dropout_rate = self.dropout_rate)


    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(self.hidden_layer_size, dropout_rate = self.dropout_rate)
        
        self.post_tfd_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    # def build_output_feed_forward(self):
    #     # self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size * len(self.quantiles)*self.patchLen)
    #     self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size*self.patchLen)

    def build_output_head(self):
        
        self.head = PredictHead(self.hidden_layer_size,self.patchLen,self.output_size,self.dropout_rate)

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

        known_regular_inputs = [self.regular_var_embeddings[i](input[:,:,i:i + 1]) for i in range(self.input_size*self.patchLen)]
        known_combined_layer = torch.stack(known_regular_inputs,axis=-1)
        # print(known_combined_layer.shape)
        
        return known_combined_layer

    def forward(self,input,length):
        # input:[batch,patch_num,patch_len*input_size]
        batch_size = input.shape[0]
        
        input_embedding = self.get_tft_embeddings(input)

        features, flags = self.temporal_historical_vsn(input_embedding) 

        h_0 = torch.randn(1, batch_size, self.hidden_layer_size).to(device)
        c_0 = torch.randn(1, batch_size, self.hidden_layer_size).to(device)

        history_lstm, _ = self.historical_lstm(features,(h_0,c_0))

        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(history_lstm, features)
        enriched = self.static_enrichment(temporal_feature_layer)

        x = self.attn_layer(enriched)

        decoder = self.GRN_positionwise(x)
        # transformer_layer:[batch,patch_num,hidden]
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)

        outputs = self.head(transformer_layer)

        if self.head_type == "prediction":
            outputs = outputs[:,self.num_encoder_steps:length,:]

        if self.output_size == 1:
            outputs = outputs.squeeze(2)

        return outputs
    
class NoattenForward(nn.Module):
    def __init__(self, args,head_type = "prediction"):
        super(NoattenForward, self).__init__()
        self.hidden_layer_size = args.hidden_layer_size
        self.d_model = args.hidden_layer_size
        self.input_size = args.input_size
        self.dropout_rate = args.dropout_rate
        self.d_ff = args.d_ff
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.output_size = args.output_size
        self.num_encoder_steps = args.num_encoder_steps
        self.patchLen = args.patchLen
        self.head_type = head_type
        # self.quantiles = [0.1, 0.5, 0.9]

        self.attn_layer = Attention(args,isfc=False)

        self.build_embeddings()
        self.build_static_context_networks()
        self.build_variable_selection_networks()
        self.build_lstm()
        self.build_post_lstm_gate_add_norm()
        self.build_static_enrichment()
        # self.build_temporal_self_attention()
        self.build_position_wise_feed_forward()
        self.build_output_head()
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
        self.regular_var_embeddings = nn.ModuleList([nn.Linear(1, self.hidden_layer_size) for i in range(self.input_size*self.patchLen)])

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

        self.temporal_historical_vsn = VariableSelectionNetwork(hidden_layer_size = self.hidden_layer_size,input_size = self.hidden_layer_size *self.input_size*self.patchLen,
                                                                output_size = self.input_size*self.patchLen,dropout_rate = self.dropout_rate)


    def build_lstm(self):
        self.historical_lstm = nn.GRU(input_size = self.hidden_layer_size,hidden_size = self.hidden_layer_size,batch_first = True,bidirectional=True)



        self.lstm_change = nn.Linear(2*self.hidden_layer_size,self.hidden_layer_size)

    def build_post_lstm_gate_add_norm(self):
        self.post_seq_encoder_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    def build_static_enrichment(self):
        self.static_enrichment = GatedResidualNetwork(self.hidden_layer_size,dropout_rate = self.dropout_rate)


    def build_position_wise_feed_forward(self):
        self.GRN_positionwise = GatedResidualNetwork(self.hidden_layer_size, dropout_rate = self.dropout_rate)
        
        self.post_tfd_gate_add_norm = GateAddNormNetwork(self.hidden_layer_size,self.hidden_layer_size,self.dropout_rate,activation = None)

    # def build_output_feed_forward(self):
    #     # self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size * len(self.quantiles)*self.patchLen)
    #     self.output_feed_forward = torch.nn.Linear(self.hidden_layer_size, self.output_size*self.patchLen)

    def build_output_head(self):
        if self.head_type == "prediction":
            self.head = PredictHead(self.hidden_layer_size,self.patchLen,self.output_size,self.dropout_rate)

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

        known_regular_inputs = [self.regular_var_embeddings[i](input[:,:,i:i + 1]) for i in range(self.input_size*self.patchLen)]
        known_combined_layer = torch.stack(known_regular_inputs,axis=-1)
        # print(known_combined_layer.shape)
        
        return known_combined_layer

    def forward(self,input,length):
        # input:[batch,patch_num,patch_len*input_size]
        batch_size = input.shape[0]
        
        input_embedding = self.get_tft_embeddings(input)

        features, flags = self.temporal_historical_vsn(input_embedding) 

        h_0 = torch.randn(2, batch_size, self.hidden_layer_size).to(device)

        history_lstm, state_h = self.historical_lstm(features,h_0)

        history_lstm = self.lstm_change(history_lstm)

        temporal_feature_layer = self.post_seq_encoder_gate_add_norm(history_lstm, features)
        enriched = self.static_enrichment(temporal_feature_layer)

        x = self.attn_layer(enriched)

        decoder = self.GRN_positionwise(x)
        # transformer_layer:[batch,patch_num,hidden]
        transformer_layer = self.post_tfd_gate_add_norm(decoder, temporal_feature_layer)

        outputs = self.head(transformer_layer)

        if self.head_type == "prediction":
            outputs = outputs[:,self.num_encoder_steps:length,:]

        if self.output_size == 1:
            outputs = outputs.squeeze(2)

        return outputs
