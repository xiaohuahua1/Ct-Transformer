from torch import nn
import torch
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMmodel(nn.Module):
    def __init__(self,args):
        super(LSTMmodel, self).__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.num_directions = 1
        self.batch_size = args.batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size,1)
    def forward(self, input_seq):
        input_seq = input_seq.to(device)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        output = self.drop(output)
        output = self.fc(output)
        # output = output[:,-1]
        output = output.squeeze(2)

        return output

class ANN(nn.Module):
    def __init__(self,args):
        super(ANN, self).__init__()
        self.seq_len = args.seq_len
        self.input_size = args.input_size
        # self.output_size = args.output_size
        self.nn = torch.nn.Sequential(
            nn.Linear(self.seq_len * self.input_size, 512),
            torch.nn.ReLU(),
            nn.Linear(512, 1024),
            torch.nn.ReLU(),
            nn.Linear(1024, 512),
            torch.nn.ReLU(),
            nn.Linear(512, self.seq_len)
        )

    def forward(self, x):
        x = x.to(device)
        # print(x.shape)
        # x(batch_size, seq_len, input_size)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.nn(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
        [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])                  # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])                  # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).to(device)               # enc_inputs: [seq_len, d_model]
    def forward(self,enc_inputs):                                         # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1),:]
        return self.dropout(enc_inputs.to(device))

class transformer_encoder(nn.Module):
    def __init__(self, args):
        super(transformer_encoder, self).__init__()
        self.embeds = nn.Linear(args.input_size, args.d_model)
        # self.embedt = nn.Linear(1, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model,0.1)
        # self.trans = nn.Transformer(d_model=args.d_model,num_encoder_layers=4, num_decoder_layers=4,batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.n_layers)
        self.fc = nn.Linear(args.d_model,args.output_size)

    def forward(self, src):

        src = src.to(device)
        

        src = self.embeds(src)

        src = self.pos_emb(src)

        out = self.transformer_encoder(src)
        
        out = self.fc(out)

        # out = out.squeeze(2)
        
        return out
    
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.embeds = nn.Linear(args.input_size, args.d_model)
        self.embedt = nn.Linear(1, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model,0.1)
        self.trans = nn.Transformer(d_model=args.d_model,num_encoder_layers=args.n_layers, num_decoder_layers=args.n_layers,batch_first=True)
        self.fc = nn.Linear(args.d_model,1)

    def forward(self, src,tgt):
        src = src.to(device)
        # start = torch.ones(src.shape[0],1,1).to(device)
        tgt = tgt.to(device)
        tgt_len = tgt.shape[1]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)
        tgt_mask = tgt_mask.to(device)

        src = self.embeds(src)
        tgt = self.embedt(tgt)

        src = self.pos_emb(src)
        tgt = self.pos_emb(tgt)

        out = self.trans(src,tgt,tgt_mask=tgt_mask)
    
        return out

class HighdemensionLoss(nn.Module):

    def __init__(self, output_size,loss_type):
        ##takes a list of quantiles
        super().__init__()
        self.output_size = output_size
        self.loss = nn.SmoothL1Loss().to(device)
        if loss_type == 1:
            self.loss = nn.MSELoss().to(device)

    def forward(self, preds, target):
        # assert not target.requires_grad
        # assert preds.size(0) == target.size(0)#检验程序使用的，如果不满足条件，程序会自动退出
        loss_list = []
        for i in range(self.output_size):
            preds_i = preds[:,:,i]
            target_i = target[:,:,i]
            loss = self.loss(preds_i,target_i)
            loss_list.append(loss.unsqueeze(0))
        # print(torch.sum(torch.cat(loss_list, dim=0),dim=0))
        # print(torch.mean(torch.sum(torch.cat(loss_list, dim=0),dim=0)))
        return torch.sum(torch.cat(loss_list, dim=0),dim=0)