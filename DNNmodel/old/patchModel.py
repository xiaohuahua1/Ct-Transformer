from torch import nn
import torch
import math
from tqdm import tqdm

from model import *
from data_process import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class TSTiEncoder(nn.Module):
    def __init__(self, args):
        super(TSTiEncoder, self).__init__()
        self.d_model = args.d_model
        self.patch_len = args.patch_len
        self.dropout_rate = args.dropout_rate
        self.W_P = nn.Linear(self.patch_len, self.d_model) 
        self.dropout = nn.Dropout(self.dropout_rate)
        self.pos_emb = PositionalEncoding(args.d_model,0.1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=args.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.n_layers)


    def forward(self, x):
        # x:[batch,input_size,patch_len,patch_num]
        input_size = x.shape[1]
        # x:[batch,input_size,patch_num,patch_len]
        x = x.permute(0,1,3,2) 
        # x:[batch,input_size,patch_num,d_model]
        x = self.W_P(x)
        # u:[batch*input_size,patch_num,d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))  
        u = self.dropout(u)

        u = self.pos_emb(u)
        # out:[batch*input_size,patch_num,d_model]
        out = self.transformer_encoder(u)
        # out:[batch,input_size,patch_num,d_model]
        out = torch.reshape(out, (-1,input_size,out.shape[-2],out.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        # out:[batch,input_size,d_model,patch_num]
        out = out.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return out 

class Flatten_Head(nn.Module):
    def __init__(self, args):
        super(Flatten_Head, self).__init__() 
        self.input_size = args.input_size
        self.patch_len = args.patch_len
        self.d_model = args.d_model
        self.head_dropout = args.head_dropout
        self.output_size = args.output_size

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.flattens = nn.ModuleList()

        for i in range(self.input_size):
            # self.flattens.append(nn.Flatten(start_dim=-2))
            self.linears.append(nn.Linear(self.d_model, self.patch_len))
            self.dropouts.append(nn.Dropout(self.head_dropout))
        self.fc = nn.Linear(self.input_size,self.output_size)

    def forward(self, x):
        # x:[batch,input_size,d_model,patch_num]
        batch_size = x.shape[0]
        # x:[batch,input_size,patch_num,d_model]
        x = x.permute(0,1,3,2)
        x_out = []
        for i in range(self.input_size):
            # z:[batch,patch_num,patch_len]
            z = self.linears[i](x[:,i,:,:]) 
            z = self.dropouts[i](z)
            x_out.append(z)
        # x:[batch,input_size,patch_num,patch_len]
        x = torch.stack(x_out, dim=1)
        # x:[batch,input_size,patch_num*patch_len]
        x = x.view(batch_size,self.input_size,-1)
        # x:[batch,patch_num*patch_len,input_size]
        x = x.permute(0,2,1)
        # x:[batch,patch_num*patch_len,out_size]
        x = self.fc(x)

        return x
        


class PatchTST(nn.Module):
    def __init__(self, args):
        super(PatchTST, self).__init__()
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.input_size = args.input_size
        
        self.affine = True
        self.subtract_last = False
        self.revin_layer = RevIN(self.input_size, affine=self.affine, subtract_last=self.subtract_last)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.transformer_encoder = TSTiEncoder(args)
        self.flatten = Flatten_Head(args)



    def forward(self, x):
        x = x.to(device)
        # x:[batch,len,input_size]
        length = x.shape[1]
        patch_num = int((length - self.patch_len)/self.stride + 1)
        # x:[batch,len,input_size]
        # x = self.revin_layer(x, 'norm')
        # x:[batch,input_size,len]
        x = x.permute(0,2,1)
        # x:[batch,input_size,len+stride]
        x = self.padding_patch_layer(x)
        # x:[batch,input_size,patch_num,patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        # x:[batch,input_size,patch_len,patch_num]
        x = x.permute(0,1,3,2) 
        # x:[batch,input_size,d_model,patch_num]
        x = self.transformer_encoder(x)
        # x[batch,patch_num*patch_len,out_size]
        x = self.flatten(x)

        x = x[:,:length,:]
        return x

if __name__ == '__main__':

    fold_data = "..\\..\\results\\ER\\d=10R=2"
    
    val_path = fold_data + "\\val"
    train_path = fold_data + "\\train"
    


    Inum = 10
    smooth = False

    args = patchTST_args_parser()
    train_data,train_label,_ = read_train_data(train_path,args,Inum,smooth)
    Dtr = create_train_dataset(args,train_data,train_label,0)
    model = PatchTST(args).to(device)

    quantiles = [0.1, 0.5, 0.9]
    loss_function = QuantileLoss(quantiles).to(device)



    for(seq,label) in tqdm(Dtr):
        # print(seq.shape)
        # print(target.shape)
        y_pred = model(seq)
        y_pred = y_pred[:,:,:].contiguous().view(-1,3)
        label = label.flatten()
        
        loss = loss_function(y_pred, label)