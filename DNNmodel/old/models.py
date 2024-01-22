from torch import nn
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        batch_size = input_seq.shape[0]
        input_seq = input_seq.view(batch_size, 1, self.input_size)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output)  # pred(batch_size, 1, output_size)
        pred = pred[:, -1, :]

        return pred, h, c


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.output_size = args.output_size
        self.Encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size)
        self.Decoder = Decoder(self.input_size, self.hidden_size, self.num_layers, self.output_size, self.batch_size)

    def forward(self, input_seq):
        input_seq = input_seq.to(device)
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, seq_len, self.output_size).to(device)
        for t in range(seq_len):
            _input = input_seq[:, t, :]
            output, h, c = self.Decoder(_input, h, c)
            outputs[:, t, :] = output

        return outputs[:, -1, :]
    

class Seq2Seq_Ct(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.output_size = args.output_size
        self.Encoder = Encoder(self.input_size, self.hidden_size, self.num_layers, self.batch_size)
        self.Decoder = Decoder(1, self.hidden_size, self.num_layers, 1, self.batch_size)

    def forward(self, input_seq):
        input_seq = input_seq.to(device)
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)

        outputs = torch.zeros(batch_size,self.output_size,1).to(device)

        Rt = input_seq[:, -1, : 1]
        for t in range(self.output_size):
            output, h, c = self.Decoder(Rt, h, c)
            # print(output.shape)
            outputs[:,t] = output
            Rt = output
        # print(outputs.shape)
        outputs = outputs.permute(0,2,1)
        outputs = torch.squeeze(outputs)
        # print(outputs.shape)
        return outputs

class ANN(nn.Module):
    def __init__(self,args):
        super(ANN, self).__init__()
        self.seq_len = args.seq_len
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.nn = torch.nn.Sequential(
            nn.Linear(self.seq_len * self.input_size, 128),
            torch.nn.ReLU(),
            nn.Linear(128, 256),
            torch.nn.ReLU(),
            nn.Linear(256, 128),
            torch.nn.ReLU(),
            nn.Linear(128, self.output_size)
        )

    def forward(self, x):
        x = x.to(device)
        # print(x.shape)
        # x(batch_size, seq_len, input_size)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.nn(x)
        return x
    

class CNN(nn.Module):
    def __init__(self,args):
        self.in_channels = args.in_channels
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.seq_len = args.seq_len
        self.output_size = args.output_size
        super(CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=self.kernel_size),  # 24 - 2 + 1 = 23
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride),  # 23 - 2 + 1 = 22
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size),  # 22 - 2 + 1 = 21
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride),  # 21 - 2 + 1 = 20
        )
        self.Linear1 = nn.Linear(128 * (self.seq_len - 4*(self.kernel_size - self.stride)), 50)
        self.Linear2 = nn.Linear(50, self.output_size)

    def forward(self, x):
        x = x.to(device)
        if len(x.shape) == 2:
            x = torch.unsqueeze(x,1)
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.size())  # 15 127 20
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        x = x.view(x.shape[0], -1)

        return x


class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=args.in_channels, out_channels=args.out_channels, kernel_size=args.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=args.kernel_size, stride=args.stride)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=args.out_channels, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.output_size)

    def forward(self, x):
        x = x.to(device)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x[:, -1, :]

        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))

        """
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
        """
        pe = torch.zeros(max_len, dim)  # max_len 是解码器生成句子的最长的长度，假设是 10
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))


        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.drop_out = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):

        emb = emb * math.sqrt(self.dim)

        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.drop_out(emb)
        return emb


# class Transformer(nn.Module):
#     def __init__(self, args):
#         super(Transformer, self).__init__()
#         self.args = args
#         self.embed = nn.Linear(args.input_size, args.d_model)
#         self.pos_emb = PositionalEncoding(args.d_model,0.1)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=args.d_model,
#             nhead=8,
#             dim_feedforward=4 * args.d_model,
#             batch_first=True,
#             dropout=0.1,
#             device=args.device
#         )
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=args.d_model,
#             nhead=8,
#             dropout=0.1,
#             dim_feedforward=4 * args.d_model,
#             batch_first=True,
#             device=args.device
#         )
#         self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=5)
#         self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=5)
#         self.fc1 = nn.Linear(args.seq_len * args.d_model, args.d_model)
#         self.fc2 = nn.Linear(args.d_model, args.output_size)

#     def forward(self, x):
#         # (128, 5, 1)
#         x = self.embed(x)  # (128, 5, 32)
#         x = self.pos_emb(x)   # (128, 5, 32)
#         x = self.encoder(x)
#         # 不经过解码器
#         x = x.flatten(start_dim=1)
#         x = self.fc1(x)
#         out = self.fc2(x)
#         # y = self.output_fc(y)   # (256, 4, 128)
#         # out = self.decoder(y, x)  # (256, 4, 128)
#         # out = out.flatten(start_dim=1)  # (256, 4 * 128)
#         # out = self.fc(out)  # (256, 4)

#         return out

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.embed = nn.Linear(args.input_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model,0.1)
        self.trans = nn.Transformer(d_model=args.d_model,batch_first=True)
        self.fc = nn.Linear(args.d_model,1)

    def forward(self, x):
        src = x[0].to(device)
        tgt = x[1].to(device)
        tgt = tgt.unsqueeze(2)

        src = self.embed(src)
        tgt = self.embed(tgt)

        src = self.pos_emb(src)
        tgt = self.pos_emb(tgt)
        print(src.shape)

        # torch.Size([128, 2, 32])
        out = self.trans(src,tgt)
        # print(type(src))
        # print(tgt)
        out = self.fc(out)
        out = out.squeeze(2)

        return out
    
class Transformer_Ct(nn.Module):
    def __init__(self, args):
        super(Transformer_Ct, self).__init__()
        self.embeds = nn.Linear(args.input_size, args.d_model)
        self.embedt = nn.Linear(1, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model,0.1)
        self.trans = nn.Transformer(d_model=args.d_model,num_encoder_layers=3, num_decoder_layers=3,batch_first=True)
        self.fc = nn.Linear(args.d_model,1)

    def forward(self, x):
        src = x[0].to(device)
        tgt = x[1].to(device)
        tgt = tgt.unsqueeze(2)
        # print(tgt.shape)

        src = self.embeds(src)
        tgt = self.embedt(tgt)

        src = self.pos_emb(src)
        tgt = self.pos_emb(tgt)

        # torch.Size([128, 2, 32])
        out = self.trans(src,tgt)
        # print(type(src))
        # print(tgt)
        out = self.fc(out)
        out = out.squeeze(2)

        return out

class Transformer_RtCt(nn.Module):
    def __init__(self, args):
        super(Transformer_RtCt, self).__init__()
        self.embed = nn.Linear(args.input_size, args.d_model)
        self.pos_emb = PositionalEncoding(args.d_model,0.1)
        self.trans = nn.Transformer(d_model=args.d_model,batch_first=True)
        self.fc = nn.Linear(args.d_model,1)

    def forward(self, x):
        src = x[0].to(device)
        tgt = x[1].to(device)
        tgt = tgt.unsqueeze(2)
        ct = x[2].to(device)

        ct = ct.view(ct.shape[0],-1)
        ct = ct.unsqueeze(2)

        ctrt = torch.cat([src,ct],dim=1)
        print(ctrt.shape)

        src = self.embed(src)
        tgt = self.embed(tgt)

        src = self.pos_emb(src)
        tgt = self.pos_emb(tgt)
        # print(src.shape)

        # torch.Size([128, 2, 32])
        out = self.trans(src,tgt)
        # print(type(src))
        # print(tgt)
        out = self.fc(out)
        out = out.squeeze(2)

        return out


