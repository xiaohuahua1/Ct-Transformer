from data_process import *
from args import *
from models import *
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import copy
from transformer import *

if __name__ == '__main__':

    Inum = 10
    smooth = True

    def getType(type):
        path_data = ""
        path_model = ""
        text = []
        r = []
        d = []
        num = 1
        if type == 'SF':
            path_data = "..\\results\\SF\\DNN"
            path_model = "model\\SF"
            r = [1.3,1.5,2.0,3.0]
            d = [8,10,12]
        elif type == 'ER':
            path_data = "..\\results\\ER\\DNN\\changeDR"
            path_model = "model\\ER\\changeDR"
            r = [1.3,1.5,2.0,2.5]
            d = [6,8,10,12]
        elif type == 'd=8R=2':
            path_data = "..\\results\\SF\\d=8R=2"
            path_model = "model\\SF\\d=8R=2"
            r = [2.0]
            d = [8]
            num = 10
        elif type == 'SF changeD':
            path_data  = "..\\results\\SF\\changeD"
            path_model = "model\\SF\\changeD"
            d = [4,6,8,10,12]
            r = [1.5,2.0]
            num = 2

        for i in range(len(d)):
            for j in range(len(r)):
                for z in range(num):
                    text.append([d[i],r[j],z])



        return path_data,path_model,text
    
    type = "d=8R=2"
    path_data,path_model,text = getType(type)

    test_path = path_data + "\\test"
    val_path = path_data + "\\val"
    train_path = path_data + "\\train"

    # path_seq2seq_roll = path_model + "\\seq2seq_CtRt.pkl"
    # path_ann_CtRt = path_model + "\\ann_CtRt.pkl"
    # path_cnn_CtRt = path_model + "\\cnn_CtRt.pkl"
    # path_cnn_lstm_CtRt = path_model + "\\cnn_lstm_CtRt.pkl"
    # path_transformer_Rt = path_model + "\\transformer_Rt.pkl"


    # args_seq2seq_roll = seq2seq_CtRt_args_parser()
    # args_ann_CtRt = ann_CtRt_args_parser()
    # args_cnn_CtRt = cnn_CtRt_args_parser()
    # args_cnn_lstm_CtRt =  cnn_lstm_CtRt_args_parser()
    # args_transformer_Rt = transformer_Rt_args_parser()

    # Dtr_seq2seq_roll, Val_seq2seq_roll, m_seq2seq_roll, n_seq2seq_roll = getTrainData(args_seq2seq_roll,train_path,val_path,[])
    # Dtr_ann_CtRt, Val_ann_CtRt, m_ann_CtRt, n_ann_CtRt = getTrainData(args_ann_CtRt,train_path,val_path,[])
    # Dtr_cnn_CtRt, Val_cnn_CtRt, m_cnn_CtRt, n_cnn_CtRt, = getTrainData(args_cnn_CtRt,train_path,val_path,[])
    # Dtr_cnn_lstm_CtRt, Val_cnn_lstm_CtRt, m_cnn_lstm_CtRt, n_cnn_lstm_CtRt = getTrainData(args_cnn_lstm_CtRt,train_path,val_path,[])
    # Dtr_transformer_Rt, Val_transformer_Rt, m_transformer_Rt, n_transformer_Rt = getTrainData(args_transformer_Rt,train_path,val_path,[])

    # model_seq2seq_roll = Seq2Seq_Ct(args_seq2seq_roll).to(device)
    # model_ann_CtRt = ANN(args_ann_CtRt).to(device)
    # model_cnn_CtRt = CNN(args_cnn_CtRt).to(device)
    # model_cnn_lstm_CtRt = CNN_LSTM(args_cnn_lstm_CtRt).to(device)
    # model_transformer_Rt = Transformer(args_transformer_Rt).to(device)


    def trainModel(args,model,Dtr,Val,path):
        optimizer = args.optimizer
        weight_decay = args.weight_decay
        lr = args.lr
        step_size = args.step_size
        gamma = args.gamma
        epochs = args.epochs
        loss_function = nn.MSELoss().to(device)

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=0.9, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        # training
        min_epochs = 10
        best_model = None
        min_val_loss = 5
        for epoch in tqdm(range(epochs)):
            train_loss = []
            for (seq, label) in Dtr:
                # seq = seq.to(device)
                label = label.to(device)
                y_pred = model(seq)
                loss = loss_function(y_pred, label)
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            # validation
            val_loss = get_val_loss(model, Val)
            if epoch > min_epochs and val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(model)

            print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
            model.train()

        state = {'models': best_model.state_dict()}
        torch.save(state, path)

    def get_val_loss(model, Val):
        model.eval()
        loss_function = nn.MSELoss().to(device)
        val_loss = []
        for (seq, label) in Val:
            with torch.no_grad():
                # seq = seq.to(device)
                label = label.to(device)
                y_pred = model(seq)
                loss = loss_function(y_pred, label)
                val_loss.append(loss.item())

        return np.mean(val_loss)


    def train(model_type):
        args = []
        Dtr = []
        Val = []
        model = []
        path = []

        if model_type == "transformer_Rt":
            path = path_model + "\\transformer_Rt.pkl"
            args = transformer_Rt_args_parser()
            Dtr, Val, m, n = getTrainData(args,train_path,val_path,[],Inum,smooth)
            model = Transformer(args).to(device)

        elif model_type == "transformer_Rt_ANN_Ct":
            path = path_model + "\\transformer_Rt_ANN_Ct.pkl"
            args = transformer_Rt_ANN_Ct_args_parser()
            Dtr, Val, m, n = getTrainData(args,train_path,val_path,[],Inum,smooth)
            model = Transformer_ANN(args).to(device)

        trainModel(args,model,Dtr,Val,path)


    # trainModel(args_seq2seq_roll,model_seq2seq_roll,Dtr_seq2seq_roll,Val_seq2seq_roll,path_seq2seq_roll)
    # trainModel(args_ann_CtRt,model_ann_CtRt,Dtr_ann_CtRt,Val_ann_CtRt,path_ann_CtRt)
    # trainModel(args_cnn_CtRt,model_cnn_CtRt,Dtr_cnn_CtRt,Val_cnn_CtRt,path_cnn_CtRt)
    # trainModel(args_cnn_lstm_CtRt,model_cnn_lstm_CtRt,Dtr_cnn_lstm_CtRt,Val_cnn_lstm_CtRt,path_cnn_lstm_CtRt)
    # trainModel(args_transformer_Rt,model_transformer_Rt,Dtr_transformer_Rt,Val_transformer_Rt,path_transformer_Rt)
    train("transformer_Rt")

    # process_seq2seq_roll = multiprocessing.Process(target=trainModel,args=[args_seq2seq_roll,model_seq2seq_roll,Dtr_seq2seq_roll,Val_seq2seq_roll,path_seq2seq_roll])


    



