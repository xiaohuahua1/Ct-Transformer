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
import multiprocessing

if __name__ == '__main__':

    test_path = "..\\results\\HM\\seq2seq\\test"
    val_path = "..\\results\\HM\\seq2seq\\val"
    train_path = "..\\results\\HM\\seq2seq\\train"

    path_seq2seq_CtRt = "model\\seq2seq.pkl"
    path_seq2seq_Rt = "model\\seq2seq_Rt.pkl"
    path_seq2seq_roll = "model\\seq2seq_Ct.pkl"
    path_ann_CtRt = "model\\ann_seq.pkl"
    path_ann_CR_pair = "model\\ann_one.pkl"
    path_cnn_CtRt = "model\\cnn_seq.pkl"
    path_cnn_CR_pair = "model\\cnn_one.pkl"
    path_cnn_lstm_CtRt = "model\\cnn_lstm_seq.pkl"
    path_seq2seq_Ct = "model\\seq2seq_onlyCt.pkl"
    path_ann_Ct = "model\\ann_onlyCt_seq.pkl"


    args_seq2seq_CtRt = seq2seq_CtRt_args_parser()
    args_seq2seq_Rt = seq2seq_Rt_args_parser()
    args_seq2seq_roll = seq2seq_CtRt_args_parser()
    args_ann_CtRt = ann_CtRt_args_parser()
    args_ann_CR_pair = ann_CR_pair_args_parser()
    args_cnn_CtRt = cnn_CtRt_args_parser()
    args_cnn_CR_pair = cnn_CR_pair_args_parser()
    args_cnn_lstm_CtRt =  cnn_lstm_CtRt_args_parser()
    args_seq2seq_Ct = seq2seq_Ct_args_parser()
    args_ann_Ct = ann_Ct_args_parser()

    Dtr_seq2seq_CtRt, Val_seq2seq_CtRt, m_seq2seq_CtRt, n_seq2seq_CtRt = getTrainData(args_seq2seq_CtRt,train_path,val_path)
    Dtr_seq2seq_Rt, Val_seq2seq_Rt, m_seq2seq_Rt, n_seq2seq_Rt = getTrainData(args_seq2seq_Rt,train_path,val_path)
    Dtr_seq2seq_roll, Val_seq2seq_roll, m_seq2seq_roll, n_seq2seq_roll = getTrainData(args_seq2seq_roll,train_path,val_path)
    Dtr_ann_CtRt, Val_ann_CtRt, m_ann_CtRt, n_ann_CtRt = getTrainData(args_ann_CtRt,train_path,val_path)
    Dtr_ann_CR_pair, Val_ann_CR_pair, m_ann_CR_pair, n_ann_CR_pair, = getTrainData(args_ann_CR_pair,train_path,val_path)
    Dtr_cnn_CtRt, Val_cnn_CtRt, m_cnn_CtRt, n_cnn_CtRt, = getTrainData(args_cnn_CtRt,train_path,val_path)
    Dtr_cnn_CR_pair, Val_cnn_CR_pair, m_cnn_CR_pair, n_cnn_CR_pair, = getTrainData(args_cnn_CR_pair,train_path,val_path)
    Dtr_cnn_lstm_CtRt, Val_cnn_lstm_CtRt, m_cnn_lstm_CtRt, n_cnn_lstm_CtRt = getTrainData(args_cnn_lstm_CtRt,train_path,val_path)
    Dtr_seq2seq_Ct, Val_seq2seq_Ct, m_seq2seq_Ct, n_seq2seq_Ct = getTrainData(args_seq2seq_Ct,train_path,val_path)
    Dtr_ann_Ct, Val_ann_Ct, m_ann_Ct, n_ann_Ct = getTrainData(args_ann_Ct,train_path,val_path)

    model_seq2seq_CtRt = Seq2Seq(args_seq2seq_CtRt).to(device)
    model_seq2seq_Rt = Seq2Seq(args_seq2seq_Rt).to(device)
    model_seq2seq_roll = Seq2Seq_Ct(args_seq2seq_roll).to(device)
    model_ann_CtRt = ANN(args_ann_CtRt).to(device)
    model_ann_CR_pair = ANN(args_ann_CR_pair).to(device)
    model_cnn_CtRt = CNN(args_cnn_CtRt).to(device)
    model_cnn_CR_pair = CNN(args_cnn_CR_pair).to(device)
    model_cnn_lstm_CtRt = CNN_LSTM(args_cnn_lstm_CtRt).to(device)
    model_seq2seq_Ct = Seq2Seq(args_seq2seq_Ct).to(device)
    model_ann_Ct = ANN(args_ann_Ct).to(device)


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
                seq = seq.to(device)
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
                seq = seq.to(device)
                label = label.to(device)
                y_pred = model(seq)
                loss = loss_function(y_pred, label)
                val_loss.append(loss.item())

        return np.mean(val_loss)


    trainModel(args_seq2seq_CtRt,model_seq2seq_CtRt,Dtr_seq2seq_CtRt,Val_seq2seq_CtRt,path_seq2seq_CtRt)
    trainModel(args_seq2seq_Rt,model_seq2seq_Rt,Dtr_seq2seq_Rt,Val_seq2seq_Rt,path_seq2seq_Rt)
    trainModel(args_seq2seq_roll,model_seq2seq_roll,Dtr_seq2seq_roll,Val_seq2seq_roll,path_seq2seq_roll)
    trainModel(args_ann_CtRt,model_ann_CtRt,Dtr_ann_CtRt,Val_ann_CtRt,path_ann_CtRt)
    trainModel(args_ann_CR_pair,model_ann_CR_pair,Dtr_ann_CR_pair,Val_ann_CR_pair,path_ann_CR_pair)
    trainModel(args_cnn_CtRt,model_cnn_CtRt,Dtr_cnn_CtRt,Val_cnn_CtRt,path_cnn_CtRt)
    trainModel(args_cnn_CR_pair,model_cnn_CR_pair,Dtr_cnn_CR_pair,Val_cnn_CR_pair,path_cnn_CR_pair)
    trainModel(args_cnn_lstm_CtRt,model_cnn_lstm_CtRt,Dtr_cnn_lstm_CtRt,Val_cnn_lstm_CtRt,path_cnn_lstm_CtRt)
    trainModel(args_seq2seq_Ct,model_seq2seq_Ct,Dtr_seq2seq_Ct,Val_seq2seq_Ct,path_seq2seq_Ct)
    trainModel(args_ann_Ct,model_ann_Ct,Dtr_ann_Ct,Val_ann_Ct,path_ann_Ct)

    process_seq2seq_roll = multiprocessing.Process(target=trainModel,args=[args_seq2seq_roll,model_seq2seq_roll,Dtr_seq2seq_roll,Val_seq2seq_roll,path_seq2seq_roll])


    



