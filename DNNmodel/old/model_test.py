from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process import device, setup_seed
from models import *

setup_seed(20)
plt.rcParams["text.usetex"] = False
plt.style.use('ggplot')

def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))


def seq2seq_test(args, Dte, path, m, n,l,type):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')
    
    if type == 'seq2seq':
        model = Seq2Seq(args).to(device)
    elif type == 'seq2seq_ct':
        model = Seq2Seq_Ct(args).to(device)
    # model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')

    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))

        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    start = 0
    end = 0
    for i in range(len(l)):
        step = l[i]
        end = start + step

        pred_step = pred[start:end]
        y_step = y[start:end]

        pred_step,y_step = np.array(pred_step),np.array(y_step)
        pred_step = (m-n)*pred_step + n
        y_step = (m-n)*y_step + n

        print('mape:', get_mape(y_step, pred_step))
        # plot
        plot(y_step, pred_step)
        start = end


def ann_test(args, Dte, path, m, n,l,type):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')
    

    model = ANN(args).to(device)
    
    # model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')

    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))

        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    start = 0
    end = 0
    for i in range(len(l)):
        step = l[i]
        end = start + step

        pred_step = pred[start:end]
        y_step = y[start:end]

        pred_step,y_step = np.array(pred_step),np.array(y_step)
        pred_step = (m-n)*pred_step + n
        y_step = (m-n)*y_step + n

        print('mape:', get_mape(y_step, pred_step))
        # plot
        plot(y_step, pred_step)
        start = end

    
def cnn_test(args, Dte, path, m, n,l,type):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')
    

    model = CNN(args).to(device)
    
    # model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')

    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))

        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    start = 0
    end = 0
    for i in range(len(l)):
        step = l[i]
        end = start + step

        pred_step = pred[start:end]
        y_step = y[start:end]

        pred_step,y_step = np.array(pred_step),np.array(y_step)
        pred_step = (m-n)*pred_step + n
        y_step = (m-n)*y_step + n

        print('mape:', get_mape(y_step, pred_step))
        # plot
        plot(y_step, pred_step)
        start = end


def cnn_lstm_test(args, Dte, path, m, n,l,type):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')

    model = CNN_LSTM(args).to(device)
    
    # model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')

    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))

        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    
    start = 0
    end = 0
    for i in range(len(l)):
        step = l[i]
        end = start + step

        pred_step = pred[start:end]
        y_step = y[start:end]

        pred_step,y_step = np.array(pred_step),np.array(y_step)
        pred_step = (m-n)*pred_step + n
        y_step = (m-n)*y_step + n

        print('mape:', get_mape(y_step, pred_step))
        # plot
        plot(y_step, pred_step)
        start = end

def compare(args, Dte_Rt, Dte_Ct,Rt_path,Ct_path, m, n,l):
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred_Ct = []
    pred_Rt= []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model_Ct = Seq2Seq(7, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model_Ct.load_state_dict(torch.load(Ct_path)['models'])
    model_Ct.eval()

    model_Rt = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model_Rt.load_state_dict(torch.load(Rt_path)['models'])
    model_Rt.eval()

    print('predicting...')

    for (seq, target) in tqdm(Dte_Rt):
        target = list(chain.from_iterable(target.data.tolist()))

        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred_Rt = model_Rt(seq)
            y_pred_Rt = list(chain.from_iterable(y_pred_Rt.data.tolist()))
            pred_Rt.extend(y_pred_Rt)

    for (seq, target) in tqdm(Dte_Ct):
        
        seq = seq.to(device)
        with torch.no_grad():
            y_pred_Ct = model_Ct(seq)
            y_pred_Ct = list(chain.from_iterable(y_pred_Ct.data.tolist()))
            pred_Ct.extend(y_pred_Ct)
    
    start = 0
    end = 0
    for i in range(len(l)):
        print(i)
        step = l[i]
        end = start + step

        pred_Ct_step = pred_Ct[start:end]
        pred_Rt_step = pred_Rt[start:end]
        y_step = y[start:end]

        pred_Ct_step,pred_Rt_step,y_step = np.array(pred_Ct_step),np.array(pred_Rt_step),np.array(y_step)

        pred_Ct_step = (m-n)*pred_Ct_step + n
        pred_Rt_step = (m-n)*pred_Rt_step + n
        y_step = (m-n)*y_step + n

        # print('mape:', get_mape(y_step, pred_step))
        # plot
        plot1(y_step, pred_Ct_step,pred_Rt_step)
        start = end




        


def plot(y, pred):
    # plot
    # x = [i for i in range(1, 150 + 1)]
    # # print(len(y))
    # x_smooth = np.linspace(np.min(x), np.max(x), 500)
    # y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    #
    # y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    # 只画出测试集中前1000个点
    # plt.plot(y[:100], c='green', label='true')
    # plt.plot(pred[:100], c='red', label='pred')
    plt.plot(y, c='green', label='true')
    plt.plot(pred, c='red', label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def plot1(y, pred1,pred2):
    plt.plot(y, c='green', label='true')
    plt.plot(pred1, c='red', label='pred_Ct')
    plt.plot(pred2, c='blue', label='pred_Rt')
    plt.grid(axis='y')
    plt.legend()
    plt.show()




