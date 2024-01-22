from data_process import *
from args import *
from models import *
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math

def test(Dte,model):
    y = []
    pred = []

    for(seq,target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)
    return y, pred

def search(pat, txt):#找到pat在txt子串的第一次出现位置
        i, N = 0, len(txt)
        j, M = 0, len(pat)
        while i < N and j < M:
            if txt[i] == pat[j]:
                j = j + 1
            else:
                i -= j
                j = 0
            i = i + 1
        if j == M:
            return i - M
        else:
            return -1

def mse(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    for i in range(num):
        result += math.pow((pred[i] - y[i]),2)
    result /= num
    return result

if __name__ == '__main__':

    
    
    test_path_te = "..\\results\\HM\\seq2seq\\testNC\\process\\test"
    test_path_tr = "..\\results\\HM\\seq2seq\\testNC\\process\\train"
    
    val_path = "..\\results\\HM\\seq2seq\\val"
    train_path = "..\\results\\HM\\seq2seq\\train"
    test_path = "..\\results\\HM\\seq2seq\\testNC"

    start1 = 37
    mid = 90
    mid1 = 120
    end1 = 169

    truth_tr,regress_tr,mean_tr,low_tr,upper_tr = readNC(test_path_tr)
    truth_te,regress_te,mean_te,low_te,upper_te = readNC(test_path_te)

    truth_tr = truth_tr[10:-1]
    mean_tr = mean_tr[10:-1]
    upper_tr = upper_tr[10:-1]
    low_tr  = low_tr[10:-1]

    truth_te = truth_te[10:]
    mean_te = mean_te[10:]
    upper_te = upper_te[10:]
    low_te = low_te[10:]
    

    path_seq2seq_CtRt = "model\\HM\\seq2seq.pkl"
    path_seq2seq_Rt = "model\\HM\\seq2seq_Rt.pkl"
    path_seq2seq_roll = "model\\HM\\seq2seq_Ct.pkl"
    path_ann_CtRt = "model\\HM\\ann_seq.pkl"
    path_ann_CR_pair = "model\\HM\\ann_one.pkl"
    path_cnn_CtRt = "model\\HM\\cnn_seq.pkl"
    path_cnn_CR_pair = "model\\HM\\cnn_one.pkl"
    path_cnn_lstm_CtRt = "model\\HM\\cnn_lstm_seq.pkl"
    path_seq2seq_Ct = "model\\HM\\seq2seq_onlyCt.pkl"
    path_ann_Ct = "model\\HM\\ann_onlyCt_seq.pkl"


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


    Dte_seq2seq_CtRt, l_seq2seq_CtRt,R0_seq2seq_CtRt = getTestData_NC(args_seq2seq_CtRt,test_path,m_seq2seq_CtRt,n_seq2seq_CtRt,start1,mid)
    Dte_seq2seq_Rt,l_seq2seq_Rt,R0_seq2seq_Rt = getTestData_NC(args_seq2seq_Rt,test_path,m_seq2seq_Rt,n_seq2seq_Rt,start1,mid)
    Dte_seq2seq_roll, l_seq2seq_roll,R0_seq2seq_roll = getTestData_NC(args_seq2seq_roll,test_path,m_seq2seq_roll,n_seq2seq_roll,start1,mid)
    Dte_ann_CtRt,l_ann_CtRt,R0_ann_CtRt = getTestData_NC(args_ann_CtRt,test_path,m_ann_CtRt,n_ann_CtRt,start1,mid)
    Dte_ann_CR_pair,l_ann_CR_pair,R0_ann_CR_pair = getTestData_NC(args_ann_CR_pair,test_path,m_ann_CR_pair,n_ann_CR_pair,start1,mid)
    Dte_cnn_CtRt,l_cnn_CtRt,R0_cnn_CtRt = getTestData_NC(args_cnn_CtRt,test_path,m_cnn_CtRt,n_cnn_CtRt,start1,mid)
    Dte_cnn_CR_pair,l_cnn_CR_pair,R0_cnn_CR_pair = getTestData_NC(args_cnn_CR_pair,test_path,m_cnn_CR_pair,n_cnn_CR_pair,start1,mid)
    Dte_cnn_lstm_CtRt,l_cnn_lstm_CtRt,R0_cnn_lstm_CtRt = getTestData_NC(args_cnn_lstm_CtRt,test_path,m_cnn_lstm_CtRt,n_cnn_lstm_CtRt,start1,mid)
    Dte_seq2seq_Ct,l_seq2seq_Ct,R0_seq2seq_Ct = getTestData_NC(args_seq2seq_Ct,test_path,m_seq2seq_Ct,n_seq2seq_Ct,start1,mid)
    Dte_ann_Ct,l_ann_Ct,R0_ann_Ct = getTestData_NC(args_ann_Ct,test_path,m_ann_Ct,n_ann_Ct,start1,mid)


    m = 0.0
    n = 0.0
    R0 = []


    m = m_seq2seq_CtRt
    n = n_seq2seq_CtRt
    R0 = R0_seq2seq_CtRt

    # print(m)
    # print(n)

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

    model_seq2seq_CtRt.load_state_dict(torch.load(path_seq2seq_CtRt)['models'])
    model_seq2seq_CtRt.eval()

    model_seq2seq_Rt.load_state_dict(torch.load(path_seq2seq_Rt)['models'])
    model_seq2seq_Rt.eval()

    model_seq2seq_roll.load_state_dict(torch.load(path_seq2seq_roll)['models'])
    model_seq2seq_roll.eval()

    model_ann_CtRt.load_state_dict(torch.load(path_ann_CtRt)['models'])
    model_ann_CtRt.eval()

    model_ann_CR_pair.load_state_dict(torch.load(path_ann_CR_pair)['models'])
    model_ann_CR_pair.eval()

    model_cnn_CtRt.load_state_dict(torch.load(path_cnn_CtRt)['models'])
    model_cnn_CtRt.eval()

    model_cnn_CR_pair.load_state_dict(torch.load(path_cnn_CR_pair)['models'])
    model_cnn_CR_pair.eval()

    model_cnn_lstm_CtRt.load_state_dict(torch.load(path_cnn_lstm_CtRt)['models'])
    model_cnn_lstm_CtRt.eval()

    model_seq2seq_Ct.load_state_dict(torch.load(path_seq2seq_Ct)['models'])
    model_seq2seq_Ct.eval()

    model_ann_Ct.load_state_dict(torch.load(path_ann_Ct)['models'])
    model_ann_Ct.eval()



    y_seq2seq_CtRt,pred_seq2seq_CtRt = test(Dte_seq2seq_CtRt,model_seq2seq_CtRt)
    y_seq2seq_Rt,pred_seq2seq_Rt = test(Dte_seq2seq_Rt,model_seq2seq_Rt)
    y_seq2seq_roll,pred_seq2seq_roll = test(Dte_seq2seq_roll,model_seq2seq_roll)
    y_ann_CtRt,pred_ann_CtRt = test(Dte_ann_CtRt,model_ann_CtRt)
    y_ann_CR_pair,pred_ann_CR_pair = test(Dte_ann_CR_pair,model_ann_CR_pair)
    y_cnn_CtRt,pred_cnn_CtRt = test(Dte_cnn_CtRt,model_cnn_CtRt)
    y_cnn_CR_pair,pred_cnn_CR_pair = test(Dte_cnn_CR_pair,model_cnn_CR_pair)
    y_cnn_lstm_CtRt,pred_cnn_lstm_CtRt = test(Dte_cnn_lstm_CtRt,model_cnn_lstm_CtRt)
    y_seq2seq_Ct,pred_seq2seq_Ct = test(Dte_seq2seq_Ct,model_seq2seq_Ct)
    y_ann_Ct,pred_ann_Ct = test(Dte_ann_Ct,model_ann_Ct)
    

    nums = len(l_seq2seq_CtRt)




    start_seq2seq_CtRt = 0
    end_seq2seq_CtRt = 0

    start_seq2seq_Rt = 0
    end_seq2seq_Rt = 0

    start_seq2seq_roll = 0
    end_seq2seq_roll = 0

    start_ann_CtRt = 0
    end_ann_CtRt = 0

    start_ann_CR_pair = 0
    end_ann_CR_pair = 0

    start_cnn_CtRt = 0
    end_cnn_CtRt = 0

    start_cnn_CR_pair = 0
    end_cnn_CR_pair = 0

    start_cnn_lstm_CtRt = 0
    end_cnn_lstm_CtRt = 0

    start_seq2seq_Ct = 0
    end_seq2seq_Ct = 0

    start_ann_Ct = 0
    end_ann_Ct = 0

    for i in range(nums):
        r0 = R0[i]

        reg = regress_tr[i][10:-1]

        step_seq2seq_CtRt = l_seq2seq_CtRt[i]
        end_seq2seq_CtRt = start_seq2seq_CtRt + step_seq2seq_CtRt
        pred_step_seq2seq_CtRt = pred_seq2seq_CtRt[start_seq2seq_CtRt:end_seq2seq_CtRt]
        y_step_seq2seq_CtRt = y_seq2seq_CtRt[start_seq2seq_CtRt:end_seq2seq_CtRt]

        step_seq2seq_Rt = l_seq2seq_Rt[i]
        end_seq2seq_Rt = start_seq2seq_Rt + step_seq2seq_Rt
        pred_step_seq2seq_Rt = pred_seq2seq_Rt[start_seq2seq_Rt:end_seq2seq_Rt]
        y_step_seq2seq_Rt = y_seq2seq_Rt[start_seq2seq_Rt:end_seq2seq_Rt]

        step_seq2seq_roll = l_seq2seq_roll[i]
        end_seq2seq_roll = start_seq2seq_roll + step_seq2seq_roll
        pred_step_seq2seq_roll = pred_seq2seq_roll[start_seq2seq_roll:end_seq2seq_roll]
        y_step_seq2seq_roll = y_seq2seq_roll[start_seq2seq_roll:end_seq2seq_roll]

        step_ann_CtRt = l_ann_CtRt[i]
        end_ann_CtRt = start_ann_CtRt + step_ann_CtRt
        pred_step_ann_CtRt = pred_ann_CtRt[start_ann_CtRt:end_ann_CtRt]
        y_step_ann_CtRt = y_ann_CtRt[start_ann_CtRt:end_ann_CtRt]

        step_ann_CR_pair = l_ann_CR_pair[i]
        end_ann_CR_pair = start_ann_CR_pair + step_ann_CR_pair
        pred_step_ann_CR_pair = pred_ann_CR_pair[start_ann_CR_pair:end_ann_CR_pair]
        y_step_ann_CR_pair = y_ann_CR_pair[start_ann_CR_pair:end_ann_CR_pair]

        step_cnn_CtRt = l_cnn_CtRt[i]
        end_cnn_CtRt = start_cnn_CtRt + step_cnn_CtRt
        pred_step_cnn_CtRt = pred_cnn_CtRt[start_cnn_CtRt:end_cnn_CtRt]
        y_step_cnn_CtRt = y_cnn_CtRt[start_cnn_CtRt:end_cnn_CtRt]

        step_cnn_CR_pair = l_cnn_CR_pair[i]
        end_cnn_CR_pair = start_cnn_CR_pair + step_cnn_CR_pair
        pred_step_cnn_CR_pair = pred_cnn_CR_pair[start_cnn_CR_pair:end_cnn_CR_pair]
        y_step_cnn_CR_pair = y_cnn_CR_pair[start_cnn_CR_pair:end_cnn_CR_pair]

        step_cnn_lstm_CtRt = l_cnn_lstm_CtRt[i]
        end_cnn_lstm_CtRt = start_cnn_lstm_CtRt + step_cnn_lstm_CtRt
        pred_step_cnn_lstm_CtRt = pred_cnn_lstm_CtRt[start_cnn_lstm_CtRt:end_cnn_lstm_CtRt]
        y_step_cnn_lstm_CtRt = y_cnn_lstm_CtRt[start_cnn_lstm_CtRt:end_cnn_lstm_CtRt]

        step_seq2seq_Ct = l_seq2seq_Ct[i]
        end_seq2seq_Ct = start_seq2seq_Ct + step_seq2seq_Ct
        pred_step_seq2seq_Ct = pred_seq2seq_Ct[start_seq2seq_Ct:end_seq2seq_Ct]
        y_step_seq2seq_Ct = y_seq2seq_Ct[start_seq2seq_Ct:end_seq2seq_Ct]

        step_ann_Ct = l_ann_Ct[i]
        end_ann_Ct = start_ann_Ct + step_ann_Ct
        pred_step_ann_Ct = pred_ann_Ct[start_ann_Ct:end_ann_Ct]
        y_step_ann_Ct = y_ann_Ct[start_ann_Ct:end_ann_Ct]


        base_len = len(y_step_cnn_CtRt)

        index_seq2seq_CtRt = search(y_step_cnn_CtRt,y_step_seq2seq_CtRt)
        pred_step_seq2seq_CtRt = pred_step_seq2seq_CtRt[index_seq2seq_CtRt:index_seq2seq_CtRt+base_len]
        y_step_seq2seq_CtRt = y_step_seq2seq_CtRt[index_seq2seq_CtRt:index_seq2seq_CtRt+base_len]

        index_seq2seq_Rt = search(y_step_cnn_CtRt,y_step_seq2seq_Rt)
        pred_step_seq2seq_Rt = pred_step_seq2seq_Rt[index_seq2seq_Rt:index_seq2seq_Rt+base_len]
        y_step_seq2seq_Rt = y_step_seq2seq_Rt[index_seq2seq_Rt:index_seq2seq_Rt+base_len]

        index_seq2seq_roll = search(y_step_cnn_CtRt,y_step_seq2seq_roll)
        pred_step_seq2seq_roll = pred_step_seq2seq_roll[index_seq2seq_roll:index_seq2seq_roll+base_len]
        y_step_seq2seq_roll = y_step_seq2seq_roll[index_seq2seq_roll:index_seq2seq_roll+base_len]

        index_ann_CtRt = search(y_step_cnn_CtRt,y_step_ann_CtRt)
        pred_step_ann_CtRt = pred_step_ann_CtRt[index_ann_CtRt:index_ann_CtRt+base_len]
        y_step_ann_CtRt = y_step_ann_CtRt[index_ann_CtRt:index_ann_CtRt+base_len]

        index_ann_CR_pair = search(y_step_cnn_CtRt,y_step_ann_CR_pair)
        pred_step_ann_CR_pair = pred_step_ann_CR_pair[index_ann_CR_pair:index_ann_CR_pair+base_len]
        y_step_ann_CR_pair = y_step_ann_CR_pair[index_ann_CR_pair:index_ann_CR_pair+base_len]

        index_cnn_CR_pair = search(y_step_cnn_CtRt,y_step_cnn_CR_pair)
        pred_step_cnn_CR_pair = pred_step_cnn_CR_pair[index_cnn_CR_pair:index_cnn_CR_pair+base_len]
        y_step_cnn_CR_pair = y_step_cnn_CR_pair[index_cnn_CR_pair:index_cnn_CR_pair+base_len]

        index_seq2seq_Ct = search(y_step_cnn_CtRt,y_step_seq2seq_Ct)
        pred_step_seq2seq_Ct = pred_step_seq2seq_Ct[index_seq2seq_Ct:index_seq2seq_Ct+base_len]
        y_step_seq2seq_Ct = y_step_seq2seq_Ct[index_seq2seq_Ct:index_seq2seq_Ct+base_len]

        index_ann_Ct = search(y_step_cnn_CtRt,y_step_ann_Ct)
        pred_step_ann_Ct = pred_step_ann_Ct[index_ann_Ct:index_ann_Ct+base_len]
        y_step_ann_Ct = y_step_ann_Ct[index_ann_Ct:index_ann_Ct+base_len]

        pred_step_seq2seq_CtRt,y_step_seq2seq_CtRt = np.array(pred_step_seq2seq_CtRt),np.array(y_step_seq2seq_CtRt)
        pred_step_seq2seq_CtRt = (m-n)*pred_step_seq2seq_CtRt + n
        y_step_seq2seq_CtRt = (m-n)*y_step_seq2seq_CtRt + n

        pred_step_seq2seq_Rt,y_step_seq2seq_Rt = np.array(pred_step_seq2seq_Rt),np.array(y_step_seq2seq_Rt)
        pred_step_seq2seq_Rt = (m-n)*pred_step_seq2seq_Rt + n
        y_step_seq2seq_Rt = (m-n)*y_step_seq2seq_Rt + n

        pred_step_seq2seq_roll,y_step_seq2seq_roll = np.array(pred_step_seq2seq_roll),np.array(y_step_seq2seq_roll)
        pred_step_seq2seq_roll = (m-n)*pred_step_seq2seq_roll + n
        y_step_seq2seq_roll = (m-n)*y_step_seq2seq_roll + n

        pred_step_ann_CtRt,y_step_ann_CtRt = np.array(pred_step_ann_CtRt),np.array(y_step_ann_CtRt)
        pred_step_ann_CtRt = (m-n)*pred_step_ann_CtRt + n
        y_step_ann_CtRt = (m-n)*y_step_ann_CtRt + n

        pred_step_ann_CR_pair,y_step_ann_CR_pair = np.array(pred_step_ann_CR_pair),np.array(y_step_ann_CR_pair)
        pred_step_ann_CR_pair = (m-n)*pred_step_ann_CR_pair + n
        y_step_ann_CR_pair = (m-n)*y_step_ann_CR_pair + n

        pred_step_cnn_CtRt,y_step_cnn_CtRt = np.array(pred_step_cnn_CtRt),np.array(y_step_cnn_CtRt)
        pred_step_cnn_CtRt = (m-n)*pred_step_cnn_CtRt + n
        y_step_cnn_CtRt = (m-n)*y_step_cnn_CtRt + n

        pred_step_cnn_CR_pair,y_step_cnn_CR_pair = np.array(pred_step_cnn_CR_pair),np.array(y_step_cnn_CR_pair)
        pred_step_cnn_CR_pair = (m-n)*pred_step_cnn_CR_pair + n
        y_step_cnn_CR_pair = (m-n)*y_step_cnn_CR_pair + n

        pred_step_cnn_lstm_CtRt,y_step_cnn_lstm_CtRt = np.array(pred_step_cnn_lstm_CtRt),np.array(y_step_cnn_lstm_CtRt)
        pred_step_cnn_lstm_CtRt = (m-n)*pred_step_cnn_lstm_CtRt + n
        y_step_cnn_lstm_CtRt = (m-n)*y_step_cnn_lstm_CtRt + n

        pred_step_ann_Ct,y_step_ann_Ct = np.array(pred_step_ann_Ct),np.array(y_step_ann_Ct)
        pred_step_ann_Ct = (m-n)*pred_step_ann_Ct + n
        y_step_ann_Ct = (m-n)*y_step_ann_Ct + n

        pred_step_seq2seq_Ct,y_step_seq2seq_Ct = np.array(pred_step_seq2seq_Ct),np.array(y_step_seq2seq_Ct)
        pred_step_seq2seq_Ct = (m-n)*pred_step_seq2seq_Ct + n
        y_step_seq2seq_Ct = (m-n)*y_step_seq2seq_Ct + n

        # # print(y_step_seq2seq_CtRt == y_step_seq2seq_Rt == y_step_seq2seq_roll == y_step_ann_CtRt == y_step_ann_CR_pair == y_step_cnn_CtRt == y_step_cnn_CR_pair == y_step_cnn_lstm_CtRt)
        # # print(y_step_seq2seq_CtRt == y_step_seq2seq_Rt)

        y = y_step_cnn_CtRt
        

        mse_seq2seq_CtRt = mse(y,pred_step_seq2seq_CtRt)
        mse_seq2seq_Rt = mse(y,pred_step_seq2seq_Rt)
        mse_seq2seq_roll = mse(y,pred_step_seq2seq_roll)
        mse_ann_CR_pair = mse(y,pred_step_ann_CR_pair)
        mse_cnn_CR_pair = mse(y,pred_step_cnn_CR_pair)
        mse_ann_CtRt = mse(y,pred_step_ann_CtRt)
        mse_cnn_CtRt = mse(y,pred_step_cnn_CtRt)
        mse_cnn_lstm_CtRt = mse(y,pred_step_cnn_lstm_CtRt)
        mse_seq2seq_Ct = mse(y,pred_step_seq2seq_Ct)
        mse_ann_Ct = mse(y,pred_step_ann_Ct)
        mse_re = mse(y,reg)
        mse_bay = mse(y,mean_tr)

        label_seq2seq_CtRt = 'seq2seq_Ct_Rt,mse=' + "{:.4f}".format(mse_seq2seq_CtRt)
        label_seq2seq_Rt = 'seq2seq_Rt,mse=' + "{:.4f}".format(mse_seq2seq_Rt)
        label_seq2seq_roll = 'seq2seq_roll,mse=' + "{:.4f}".format(mse_seq2seq_roll)
        label_ann_CR_pair = 'ann_one,mse=' + "{:.4f}".format(mse_ann_CR_pair)
        label_cnn_CR_pair = 'cnn_one,mse=' + "{:.4f}".format(mse_cnn_CR_pair)
        label_ann_CtRt = 'ann_seq,mse=' + "{:.4f}".format(mse_ann_CtRt)
        label_cnn_CtRt = 'cnn_seq,mse=' + "{:.4f}".format(mse_cnn_CtRt)
        label_cnn_lstm_CtRt = 'cnn_lstm_seq,mse=' + "{:.4f}".format(mse_cnn_lstm_CtRt)
        label_seq2seq_Ct = 'seq2seq_onlyCt,mse=' + "{:.4f}".format(mse_seq2seq_Ct)
        label_ann_Ct = 'ann_onlyCt_seq,mse=' + "{:.4f}".format(mse_ann_Ct)
        label_re = 'regress model,mse=' + "{:.4f}".format(mse_re)
        label_bay = 'bayes,mse=' + "{:.4f}".format(mse_bay)

        x = [i for i in range(start1 + 10,mid -1)]
        plt.plot(x,y, c='black', label='true')
        # plt.plot(pred_step_seq2seq_CtRt, c='darkgreen', label=label_seq2seq_CtRt)
        # plt.plot(pred_step_seq2seq_Rt, c='lightgreen', label=label_seq2seq_Rt)
        plt.plot(x,pred_step_seq2seq_roll, c='lightseagreen', label=label_seq2seq_roll)
        plt.plot(x,pred_step_ann_CtRt, c='skyblue', label=label_ann_CtRt)
        # plt.plot(pred_step_ann_CR_pair, c='tomato', label=label_ann_CR_pair)
        plt.plot(x,pred_step_cnn_CtRt, c='dodgerblue', label=label_cnn_CtRt)
        # plt.plot(pred_step_cnn_CR_pair, c='chocolate', label=label_cnn_CR_pair)
        plt.plot(x,pred_step_cnn_lstm_CtRt, c='m', label=label_cnn_lstm_CtRt)
        # plt.plot(pred_step_seq2seq_Ct, c='yellow', label=label_seq2seq_Ct)
        # plt.plot(pred_step_ann_Ct, c='greenyellow', label=label_ann_Ct)
        plt.plot(x,reg,c='purple',label=label_re)
        plt.plot(x,mean_tr,c='red',label=label_bay)
        plt.fill_between(x,low_tr,upper_tr,color='red', alpha=0.2)

        plt.grid(axis='y')
        plt.title("test = " + str(r0))
        plt.legend()
        plt.show()


        start_seq2seq_CtRt = end_seq2seq_CtRt
        start_seq2seq_Rt = end_seq2seq_Rt
        start_seq2seq_roll = end_seq2seq_roll
        start_ann_CtRt = end_ann_CtRt
        start_ann_CR_pair = end_ann_CR_pair
        start_cnn_CtRt = end_cnn_CtRt
        start_cnn_CR_pair = end_cnn_CR_pair
        start_cnn_lstm_CtRt = end_cnn_lstm_CtRt
        start_seq2seq_Ct = end_seq2seq_Ct
        start_ann_Ct = end_ann_Ct



    Dte_seq2seq_CtRt, l_seq2seq_CtRt,R0_seq2seq_CtRt = getTestData_NC(args_seq2seq_CtRt,test_path,m_seq2seq_CtRt,n_seq2seq_CtRt,mid1,end1)
    Dte_seq2seq_Rt,l_seq2seq_Rt,R0_seq2seq_Rt = getTestData_NC(args_seq2seq_Rt,test_path,m_seq2seq_Rt,n_seq2seq_Rt,mid1,end1)
    Dte_seq2seq_roll, l_seq2seq_roll,R0_seq2seq_roll = getTestData_NC(args_seq2seq_roll,test_path,m_seq2seq_roll,n_seq2seq_roll,mid1,end1)
    Dte_ann_CtRt,l_ann_CtRt,R0_ann_CtRt = getTestData_NC(args_ann_CtRt,test_path,m_ann_CtRt,n_ann_CtRt,mid1,end1)
    Dte_ann_CR_pair,l_ann_CR_pair,R0_ann_CR_pair = getTestData_NC(args_ann_CR_pair,test_path,m_ann_CR_pair,n_ann_CR_pair,mid1,end1)
    Dte_cnn_CtRt,l_cnn_CtRt,R0_cnn_CtRt = getTestData_NC(args_cnn_CtRt,test_path,m_cnn_CtRt,n_cnn_CtRt,mid1,end1)
    Dte_cnn_CR_pair,l_cnn_CR_pair,R0_cnn_CR_pair = getTestData_NC(args_cnn_CR_pair,test_path,m_cnn_CR_pair,n_cnn_CR_pair,mid1,end1)
    Dte_cnn_lstm_CtRt,l_cnn_lstm_CtRt,R0_cnn_lstm_CtRt = getTestData_NC(args_cnn_lstm_CtRt,test_path,m_cnn_lstm_CtRt,n_cnn_lstm_CtRt,mid1,end1)
    Dte_seq2seq_Ct,l_seq2seq_Ct,R0_seq2seq_Ct = getTestData_NC(args_seq2seq_Ct,test_path,m_seq2seq_Ct,n_seq2seq_Ct,mid1,end1)
    Dte_ann_Ct,l_ann_Ct,R0_ann_Ct = getTestData_NC(args_ann_Ct,test_path,m_ann_Ct,n_ann_Ct,mid1,end1)

    y_seq2seq_CtRt,pred_seq2seq_CtRt = test(Dte_seq2seq_CtRt,model_seq2seq_CtRt)
    y_seq2seq_Rt,pred_seq2seq_Rt = test(Dte_seq2seq_Rt,model_seq2seq_Rt)
    y_seq2seq_roll,pred_seq2seq_roll = test(Dte_seq2seq_roll,model_seq2seq_roll)
    y_ann_CtRt,pred_ann_CtRt = test(Dte_ann_CtRt,model_ann_CtRt)
    y_ann_CR_pair,pred_ann_CR_pair = test(Dte_ann_CR_pair,model_ann_CR_pair)
    y_cnn_CtRt,pred_cnn_CtRt = test(Dte_cnn_CtRt,model_cnn_CtRt)
    y_cnn_CR_pair,pred_cnn_CR_pair = test(Dte_cnn_CR_pair,model_cnn_CR_pair)
    y_cnn_lstm_CtRt,pred_cnn_lstm_CtRt = test(Dte_cnn_lstm_CtRt,model_cnn_lstm_CtRt)
    y_seq2seq_Ct,pred_seq2seq_Ct = test(Dte_seq2seq_Ct,model_seq2seq_Ct)
    y_ann_Ct,pred_ann_Ct = test(Dte_ann_Ct,model_ann_Ct)


    start_seq2seq_CtRt = 0
    end_seq2seq_CtRt = 0

    start_seq2seq_Rt = 0
    end_seq2seq_Rt = 0

    start_seq2seq_roll = 0
    end_seq2seq_roll = 0

    start_ann_CtRt = 0
    end_ann_CtRt = 0

    start_ann_CR_pair = 0
    end_ann_CR_pair = 0

    start_cnn_CtRt = 0
    end_cnn_CtRt = 0

    start_cnn_CR_pair = 0
    end_cnn_CR_pair = 0

    start_cnn_lstm_CtRt = 0
    end_cnn_lstm_CtRt = 0

    start_seq2seq_Ct = 0
    end_seq2seq_Ct = 0

    start_ann_Ct = 0
    end_ann_Ct = 0

    for i in range(nums):
        r0 = R0[i]

        reg = regress_te[i][10:]

        step_seq2seq_CtRt = l_seq2seq_CtRt[i]
        end_seq2seq_CtRt = start_seq2seq_CtRt + step_seq2seq_CtRt
        pred_step_seq2seq_CtRt = pred_seq2seq_CtRt[start_seq2seq_CtRt:end_seq2seq_CtRt]
        y_step_seq2seq_CtRt = y_seq2seq_CtRt[start_seq2seq_CtRt:end_seq2seq_CtRt]

        step_seq2seq_Rt = l_seq2seq_Rt[i]
        end_seq2seq_Rt = start_seq2seq_Rt + step_seq2seq_Rt
        pred_step_seq2seq_Rt = pred_seq2seq_Rt[start_seq2seq_Rt:end_seq2seq_Rt]
        y_step_seq2seq_Rt = y_seq2seq_Rt[start_seq2seq_Rt:end_seq2seq_Rt]

        step_seq2seq_roll = l_seq2seq_roll[i]
        end_seq2seq_roll = start_seq2seq_roll + step_seq2seq_roll
        pred_step_seq2seq_roll = pred_seq2seq_roll[start_seq2seq_roll:end_seq2seq_roll]
        y_step_seq2seq_roll = y_seq2seq_roll[start_seq2seq_roll:end_seq2seq_roll]

        step_ann_CtRt = l_ann_CtRt[i]
        end_ann_CtRt = start_ann_CtRt + step_ann_CtRt
        pred_step_ann_CtRt = pred_ann_CtRt[start_ann_CtRt:end_ann_CtRt]
        y_step_ann_CtRt = y_ann_CtRt[start_ann_CtRt:end_ann_CtRt]

        step_ann_CR_pair = l_ann_CR_pair[i]
        end_ann_CR_pair = start_ann_CR_pair + step_ann_CR_pair
        pred_step_ann_CR_pair = pred_ann_CR_pair[start_ann_CR_pair:end_ann_CR_pair]
        y_step_ann_CR_pair = y_ann_CR_pair[start_ann_CR_pair:end_ann_CR_pair]

        step_cnn_CtRt = l_cnn_CtRt[i]
        end_cnn_CtRt = start_cnn_CtRt + step_cnn_CtRt
        pred_step_cnn_CtRt = pred_cnn_CtRt[start_cnn_CtRt:end_cnn_CtRt]
        y_step_cnn_CtRt = y_cnn_CtRt[start_cnn_CtRt:end_cnn_CtRt]

        step_cnn_CR_pair = l_cnn_CR_pair[i]
        end_cnn_CR_pair = start_cnn_CR_pair + step_cnn_CR_pair
        pred_step_cnn_CR_pair = pred_cnn_CR_pair[start_cnn_CR_pair:end_cnn_CR_pair]
        y_step_cnn_CR_pair = y_cnn_CR_pair[start_cnn_CR_pair:end_cnn_CR_pair]

        step_cnn_lstm_CtRt = l_cnn_lstm_CtRt[i]
        end_cnn_lstm_CtRt = start_cnn_lstm_CtRt + step_cnn_lstm_CtRt
        pred_step_cnn_lstm_CtRt = pred_cnn_lstm_CtRt[start_cnn_lstm_CtRt:end_cnn_lstm_CtRt]
        y_step_cnn_lstm_CtRt = y_cnn_lstm_CtRt[start_cnn_lstm_CtRt:end_cnn_lstm_CtRt]

        step_seq2seq_Ct = l_seq2seq_Ct[i]
        end_seq2seq_Ct = start_seq2seq_Ct + step_seq2seq_Ct
        pred_step_seq2seq_Ct = pred_seq2seq_Ct[start_seq2seq_Ct:end_seq2seq_Ct]
        y_step_seq2seq_Ct = y_seq2seq_Ct[start_seq2seq_Ct:end_seq2seq_Ct]

        step_ann_Ct = l_ann_Ct[i]
        end_ann_Ct = start_ann_Ct + step_ann_Ct
        pred_step_ann_Ct = pred_ann_Ct[start_ann_Ct:end_ann_Ct]
        y_step_ann_Ct = y_ann_Ct[start_ann_Ct:end_ann_Ct]


        base_len = len(y_step_cnn_CtRt)

        index_seq2seq_CtRt = search(y_step_cnn_CtRt,y_step_seq2seq_CtRt)
        pred_step_seq2seq_CtRt = pred_step_seq2seq_CtRt[index_seq2seq_CtRt:index_seq2seq_CtRt+base_len]
        y_step_seq2seq_CtRt = y_step_seq2seq_CtRt[index_seq2seq_CtRt:index_seq2seq_CtRt+base_len]

        index_seq2seq_Rt = search(y_step_cnn_CtRt,y_step_seq2seq_Rt)
        pred_step_seq2seq_Rt = pred_step_seq2seq_Rt[index_seq2seq_Rt:index_seq2seq_Rt+base_len]
        y_step_seq2seq_Rt = y_step_seq2seq_Rt[index_seq2seq_Rt:index_seq2seq_Rt+base_len]

        index_seq2seq_roll = search(y_step_cnn_CtRt,y_step_seq2seq_roll)
        pred_step_seq2seq_roll = pred_step_seq2seq_roll[index_seq2seq_roll:index_seq2seq_roll+base_len]
        y_step_seq2seq_roll = y_step_seq2seq_roll[index_seq2seq_roll:index_seq2seq_roll+base_len]

        index_ann_CtRt = search(y_step_cnn_CtRt,y_step_ann_CtRt)
        pred_step_ann_CtRt = pred_step_ann_CtRt[index_ann_CtRt:index_ann_CtRt+base_len]
        y_step_ann_CtRt = y_step_ann_CtRt[index_ann_CtRt:index_ann_CtRt+base_len]

        index_ann_CR_pair = search(y_step_cnn_CtRt,y_step_ann_CR_pair)
        pred_step_ann_CR_pair = pred_step_ann_CR_pair[index_ann_CR_pair:index_ann_CR_pair+base_len]
        y_step_ann_CR_pair = y_step_ann_CR_pair[index_ann_CR_pair:index_ann_CR_pair+base_len]

        index_cnn_CR_pair = search(y_step_cnn_CtRt,y_step_cnn_CR_pair)
        pred_step_cnn_CR_pair = pred_step_cnn_CR_pair[index_cnn_CR_pair:index_cnn_CR_pair+base_len]
        y_step_cnn_CR_pair = y_step_cnn_CR_pair[index_cnn_CR_pair:index_cnn_CR_pair+base_len]

        index_seq2seq_Ct = search(y_step_cnn_CtRt,y_step_seq2seq_Ct)
        pred_step_seq2seq_Ct = pred_step_seq2seq_Ct[index_seq2seq_Ct:index_seq2seq_Ct+base_len]
        y_step_seq2seq_Ct = y_step_seq2seq_Ct[index_seq2seq_Ct:index_seq2seq_Ct+base_len]

        index_ann_Ct = search(y_step_cnn_CtRt,y_step_ann_Ct)
        pred_step_ann_Ct = pred_step_ann_Ct[index_ann_Ct:index_ann_Ct+base_len]
        y_step_ann_Ct = y_step_ann_Ct[index_ann_Ct:index_ann_Ct+base_len]

        pred_step_seq2seq_CtRt,y_step_seq2seq_CtRt = np.array(pred_step_seq2seq_CtRt),np.array(y_step_seq2seq_CtRt)
        pred_step_seq2seq_CtRt = (m-n)*pred_step_seq2seq_CtRt + n
        y_step_seq2seq_CtRt = (m-n)*y_step_seq2seq_CtRt + n

        pred_step_seq2seq_Rt,y_step_seq2seq_Rt = np.array(pred_step_seq2seq_Rt),np.array(y_step_seq2seq_Rt)
        pred_step_seq2seq_Rt = (m-n)*pred_step_seq2seq_Rt + n
        y_step_seq2seq_Rt = (m-n)*y_step_seq2seq_Rt + n

        pred_step_seq2seq_roll,y_step_seq2seq_roll = np.array(pred_step_seq2seq_roll),np.array(y_step_seq2seq_roll)
        pred_step_seq2seq_roll = (m-n)*pred_step_seq2seq_roll + n
        y_step_seq2seq_roll = (m-n)*y_step_seq2seq_roll + n

        pred_step_ann_CtRt,y_step_ann_CtRt = np.array(pred_step_ann_CtRt),np.array(y_step_ann_CtRt)
        pred_step_ann_CtRt = (m-n)*pred_step_ann_CtRt + n
        y_step_ann_CtRt = (m-n)*y_step_ann_CtRt + n

        pred_step_ann_CR_pair,y_step_ann_CR_pair = np.array(pred_step_ann_CR_pair),np.array(y_step_ann_CR_pair)
        pred_step_ann_CR_pair = (m-n)*pred_step_ann_CR_pair + n
        y_step_ann_CR_pair = (m-n)*y_step_ann_CR_pair + n

        pred_step_cnn_CtRt,y_step_cnn_CtRt = np.array(pred_step_cnn_CtRt),np.array(y_step_cnn_CtRt)
        pred_step_cnn_CtRt = (m-n)*pred_step_cnn_CtRt + n
        y_step_cnn_CtRt = (m-n)*y_step_cnn_CtRt + n

        pred_step_cnn_CR_pair,y_step_cnn_CR_pair = np.array(pred_step_cnn_CR_pair),np.array(y_step_cnn_CR_pair)
        pred_step_cnn_CR_pair = (m-n)*pred_step_cnn_CR_pair + n
        y_step_cnn_CR_pair = (m-n)*y_step_cnn_CR_pair + n

        pred_step_cnn_lstm_CtRt,y_step_cnn_lstm_CtRt = np.array(pred_step_cnn_lstm_CtRt),np.array(y_step_cnn_lstm_CtRt)
        pred_step_cnn_lstm_CtRt = (m-n)*pred_step_cnn_lstm_CtRt + n
        y_step_cnn_lstm_CtRt = (m-n)*y_step_cnn_lstm_CtRt + n

        pred_step_ann_Ct,y_step_ann_Ct = np.array(pred_step_ann_Ct),np.array(y_step_ann_Ct)
        pred_step_ann_Ct = (m-n)*pred_step_ann_Ct + n
        y_step_ann_Ct = (m-n)*y_step_ann_Ct + n

        pred_step_seq2seq_Ct,y_step_seq2seq_Ct = np.array(pred_step_seq2seq_Ct),np.array(y_step_seq2seq_Ct)
        pred_step_seq2seq_Ct = (m-n)*pred_step_seq2seq_Ct + n
        y_step_seq2seq_Ct = (m-n)*y_step_seq2seq_Ct + n

        # # print(y_step_seq2seq_CtRt == y_step_seq2seq_Rt == y_step_seq2seq_roll == y_step_ann_CtRt == y_step_ann_CR_pair == y_step_cnn_CtRt == y_step_cnn_CR_pair == y_step_cnn_lstm_CtRt)
        # # print(y_step_seq2seq_CtRt == y_step_seq2seq_Rt)

        y = y_step_cnn_CtRt
        
        mse_seq2seq_CtRt = mse(y,pred_step_seq2seq_CtRt)
        mse_seq2seq_Rt = mse(y,pred_step_seq2seq_Rt)
        mse_seq2seq_roll = mse(y,pred_step_seq2seq_roll)
        mse_ann_CR_pair = mse(y,pred_step_ann_CR_pair)
        mse_cnn_CR_pair = mse(y,pred_step_cnn_CR_pair)
        mse_ann_CtRt = mse(y,pred_step_ann_CtRt)
        mse_cnn_CtRt = mse(y,pred_step_cnn_CtRt)
        mse_cnn_lstm_CtRt = mse(y,pred_step_cnn_lstm_CtRt)
        mse_seq2seq_Ct = mse(y,pred_step_seq2seq_Ct)
        mse_ann_Ct = mse(y,pred_step_ann_Ct)
        mse_re = mse(y,reg)
        mse_bay = mse(y,mean_te)

        label_seq2seq_CtRt = 'seq2seq_Ct_Rt,mse=' + "{:.4f}".format(mse_seq2seq_CtRt)
        label_seq2seq_Rt = 'seq2seq_Rt,mse=' + "{:.4f}".format(mse_seq2seq_Rt)
        label_seq2seq_roll = 'seq2seq_roll,mse=' + "{:.4f}".format(mse_seq2seq_roll)
        label_ann_CR_pair = 'ann_one,mse=' + "{:.4f}".format(mse_ann_CR_pair)
        label_cnn_CR_pair = 'cnn_one,mse=' + "{:.4f}".format(mse_cnn_CR_pair)
        label_ann_CtRt = 'ann_seq,mse=' + "{:.4f}".format(mse_ann_CtRt)
        label_cnn_CtRt = 'cnn_seq,mse=' + "{:.4f}".format(mse_cnn_CtRt)
        label_cnn_lstm_CtRt = 'cnn_lstm_seq,mse=' + "{:.4f}".format(mse_cnn_lstm_CtRt)
        label_seq2seq_Ct = 'seq2seq_onlyCt,mse=' + "{:.4f}".format(mse_seq2seq_Ct)
        label_ann_Ct = 'ann_onlyCt_seq,mse=' + "{:.4f}".format(mse_ann_Ct)
        label_re = 'regress model,mse=' + "{:.4f}".format(mse_re)
        label_bay = 'bayes,mse=' + "{:.4f}".format(mse_bay)

        x = [i for i in range(mid1 + 10,end1)]
        plt.plot(x,y, c='black', label='true')
        # plt.plot(pred_step_seq2seq_CtRt, c='darkgreen', label=label_seq2seq_CtRt)
        # plt.plot(pred_step_seq2seq_Rt, c='lightgreen', label=label_seq2seq_Rt)
        plt.plot(x,pred_step_seq2seq_roll, c='lightseagreen', label=label_seq2seq_roll)
        plt.plot(x,pred_step_ann_CtRt, c='skyblue', label=label_ann_CtRt)
        # plt.plot(pred_step_ann_CR_pair, c='tomato', label=label_ann_CR_pair)
        plt.plot(x,pred_step_cnn_CtRt, c='dodgerblue', label=label_cnn_CtRt)
        # plt.plot(pred_step_cnn_CR_pair, c='chocolate', label=label_cnn_CR_pair)
        plt.plot(x,pred_step_cnn_lstm_CtRt, c='m', label=label_cnn_lstm_CtRt)
        # plt.plot(pred_step_seq2seq_Ct, c='yellow', label=label_seq2seq_Ct)
        # plt.plot(pred_step_ann_Ct, c='greenyellow', label=label_ann_Ct)
        plt.plot(x,reg,c='purple',label=label_re)
        plt.plot(x,mean_te,c='red',label=label_bay)
        plt.fill_between(x,low_te,upper_te,color='red', alpha=0.2)

        plt.grid(axis='y')
        plt.title("test = " + str(r0))
        plt.legend()
        plt.show()









        







    












