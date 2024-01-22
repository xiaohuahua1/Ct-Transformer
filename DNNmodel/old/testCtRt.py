from data_process import *
from args import *
from models import *
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math


if __name__ == '__main__':

    def getType(Type):
        path_data = ""
        path_model = ""
        text = []
        r = []
        d = []
        num = 1
        if Type == 'SF':
            path_data = "..\\results\\SF\\DNN"
            path_model = "model\\SF"
            r = [1.3,1.5,2.0,3.0]
            d = [8,10,12]
        elif Type == 'ER':
            path_data = "..\\results\\ER\\DNN\\changeDR"
            path_model = "model\\ER\\changeDR"
            r = [1.3,1.5,2.0,2.5]
            d = [6,8,10,12]
        elif Type == 'd=8R=2':
            path_data = "..\\results\\SF\\d=8R=2"
            path_model = "model\\SF\\d=8R=2"
            r = [2.0]
            d = [8]
            num = 10
        elif Type == 'SF changeD':
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
    
    Type = "d=8R=2"
    path_data,path_model,text = getType(Type)

    Inum = 10
    isB = True
    isR = False


    test_path = path_data + "\\test"
    val_path = path_data + "\\val"
    train_path = path_data + "\\train"

    def test(Dte,model):
        y = []
        pred = []

        for(seq,target) in tqdm(Dte):
            target = list(chain.from_iterable(target.data.tolist()))
            y.extend(target)
            # seq = seq.to(device)
            with torch.no_grad():
               y_pred = model(seq)
               y_pred = list(chain.from_iterable(y_pred.data.tolist()))
               for i in range(len(y_pred)):
                   if y_pred[i] < 0:
                       y_pred[i] = 0
               pred.extend(y_pred)
        return y, pred 

    def predict(model_type):
        Dte = []
        l = []
        d = []
        te = []
        m = 0
        n = 0

        if model_type == "seq2seq_CtRt":
            model_path = path_model + "\\seq2seq_CtRt.pkl"
            args = seq2seq_CtRt_args_parser()
            Dtr,Val,m,n = getTrainData(args,train_path,val_path,[],Inum,False)
            Dte,l,te,d= getTestData_fixed(args,test_path,m,n,text,Inum)
            model = Seq2Seq_Ct(args).to(device)
        elif model_type == "transformer_Rt":
            model_path = path_model + "\\transformer_Rt.pkl"
            args = transformer_Rt_args_parser()
            Dtr,Val,m,n = getTrainData(args,train_path,val_path,[],Inum,False)
            Dte,l,te,d = getTestData_fixed(args,test_path,m,n,text,Inum)
            model = Transformer(args).to(device)
        elif model_type == "transformer_Ct":
            model_path = path_model + "\\transformer_Ct.pkl"
            args = transformer_Ct_args_parser()
            Dtr,Val,m,n = getTrainData(args,train_path,val_path,[],Inum,False)
            Dte,l,te,d = getTestData_fixed(args,test_path,m,n,text,Inum)
            model = Transformer_Ct(args).to(device)

        model.load_state_dict(torch.load(model_path)['models'])
        model.eval()
        y,pred = test(Dte,model)
        return y,pred,l,d,te,m,n


    def mse(y,pred):
        result = 0.0
        num  = len(y)
        if num == 0:
            return result
        for i in range(num):
            result += math.pow((pred[i] - y[i]),2)
        result /= num
        return result


    y_seq2seq_roll,pred_seq2seq_roll,l_seq2seq_roll,d_seq2seq_roll,te,m,n = predict("seq2seq_CtRt")
    # y_transformer_Rt,pred_transformer_Rt,l_transformer_Rt,d_transformer_Rt,te,m,n = predict("transformer_Rt")
    # y_transformer_Ct,pred_transformer_Ct,l_transformer_Ct,d_transformer_Ct,te,m,n = predict("transformer_Ct")

    nums = len(l_seq2seq_roll)

    def keshi(y,pred,l,d,i,start,end,label,color):
        step = l[i]
        end = start + step
        pred_step = pred[start:end]
        y_step = y[start:end]

        pred_step,y_step = np.array(pred_step),np.array(y_step)
        pred_step = (m-n)*pred_step + n
        y_step = (m-n)*y_step + n

        x = [i for i in range(d[i][0],d[i][1])]
        mse_value = mse(y_step,pred_step)
        label_value = label + "{:.4f}".format(mse_value)

        plt.plot(x,y_step, c='black', label='true')
        plt.plot(x,pred_step, c=color, label=label_value)

        start = end
        return start,end


    start_seq2seq_roll = 0
    end_seq2seq_roll = 0

    start_ann_CtRt = 0
    end_ann_CtRt = 0


    start_cnn_CtRt = 0
    end_cnn_CtRt = 0

    start_cnn_lstm_CtRt = 0
    end_cnn_lstm_CtRt = 0

    start_transformer_Rt = 0
    end_transformer_Rt = 0

    start_transformer_Ct = 0
    end_transformer_Ct = 0

    if isB:
        Ts,M,U,L,R = getBayes(test_path)

    for i in range(nums):

        t = te[i]
        start_seq2seq_roll,end_seq2seq_roll = keshi(y_seq2seq_roll,pred_seq2seq_roll,l_seq2seq_roll,d_seq2seq_roll,i,start_seq2seq_roll,end_seq2seq_roll,"seq2seq_roll,mse=","lightseagreen")
        # start_transformer_Rt,end_transformer_Rt = keshi(y_transformer_Rt,pred_transformer_Rt,l_transformer_Rt,d_transformer_Rt,i,start_transformer_Rt,end_transformer_Rt,"transformer_Rt,mse=","m")
        # start_transformer_Rt_ANN_Ct,end_transformer_Rt_ANN_Ct = keshi(y_transformer_Rt_ANN_Ct,pred_transformer_Rt_ANN_Ct,l_transformer_Rt_ANN_Ct,d_transformer_Rt_ANN_Ct,i,start_transformer_Rt_ANN_Ct,end_transformer_Rt_ANN_Ct,"transformer_Rt_ANN_Ct,mse=","skyblue")
        # start_transformer_Ct,end_transformer_Ct = keshi(y_transformer_Ct,pred_transformer_Ct,l_transformer_Ct,d_transformer_Ct,i,start_transformer_Ct,end_transformer_Ct,"transformer_Ct,mse=","yellow")
        if isB:
            x = [j for j in range(Ts[i][0],Ts[i][1])]
            mse_base = mse(R[i],M[i])
            # plt.plot(x,R[i], c='black', label='true')
            plt.plot(x,R[i], c='black')
            plt.plot(x,M[i],c='r',label="bayesian,mse="+ "{:.4f}".format(mse_base))
            plt.fill_between(x,L[i],U[i],color='red', alpha=0.2)

        plt.grid(axis='y')
        plt.axhline(y=1)
        plt.title("d = " + str(t[0]) + ",R0 = " + str(t[1]) + ",num = " + str(t[2]))
        plt.legend()
        plt.show()















    # for i in range(nums):

    #     label = te[i]


    #     step_seq2seq_roll = l_seq2seq_roll[i]
    #     end_seq2seq_roll = start_seq2seq_roll + step_seq2seq_roll
    #     pred_step_seq2seq_roll = pred_seq2seq_roll[start_seq2seq_roll:end_seq2seq_roll]
    #     y_step_seq2seq_roll = y_seq2seq_roll[start_seq2seq_roll:end_seq2seq_roll]

    #     # step_ann_CtRt = l_ann_CtRt[i]
    #     # end_ann_CtRt = start_ann_CtRt + step_ann_CtRt
    #     # pred_step_ann_CtRt = pred_ann_CtRt[start_ann_CtRt:end_ann_CtRt]
    #     # y_step_ann_CtRt = y_ann_CtRt[start_ann_CtRt:end_ann_CtRt]


    #     # step_cnn_CtRt = l_cnn_CtRt[i]
    #     # end_cnn_CtRt = start_cnn_CtRt + step_cnn_CtRt
    #     # pred_step_cnn_CtRt = pred_cnn_CtRt[start_cnn_CtRt:end_cnn_CtRt]
    #     # y_step_cnn_CtRt = y_cnn_CtRt[start_cnn_CtRt:end_cnn_CtRt]


    #     # step_cnn_lstm_CtRt = l_cnn_lstm_CtRt[i]
    #     # end_cnn_lstm_CtRt = start_cnn_lstm_CtRt + step_cnn_lstm_CtRt
    #     # pred_step_cnn_lstm_CtRt = pred_cnn_lstm_CtRt[start_cnn_lstm_CtRt:end_cnn_lstm_CtRt]
    #     # y_step_cnn_lstm_CtRt = y_cnn_lstm_CtRt[start_cnn_lstm_CtRt:end_cnn_lstm_CtRt]

    

    #     pred_step_seq2seq_roll,y_step_seq2seq_roll = np.array(pred_step_seq2seq_roll),np.array(y_step_seq2seq_roll)
    #     pred_step_seq2seq_roll = (m-n)*pred_step_seq2seq_roll + n
    #     y_step_seq2seq_roll = (m-n)*y_step_seq2seq_roll + n

    #     # pred_step_ann_CtRt,y_step_ann_CtRt = np.array(pred_step_ann_CtRt),np.array(y_step_ann_CtRt)
    #     # pred_step_ann_CtRt = (m-n)*pred_step_ann_CtRt + n
    #     # y_step_ann_CtRt = (m-n)*y_step_ann_CtRt + n

    #     # pred_step_cnn_CtRt,y_step_cnn_CtRt = np.array(pred_step_cnn_CtRt),np.array(y_step_cnn_CtRt)
    #     # pred_step_cnn_CtRt = (m-n)*pred_step_cnn_CtRt + n
    #     # y_step_cnn_CtRt = (m-n)*y_step_cnn_CtRt + n

    #     # pred_step_cnn_lstm_CtRt,y_step_cnn_lstm_CtRt = np.array(pred_step_cnn_lstm_CtRt),np.array(y_step_cnn_lstm_CtRt)
    #     # pred_step_cnn_lstm_CtRt = (m-n)*pred_step_cnn_lstm_CtRt + n
    #     # y_step_cnn_lstm_CtRt = (m-n)*y_step_cnn_lstm_CtRt + n


    #     x_seq2seq_roll = [i for i in range(d_seq2seq_roll[i][0],d_seq2seq_roll[i][1])]
    #     # x_ann_CtRt = [i for i in range(d_ann_CtRt[i][0],d_ann_CtRt[i][1])]
    #     # x_cnn_CtRt = [i for i in range(d_cnn_CtRt[i][0],d_cnn_CtRt[i][1])]
    #     # x_cnn_lstm_CtRt = [i for i in range(d_cnn_lstm_CtRt[i][0],d_cnn_lstm_CtRt[i][1])]

    #     mse_seq2seq_roll = mse(y_step_seq2seq_roll,pred_step_seq2seq_roll)
    #     # mse_ann_CtRt = mse(y_step_ann_CtRt,pred_step_ann_CtRt)
    #     # mse_cnn_CtRt = mse(y_step_cnn_CtRt,pred_step_cnn_CtRt)
    #     # mse_cnn_lstm_CtRt = mse(y_step_cnn_lstm_CtRt,pred_step_cnn_lstm_CtRt)

    #     label_seq2seq_roll = 'seq2seq_roll,mse=' + "{:.4f}".format(mse_seq2seq_roll)
    #     # label_ann_CtRt = 'ann_seq,mse=' + "{:.4f}".format(mse_ann_CtRt)
    #     # label_cnn_CtRt = 'cnn_seq,mse=' + "{:.4f}".format(mse_cnn_CtRt)
    #     # label_cnn_lstm_CtRt = 'cnn_lstm_seq,mse=' + "{:.4f}".format(mse_cnn_lstm_CtRt)

    #     plt.plot(x_seq2seq_roll,y_step_seq2seq_roll, c='black', label='true')
    #     plt.plot(x_seq2seq_roll,pred_step_seq2seq_roll, c='lightseagreen', label=label_seq2seq_roll)

    #     # plt.plot(x_ann_CtRt,y_step_ann_CtRt, c='black')
    #     # plt.plot(x_ann_CtRt,pred_step_ann_CtRt, c='skyblue', label=label_ann_CtRt)

    #     # plt.plot(x_cnn_CtRt,y_step_cnn_CtRt, c='black')
    #     # plt.plot(x_cnn_CtRt,pred_step_cnn_CtRt, c='dodgerblue', label=label_cnn_CtRt)

    #     # plt.plot(x_cnn_lstm_CtRt,y_step_cnn_lstm_CtRt, c='black')
    #     # plt.plot(x_cnn_lstm_CtRt,pred_step_cnn_lstm_CtRt, c='m', label=label_cnn_lstm_CtRt)

    #     plt.grid(axis='y')
    #     plt.axhline(y=1)
    #     plt.title("d = " + str(te[0]) + ",R0 = " + str(te[1]) + ",num = " + str(te[2]))
    #     plt.legend()
    #     plt.show()

    #     start_seq2seq_roll = end_seq2seq_roll
    #     # start_ann_CtRt = end_ann_CtRt
    #     # start_cnn_CtRt = end_cnn_CtRt
    #     # start_cnn_lstm_CtRt = end_cnn_lstm_CtRt




# path_seq2seq_roll = path_model + "\\seq2seq_CtRt1.pkl"
    # path_ann_CtRt = path_model + "\\ann_CtRt.pkl"
    # path_cnn_CtRt = path_model + "\\cnn_CtRt.pkl"
    # path_cnn_lstm_CtRt = path_model + "\\cnn_lstm_CtRt.pkl"


    # args_seq2seq_roll = seq2seq_CtRt_args_parser()
    # args_ann_CtRt = ann_CtRt_args_parser()
    # args_cnn_CtRt = cnn_CtRt_args_parser()
    # args_cnn_lstm_CtRt =  cnn_lstm_CtRt_args_parser()


    # Dtr_seq2seq_roll, Val_seq2seq_roll, m_seq2seq_roll, n_seq2seq_roll = getTrainData(args_seq2seq_roll,train_path,val_path,Inum)
    # Dtr_ann_CtRt, Val_ann_CtRt, m_ann_CtRt, n_ann_CtRt = getTrainData(args_ann_CtRt,train_path,val_path,[])
    # Dtr_cnn_CtRt, Val_cnn_CtRt, m_cnn_CtRt, n_cnn_CtRt, = getTrainData(args_cnn_CtRt,train_path,val_path,[])
    # Dtr_cnn_lstm_CtRt, Val_cnn_lstm_CtRt, m_cnn_lstm_CtRt, n_cnn_lstm_CtRt = getTrainData(args_cnn_lstm_CtRt,train_path,val_path,[])
    
    
    # Dte_seq2seq_roll, l_seq2seq_roll,R0_seq2seq_roll,d_seq2seq_roll = getTestData(args_seq2seq_roll,test_path,m_seq2seq_roll,n_seq2seq_roll,Inum,text,isB,isR)
    # Dte_ann_CtRt,l_ann_CtRt,R0_ann_CtRt,d_ann_CtRt = getTestData_fixed(args_ann_CtRt,test_path,m_ann_CtRt,n_ann_CtRt,text)
    # Dte_cnn_CtRt,l_cnn_CtRt,R0_cnn_CtRt,d_cnn_CtRt = getTestData_fixed(args_cnn_CtRt,test_path,m_cnn_CtRt,n_cnn_CtRt,text)
    # Dte_cnn_lstm_CtRt,l_cnn_lstm_CtRt,R0_cnn_lstm_CtRt,d_cnn_lstm_CtRt = getTestData_fixed(args_cnn_lstm_CtRt,test_path,m_cnn_lstm_CtRt,n_cnn_lstm_CtRt,text)
    

    # m = 0.0
    # n = 0.0
    # R0 = []
    

    # # print(m)
    # # print(n)
    # m = m_seq2seq_roll
    # n = n_seq2seq_roll
    # R0 = R0_seq2seq_roll



    # model_seq2seq_roll = Seq2Seq_Ct(args_seq2seq_roll).to(device)
    # # model_ann_CtRt = ANN(args_ann_CtRt).to(device)
    # # model_cnn_CtRt = CNN(args_cnn_CtRt).to(device)
    # # model_cnn_lstm_CtRt = CNN_LSTM(args_cnn_lstm_CtRt).to(device)



    # model_seq2seq_roll.load_state_dict(torch.load(path_seq2seq_roll)['models'])
    # model_seq2seq_roll.eval()

    # # model_ann_CtRt.load_state_dict(torch.load(path_ann_CtRt)['models'])
    # # model_ann_CtRt.eval()


    # # model_cnn_CtRt.load_state_dict(torch.load(path_cnn_CtRt)['models'])
    # # model_cnn_CtRt.eval()


    # # model_cnn_lstm_CtRt.load_state_dict(torch.load(path_cnn_lstm_CtRt)['models'])
    # # model_cnn_lstm_CtRt.eval()



    # y_seq2seq_roll,pred_seq2seq_roll = test(Dte_seq2seq_roll,model_seq2seq_roll)
    # # y_ann_CtRt,pred_ann_CtRt = test(Dte_ann_CtRt,model_ann_CtRt)
    # # y_cnn_CtRt,pred_cnn_CtRt = test(Dte_cnn_CtRt,model_cnn_CtRt)
    # # y_cnn_lstm_CtRt,pred_cnn_lstm_CtRt = test(Dte_cnn_lstm_CtRt,model_cnn_lstm_CtRt)

    # # print(pred_cnn_lstm_CtRt)
    

    # nums = len(l_seq2seq_roll)

    # def search(pat, txt):#找到pat在txt子串的第一次出现位置
    #     i, N = 0, len(txt)
    #     j, M = 0, len(pat)
    #     while i < N and j < M:
    #         if txt[i] == pat[j]:
    #             j = j + 1
    #         else:
    #             i -= j
    #             j = 0
    #         i = i + 1
    #     if j == M:
    #         return i - M
    #     else:
    #         return -1
