from cmath import inf
from data_process import *
from args import *
from model import *
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math
# import statsmodels.api as sm
from TFT import *
from patchTST import *
from myModels import *


if __name__ == '__main__':


    
    Inum = 100
    smooth = True
    isR = False
    isB = False


    fold_test_data = "../data/SF/changeR"
    fold_model = "model/SF"
    fold_img = "img/SF/changeR"

    text = []
    net = ["SF"]
    d = [10]
    r = [1.2,1.5,1.8,2.0,2.2,2.5,3.0,3.3,4.0]
    num = 5

    for i in range(len(net)):
        for j in range(len(d)):
            for z in range(len(r)):
                for x in range(num):
                    text.append([net[i],d[j],r[z],x])


    test_path = fold_test_data + "/test"

    fold_train_data = "../data"
    net_train = ["SF"]
    d_train = [10]
    R_train = [1.5,2,2.5,3]

    # net_train = ["SF"]
    # d_train = [10]
    # R_train = [2]

    def test(args,Dte,model):
        y = []
        pred = []
        loss_type = args.loss_type
        for(seq,target) in tqdm(Dte):
            # target = target[:,:,0]
            target = list(chain.from_iterable(target.data.tolist()))
            y.extend(target)
            # seq = seq.to(device)
            with torch.no_grad():
                if args.id == 3:
                    patch_len = args.patchLen
                    patch_input,length = create_patch(seq,patch_len)
                    y_pred = model(patch_input,length)
                else:          
                    y_pred = model(seq)

                if loss_type == 2:
                    y_pred = y_pred[:,:,1]
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # for i in range(len(y_pred)):
                #     if y_pred[i] < 0:
                #        y_pred[i] = 0
                pred.extend(y_pred)
        y = getSmooth(y,7)
        pred = getSmooth(pred,7)
        y = y[7:]
        pred = pred[7:]
        return y, pred 


    def predict(model_type,info,id):
        Dte = []
        y_list = []
        pred_list = []
        duration = []
        valid_text = []
        args = CtTransformer_args_parser()
        max_list,min_list,max_rt,min_rt = getBound(args,fold_train_data,net_train,d_train,R_train,Inum,1)

        if model_type == "transformerEncoder":
            model_path = fold_model + "/transformerEncoder_" + info +  ".pkl"
            args = transformer_encoder_args_parser()
            Dte,valid_text,duration = getTestRateData(args,test_path,Inum,smooth,text,max_list,min_list,max_rt,min_rt,id)
            model = transformer_encoder(args).to(device)
        elif model_type == "TFT":
            model_path = fold_model + "/TFT_" + info + ".pkl"
            args = TFT_args_parser()
            Dte,valid_text,duration = getTestRateData(args,test_path,Inum,smooth,text,max_list,min_list,max_rt,min_rt,id)
            model = TemporalFusionTransformer(args).to(device)
        elif model_type == "myModel":
            model_path = fold_model + "/myModel_" + info + ".pkl"
            args = CtTransformer_args_parser()
            Dte,valid_text,duration = getTestRateData(args,test_path,Inum,smooth,text,max_list,min_list,max_rt,min_rt,id)
            model = CtTransformer(args,head_type="prediction").to(device)
            
        model.load_state_dict(torch.load(model_path)['models'])
        model.eval()

        for i in range(len(Dte)):
            Dte_s = Dte[i]
            y,pred = test(args,Dte_s,model)
            y = np.exp(y)
            pred = np.exp(pred)
            y_list.append(y)
            pred_list.append(pred)
            # duration[i][0] = duration[i][0] + 6
        return y_list,pred_list,valid_text,duration

    def mse(y,pred):
        result = 0.0
        num  = len(y)
        if num == 0:
            return result
        for i in range(num):
            result += math.pow((pred[i] - y[i]),2)
        result /= num
        return result
    
    def mae(y,pred):
        result = 0.0
        num  = len(y)
        if num == 0:
            return result
        for i in range(num):
            result += abs(pred[i] - y[i])
        result /= num
        return result


    y_list_all = []
    pred_list_all = []
    label_all = []
    valid_text = []
    duration = []

    info = "ER"
    info = "end-to-end_ER"
    label = "myModel,mse="
    y_list,pred_list,valid_text,duration_list = predict("myModel",info)
    y_list_all.append(y_list)
    pred_list_all.append(pred_list)
    label_all.append(label)
    duration.append(duration_list)
    
    # label = "TFT,mse="
    # y_list,pred_list,valid_text,duration_list = predict("TFT",info)
    # y_list_all.append(y_list)
    # pred_list_all.append(pred_list)
    # label_all.append(label)
    # duration.append(duration_list)

    # label = "transformerEncoder,mse="
    # y_list,pred_list,valid_text,duration_list = predict("transformerEncoder",info)
    # y_list_all.append(y_list)
    # pred_list_all.append(pred_list)
    # label_all.append(label)
    # duration.append(duration_list)

    if isR:
        label = "Regression,mse="
        Rpre,_ = getRegression(test_path ,Inum,pre_step=8)
        y_list_all.append(y_list)
        pred_list_all.append(Rpre)
        label_all.append(label)
        duration.append(duration_list)

    if isB:
        label = "Bayesian,mse="
        Ts,M,U,L,R = getBayes(test_path,Inum,pre_step=8)
        y_list_all.append(y_list)
        pred_list_all.append(M)
        label_all.append(label)
        duration.append(duration_list)
    


    def keshi(y,pred,start,end,label,color):
        
        x = [i for i in range(start,start+len(y))]
        mse_value = mse(y,pred)
        mae_value = mae(y,pred)
        label_value = label + "{:.4f}".format(mse_value)

        plt.plot(x,y, c='black')
        plt.plot(x,pred, c=color, label=label_value)
        return mse_value,mae_value

    mse_list = []
    mae_list = []

    for i in range(len(y_list_all)):
        mse_list.append([])
        mae_list.append([])
    

    for x in range(len(y_list_all[0])):

        color = ["rosybrown","tomato","chocolate","goldenrod","y","darksage","green","lightseagreen","teal","deepskyblue","blue","blueviolet","m","deeppink"]
        

        for y in range(len(y_list_all)):
            mse_value,mae_value = keshi(y_list_all[y][x],pred_list_all[y][x],duration[y][x][0],duration[y][x][1],label_all[y],color[y])
            mse_list[y].append(mse_value)
            mae_list[y].append(mae_value)
            # pred_list_merge.append(pred_list_all[y][x])
        # merge_result = build_merge_regression(pred_list_merge,y_list_all[0][x])
        # keshi(y_list_all[0][x],merge_result,duration_lstm[x][0],duration_lstm[x][1],"merge,mse=",color[len(y_list_all)])

        plt.grid(axis='y')
        plt.axhline(y=1)
        plt.title("net = " + str(valid_text[x][0]) + ",d = " + str(valid_text[x][1]) + ",R0 = " + str(valid_text[x][2]) + ",num = " + str(valid_text[x][3]))
        plt.legend()
        # plt.show()
        plt.savefig(fold_img + "/ERModel_sf_e" + str(x) + ".png")
        plt.close()

    for i in range(len(mse_list)):
        model_list = mse_list[i]
        total = 0.0
        n = len(model_list)
        for j in range(n):
            total += model_list[j]
        total /= n
        print(label_all[i] + ",mse=" + str(total))

        model_list = mae_list[i]
        total = 0.0
        n = len(model_list)
        for j in range(n):
            total += model_list[j]
        total /= n
        print(label_all[i] + ",mae=" + str(total))

        print(' ')

