from cmath import inf
from data_process import *
from args import *

from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
# import statsmodels.api as sm
from modelAbation import *

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
    y = getSmooth(y,4)
    pred = getSmooth(pred,4)
    # y = y[3:]
    # pred = pred[3:]
    return y, pred 


def predict(model_type,fold_model,fold_train_data,net_train,d_train,R_train,Inum,test_path,smooth,text):
    Dte = []
    y_list = []
    pred_list = []
    duration = []
    valid_text = []
    args = Ablation_args_parser()
    max_list,min_list,max_rt,min_rt = getBound(args,fold_train_data,net_train,d_train,R_train,Inum,1)
    Dte,valid_text,duration = getTestData(args,test_path,Inum,smooth,text,max_list,min_list,max_rt,min_rt)

    if model_type == "NoPatch":
        args.id = 2
        model_path = fold_model + "/NoPatch.pkl"
        model = NoPatch(args).to(device)
    elif model_type == "NoGRU":
        model_path = fold_model + "/NoGRU.pkl"
        model = NoGRU(args).to(device)
    elif model_type == "NoattenForward":
        model_path = fold_model + "/NoattenForward.pkl"
        model = NoattenForward(args).to(device)
            
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

def rmse(mse):
    return math.sqrt(mse)
    
def mape(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    for i in range(num):
        result += abs((y[i] - pred[i]) / y[i])
    result /= num
    result *= 100
    return result

def r2(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    mean_actual = sum(y) / len(y)
    ss_total = sum((actual_val - mean_actual) ** 2 for actual_val in y)
    ss_residual = sum((actual_val - predicted_val) ** 2 for actual_val, predicted_val in zip(y, pred))
    
    result = 1 - (ss_residual / ss_total)
    return result

if __name__ == '__main__':


    
    # Inum = 15
    Inum = 15
    smooth = True

    fold_train_data = "data"
    net_train = ["ER"]
    d_train = [10]
    # R_train = [1.5,2,2.5,3]
    R_train = [1.5,2,2.5,3]

    text_list = ["net = ER , d = 10 , R0 = 1.2","net = ER , d = 10 , R0 = 1.4","net = ER , d = 10 , R0 = 1.6",
                 "net = ER , d = 10 , R0 = 1.8","net = ER , d = 10 , R0 = 2.2","net = ER , d = 10 , R0 = 2.4",
                 "net = ER , d = 10 , R0 = 2.6","net = ER , d = 10 , R0 = 2.8","net = ER , d = 10 , R0 = 3.2",
                 "net = ER , d = 10 , R0 = 3.4"]
    
    test_fold_list = ["data/ER/d=10R=1.2/test","data/ER/d=10R=1.4/test","data/ER/d=10R=1.6/test",
                      "data/ER/d=10R=1.8/test","data/ER/d=10R=2.2/test","data/ER/d=10R=2.4/test",
                      "data/ER/d=10R=2.6/test","data/ER/d=10R=2.8/test","data/ER/d=10R=3.2/test",
                      "data/ER/d=10R=3.4/test"]
    
    total_num = 30
    text_num = [i for i in range(total_num)]
    
    model_fold = "model/ER/ablation"
    result_fold = "/result/ER/ablation"

    for i in range(len(test_fold_list)):
        test_fold = test_fold_list[i]
        text = text_list[i]
        result_fold_combine = test_fold + result_fold

        y_list,pred_list,valid_text,duration_list = predict("NoPatch",model_fold,fold_train_data,net_train,
                                                            d_train,R_train,Inum,test_fold,smooth,text_num)
        
        Cf = open(result_fold_combine + "/NoPatch.txt","w")
        id = 0
        for j in range(total_num):
            if j not in valid_text:
                Cf.write('0 \n')
            else:
                pred = pred_list[id]
                y = y_list[id]

                for z in range(len(pred)):
                    Cf.write("{:.6f}".format(pred[z]) + " ")
                Cf.write("\n")
                
                duration = duration_list[id]

                id += 1
        Cf.close()

        MSE_list = []
        MAE_list = []
        RMSE_list = []
        MAPE_list = []
        R2_list = []

        csvCf = open(result_fold_combine + "/NoPatch.csv","w",newline='')
        csvC_result = []
        csvC_head = ["id","MSE","MAE","RMSE","MAPE","R2"]
        csvC_result.append(csvC_head)

        id = 0
        for j in range(total_num):
            if j not in valid_text:
                csvC_row = [str(j),"0","0","0","0","0","0"]
            else:
                pred = pred_list[id]
                y = y_list[id]

                mse_value = mse(y,pred)
                mae_value = mae(y,pred)
                rmse_value = rmse(mse_value)
                mape_value = mape(y,pred)
                r2_value = r2(y,pred)

                MSE_list.append(mse_value)
                MAE_list.append(mae_value)
                RMSE_list.append(rmse_value)
                MAPE_list.append(mape_value)
                R2_list.append(r2_value)

                csvC_row = [str(j),"{:.4f}".format(mse_value),"{:.4f}".format(mae_value),
                            "{:.4f}".format(rmse_value),"{:.4f}".format(mape_value),"{:.4f}".format(r2_value)]
                
                id += 1
            csvC_result.append(csvC_row)
        
        MSE_result = 0.0
        MAE_result = 0.0
        RMSE_result = 0.0
        MAPE_result = 0.0
        R2_result = 0.0
        n = len(MSE_list)
        for j in range(n):
            MSE_result += MSE_list[j]
            MAE_result += MAE_list[j]
            RMSE_result += RMSE_list[j]
            MAPE_result += MAPE_list[j]
            R2_result += R2_list[j]
        
        MSE_result /= n
        MAE_result /= n
        RMSE_result /= n
        MAPE_result /= n
        R2_result /= n

        csvC_row = ["ave","{:.4f}".format(MSE_result),"{:.4f}".format(MAE_result),
                            "{:.4f}".format(RMSE_result),"{:.4f}".format(MAPE_result),"{:.4f}".format(R2_result)]
        
        csvC_result.append(csvC_row)

        writer = csv.writer(csvCf)
        writer.writerows(csvC_result)
        csvCf.close()


        # info = "best"
        y_list,pred_list,valid_text,duration_list = predict("NoGRU",model_fold,fold_train_data,net_train,
                                                            d_train,R_train,Inum,test_fold,smooth,text_num)
        
        Tf = open(result_fold_combine + "/NoGRU.txt","w")
        id = 0
        for j in range(total_num):
            if j not in valid_text:
                Tf.write('0 \n')
            else:
                pred = pred_list[id]
                for z in range(len(pred)):
                    Tf.write("{:.6f}".format(pred[z]) + " ")
                Tf.write("\n")

                id += 1
        Tf.close()

        MSE_list = []
        MAE_list = []
        RMSE_list = []
        MAPE_list = []
        R2_list = []

        csvTf = open(result_fold_combine + "/NoGRU.csv","w",newline='')
        csvT_result = []
        csvT_head = ["id","MSE","MAE","RMSE","MAPE","R2"]
        csvT_result.append(csvT_head)

        id = 0
        for j in range(total_num):
            if j not in valid_text:
                csvT_row = [str(j),"0","0","0","0","0","0"]
            else:
                pred = pred_list[id]
                y = y_list[id]

                mse_value = mse(y,pred)
                mae_value = mae(y,pred)
                rmse_value = rmse(mse_value)
                mape_value = mape(y,pred)
                r2_value = r2(y,pred)

                MSE_list.append(mse_value)
                MAE_list.append(mae_value)
                RMSE_list.append(rmse_value)
                MAPE_list.append(mape_value)
                R2_list.append(r2_value)

                csvT_row = [str(j),"{:.4f}".format(mse_value),"{:.4f}".format(mae_value),
                            "{:.4f}".format(rmse_value),"{:.4f}".format(mape_value),"{:.4f}".format(r2_value)]
                
                id += 1
            csvT_result.append(csvT_row)
        
        MSE_result = 0.0
        MAE_result = 0.0
        RMSE_result = 0.0
        MAPE_result = 0.0
        R2_result = 0.0
        n = len(MSE_list)
        for j in range(n):
            MSE_result += MSE_list[j]
            MAE_result += MAE_list[j]
            RMSE_result += RMSE_list[j]
            MAPE_result += MAPE_list[j]
            R2_result += R2_list[j]
        
        MSE_result /= n
        MAE_result /= n
        RMSE_result /= n
        MAPE_result /= n
        R2_result /= n

        csvT_row = ["ave","{:.4f}".format(MSE_result),"{:.4f}".format(MAE_result),
                            "{:.4f}".format(RMSE_result),"{:.4f}".format(MAPE_result),"{:.4f}".format(R2_result)]
        
        csvT_result.append(csvT_row)

        writer = csv.writer(csvTf)
        writer.writerows(csvT_result)
        csvTf.close()


        y_list,pred_list,valid_text,duration_list = predict("NoattenForward",model_fold,fold_train_data,net_train,
                                                            d_train,R_train,Inum,test_fold,smooth,text_num)
        
        Ef = open(result_fold_combine + "/NoattenForward.txt","w")
        id = 0
        for j in range(total_num):
            if j not in valid_text:
                Ef.write('0 \n')
            else:
                pred = pred_list[id]
                for z in range(len(pred)):
                    Ef.write("{:.6f}".format(pred[z]) + " ")
                Ef.write("\n")

                id += 1
        Ef.close()

        MSE_list = []
        MAE_list = []
        RMSE_list = []
        MAPE_list = []
        R2_list = []

        csvEf = open(result_fold_combine + "/NoattenForward.csv","w",newline='')
        csvE_result = []
        csvE_head = ["id","MSE","MAE","RMSE","MAPE","R2"]
        csvE_result.append(csvE_head)

        id = 0
        for j in range(total_num):
            if j not in valid_text:
                csvE_row = [str(j),"0","0","0","0","0","0"]
            else:
                pred = pred_list[id]
                y = y_list[id]

                mse_value = mse(y,pred)
                mae_value = mae(y,pred)
                rmse_value = rmse(mse_value)
                mape_value = mape(y,pred)
                r2_value = r2(y,pred)

                MSE_list.append(mse_value)
                MAE_list.append(mae_value)
                RMSE_list.append(rmse_value)
                MAPE_list.append(mape_value)
                R2_list.append(r2_value)

                csvE_row = [str(j),"{:.4f}".format(mse_value),"{:.4f}".format(mae_value),
                            "{:.4f}".format(rmse_value),"{:.4f}".format(mape_value),"{:.4f}".format(r2_value)]
                
                id += 1
            csvE_result.append(csvE_row)
        
        MSE_result = 0.0
        MAE_result = 0.0
        RMSE_result = 0.0
        MAPE_result = 0.0
        R2_result = 0.0
        n = len(MSE_list)
        for j in range(n):
            MSE_result += MSE_list[j]
            MAE_result += MAE_list[j]
            RMSE_result += RMSE_list[j]
            MAPE_result += MAPE_list[j]
            R2_result += R2_list[j]
        
        MSE_result /= n
        MAE_result /= n
        RMSE_result /= n
        MAPE_result /= n
        R2_result /= n

        csvE_row = ["ave","{:.4f}".format(MSE_result),"{:.4f}".format(MAE_result),
                            "{:.4f}".format(RMSE_result),"{:.4f}".format(MAPE_result),"{:.4f}".format(R2_result)]
        
        csvE_result.append(csvE_row)

        writer = csv.writer(csvEf)
        writer.writerows(csvE_result)
        csvEf.close()

