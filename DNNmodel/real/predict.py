from cmath import inf
from args import *
from process import *
from tqdm import tqdm
from itertools import chain
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
sys.path.append("..\\")
from myModel import *
from loss import *
from myModel_patch import *

if __name__ == '__main__':
    smooth = True
    fold_model = "model"
    fold_img = "img"

    def test(args,Dte,model):
        pred = []
        loss_type = args.loss_type
        for(seq,target) in tqdm(Dte):
            
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
        pred = getSmooth(pred,7)
        
        return pred
    
    def predict(info):
        Dte = []
        y_list = []
        pred_list = []
        duration = []
        valid_text = []
        args = CtTransformer_pretrain_args_parser()
        model_path = fold_model + "/myModel_" + info + ".pkl"
        _,ct_min,ct_max = getTrainData(args,50,smooth)
        Dte = getTestData(args,50,smooth,ct_min,ct_max)
        model = CtTransformer(args,head_type="prediction").to(device)

        model.load_state_dict(torch.load(model_path)['models'])
        model.eval()

        pred = test(args,Dte,model)
        pred = np.exp(pred)
        return pred

        
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
    
    info = "ER"
    info = "end-to-end_ER"
    label = "myModel,mse="

    pred = predict(info)
    
    date_list,records_list,Rt_mean_list,Rt_lower_list,Rt_upper_list,ct_mean_list,ct_skew_list = readData("test")

    dates = [datetime.strptime(i, '%Y-%m-%d').date() for i in date_list]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(dates[::7])

    plt.plot(dates,Rt_mean_list,'r')
    plt.fill_between(dates,Rt_lower_list,Rt_upper_list,color='red', alpha=0.2)
    plt.plot(dates,pred)
    plt.axhline(y=1) 

    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(axis='y')
    plt.xlabel("date")
    plt.ylabel("Rt")
    plt.savefig(fold_img + info + ".png")
    plt.close()

    print("mse=" + str(mse(Rt_mean_list,pred)))
    print(' ')
    print("mae=" + str(mae(Rt_mean_list,pred)))




