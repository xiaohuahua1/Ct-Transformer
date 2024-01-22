import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from tqdm import tqdm
from itertools import chain
import torch
from process import *
import sys
sys.path.append("..\\DNNmodel")
# C:\Users\zxy\Desktop\Epidemic\viral loads\code\network\network\DNNmodel
# sys.path.append("C:\\Users\\zxy\\Desktop\\Epidemic\\viral loads\\code\\network\\network\\DNNmodel")
from models import *
from args import *
# df = pd.read_csv('data_daily_all.csv')
# phase1 = df[36:109]
# # print(phase1)

# date = np.array(phase1["date"])
# date_list = date.tolist()

# records = np.array(phase1["records"])
# records_list = records.tolist()

# Rt_mean = np.array(phase1['local.rt.mean'])
# Rt_mean_list = Rt_mean.tolist()

# Rt_lower = np.array(phase1['local.rt.lower'])
# Rt_lower_list = Rt_lower.tolist()

# Rt_upper = np.array(phase1['local.rt.upper'])
# Rt_upper_list = Rt_upper.tolist()

# ct_mean = np.array(phase1["mean"])
# ct_mean_list = ct_mean.tolist()

# ct_skew = np.array(phase1["skewness"])
# ct_skew_list = ct_skew.tolist()

# dates = [datetime.strptime(i, '%Y-%m-%d').date() for i in date_list]


# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.xticks(dates[::7])
# plt.plot(dates,records_list,'r')

# plt.gcf().autofmt_xdate() 
# plt.show()
# # print(date)

# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.xticks(dates[::7])
# plt.plot(dates,Rt_mean_list,'r')
# plt.fill_between(dates,Rt_lower_list,Rt_upper_list,color='red', alpha=0.2)

# plt.gcf().autofmt_xdate() 
# plt.show()

# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.xticks(dates[::7])
# plt.plot(dates,ct_mean_list,'r')

# plt.gcf().autofmt_xdate() 
# plt.show()

# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
# plt.xticks(dates[::7])
# plt.plot(dates,ct_skew_list,'r')

# plt.gcf().autofmt_xdate() 
# plt.show()

path_model = "..\\DNNmodel\\model\\SF\\d=8R=2"

name_model = "transformer_Rt"

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

date_list,records_list,Rt_mean_list,Rt_lower_list,Rt_upper_list,ct_mean_list,ct_skew_list = readData()

dates = [datetime.strptime(i, '%Y-%m-%d').date() for i in date_list]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(dates[::7])

def predict(Type):
   
   if Type == "transformer_Rt":
       model_path = path_model + "\\transformer_Rt.pkl"
       args = transformer_Rt_args_parser()
       seq,start,end,m,n = createDataset(args,Rt_mean_list,ct_mean_list,ct_skew_list)
       model = Transformer(args).to(device)
   model.load_state_dict(torch.load(model_path)['models'])
   model.eval()
   y,pred = test(seq,model)
   return y,pred,start,end,m,n

def mse(y,pred):
        result = 0.0
        num  = len(y)
        if num == 0:
            return result
        for i in range(num):
            result += math.pow((pred[i] - y[i]),2)
        result /= num
        return result

y_transformer_Rt,pred_transformer_Rt,start_transformer_Rt,end_transformer_Rt,m,n = predict("transformer_Rt")

def keshi(y,pred,start,end,label,color):
    pred,y = np.array(pred),np.array(y)
    pred = (m-n)*pred + n
    y = (m-n)*y + n

    mse_value = mse(y,pred)
    label_value = label + "{:.4f}".format(mse_value)

    plt.plot(dates[start:end], pred,c=color, label=label_value)


keshi(y_transformer_Rt,pred_transformer_Rt,start_transformer_Rt,end_transformer_Rt,"transformer_Rt,mse=","m")
plt.plot(dates,Rt_mean_list,'r')
plt.fill_between(dates,Rt_lower_list,Rt_upper_list,color='red', alpha=0.2)
plt.gcf().autofmt_xdate() 
plt.show()



