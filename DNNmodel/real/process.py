import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import random
from scipy.stats import skewnorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def getSmooth(dataList,tau):
    result = []
    ave = 0.0
    n = len(dataList)
    size = int(tau/2)
    for i in range(n):
        ave = 0.0
        if i - size < 0:
            for j in range(tau):
                ave += dataList[j]
            ave /= tau
            result.append(ave)
        elif i - size + tau > n:
            for j in range(n-tau,n):
                ave += dataList[j]
            ave /= tau
            result.append(ave)
        else:
            for j in range(i - size, i - size + tau):
                ave += dataList[j]
            ave /= tau
            result.append(ave)
    return result

def calculate(dataList):
    mean = 0.0
    upper = 0.0
    lower = 0.0
    data = []
    for i in range(len(dataList)):
        data.append(dataList[i])

    n = len(data)
    if n > 0:
        size = int(n*0.025)
        for i in range(n):
            mean += dataList[i]
        mean /= n
        dataList.sort()
        lower = dataList[size]
        upper = dataList[n - size - 1]

    return mean,lower,upper
    
def readData(data_type):
    df = pd.read_csv('data_daily_all.csv')
    if data_type == "train":
        phase1 = df[36:109]
    elif data_type == "test":
        phase1 = df[170:296]

    date = np.array(phase1["date"])
    date_list = date.tolist()

    records = np.array(phase1["records"])
    records_list = records.tolist()

    Rt_mean = np.array(phase1['local.rt.mean'])
    Rt_mean_list = Rt_mean.tolist()

    Rt_lower = np.array(phase1['local.rt.lower'])
    Rt_lower_list = Rt_lower.tolist()

    Rt_upper = np.array(phase1['local.rt.upper'])
    Rt_upper_list = Rt_upper.tolist()

    ct_mean = np.array(phase1["mean"])
    ct_mean_list = ct_mean.tolist()

    ct_skew = np.array(phase1["skewness"])
    ct_skew_list = ct_skew.tolist()

    return date_list,records_list,Rt_mean_list,Rt_lower_list,Rt_upper_list,ct_mean_list,ct_skew_list

def generate_distrb(mean,skew,group,i):
    if group == 3:
        bound = [[16,23],[24,31],[32,40]]
    elif group == 4:
        bound = [[16,22],[23,28],[29,34],[35,40]]
    elif group == 5:
        bound = [[16,20],[21,25],[26,30],[31,35],[36,40]]
    elif group == 6:
        bound = [[0,21],[22,23],[23,24],[25,26],[27,28],[29,40]]
    elif group == 8:
        bound = [[0,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,40]]
    elif group == 12:
        bound = [[16,18],[19,20],[21,22],[23,24],[25,26],[27,28],[29,30],[31,32],[33,34],[35,36],[37,38],[39,40]]

    distrb = []
    if i > 20:
        scale = 1 + 0.1*i
    else:
        scale = 0.5 + 0.2*i
    rv = skewnorm(skew,loc=mean,scale=scale)
    for i in range(group):
        d = rv.cdf(bound[i][1]) - rv.cdf(bound[i][0])
        distrb.append(d)
    
    return distrb

def generate_train_data(args,Rt_mean_list,ct_mean_list,ct_skew_list,num,smooth):
    data = []
    label = []
    length = len(Rt_mean_list)

    group = args.group
    input_size = args.input_size

    ct_all = []
    ct_cal = []
    ct_min,ct_max = [],[]

    for i in range(group):
        ct_all.append([])
        ct_cal.append([])
        for j in range(length):
            ct_all[i].append([])

    if smooth:
        ct_mean_list = getSmooth(ct_mean_list,7)
        ct_skew_list = getSmooth(ct_skew_list,7)

    for i in range(num):
        for j in range(length):
            distrb = generate_distrb(ct_mean_list[j],ct_skew_list[j],group,i)
            for z in range(group):
                ct_all[z][j].append(distrb[z])
                # ct_cal[z].append(distrb[z])

    for i in range(group):
        for j in range(length):
            median,lower,upper = calculate(ct_all[i][j])
            ct_cal[i].append(median)
    
    for i in range(group):
        value_min,value_max = np.min(ct_cal[i]),np.max(ct_cal[i])
        ct_min.append(value_min)
        ct_max.append(value_max)
    value_min,value_max = np.min(ct_mean_list),np.max(ct_mean_list)
    ct_min.append(value_min)
    ct_max.append(value_max)
    value_min,value_max = np.min(ct_skew_list),np.max(ct_skew_list)
    ct_min.append(value_min)
    ct_max.append(value_max)

    for i in range(num):
        print(i)
        data_t = []
        label_t = []
        for j in range(length):
            label_t.append(np.log(Rt_mean_list[j]))
            if input_size >= group:
                distrb = generate_distrb(ct_mean_list[j],ct_skew_list[j],group,i)
                if input_size == group + 1:
                    distrb.append(ct_mean_list[j])
                elif input_size == group + 2:
                    distrb.append(ct_mean_list[j])
                    distrb.append(ct_skew_list[j])
                data_t.append(distrb)
            else:
                elem = []
                if input_size == 1:
                    elem.append(ct_mean_list[j])
                elif input_size == 2:
                    elem.append(ct_mean_list[j])
                    elem.append(ct_skew_list[j])
                data_t.append(elem)

        smooth = []
        for j in range(input_size):
            smooth.append([])
            for z in range(length):
                smooth[j].append(data_t[z][j])
            if input_size >= group:
                smooth[j] = (smooth[j] - ct_min[j]) / (ct_max[j] - ct_min[j])
            else:
                smooth[j] = (smooth[j] - ct_min[j + group]) / (ct_max[j + group] - ct_min[j + group])
            smooth[j] = getSmooth(smooth[j],7)
        for j in range(length):
            for z in range(input_size):
                data_t[j][z] = smooth[z][j]

        data.append(data_t)
        label.append(label_t)

    
    return data,label,ct_all,ct_min,ct_max

def create_train_Dataset(args,data,label):
    seq = []
    n = len(data)
    batch_size = args.batch_size
    num_workers=args.workers
    for i in range(n):
        train_data = data[i]
        train_label = label[i]

        train_data = torch.FloatTensor(train_data)
        train_label = torch.FloatTensor(train_label)

        seq.append((train_data,train_label))

    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    return seq

def getTrainData(args,num,smooth):
    date_list,records_list,Rt_mean_list,Rt_lower_list,Rt_upper_list,ct_mean_list,ct_skew_list = readData("train")
    data,label,ct_all,ct_min,ct_max = generate_train_data(args,Rt_mean_list,ct_mean_list,ct_skew_list,num,smooth)
    seq = create_train_Dataset(args,data,label)
    return seq,ct_min,ct_max

def generate_test_data(args,Rt_mean_list,ct_mean_list,ct_skew_list,num,smooth,ct_min,ct_max):
    data = []
    label = []
    length = len(Rt_mean_list)

    group = args.group
    input_size = args.input_size

    ct_all = []
    for i in range(group):
        ct_all.append([])
        for j in range(length):
            ct_all[i].append([])
    
    mid_all = []
    
    if smooth:
        ct_mean_list = getSmooth(ct_mean_list,7)
        ct_skew_list = getSmooth(ct_skew_list,7)

    for i in range(num):
        for j in range(length):
            distrb = generate_distrb(ct_mean_list[j],ct_skew_list[j],group,i)
            for z in range(group):
                ct_all[z][j].append(distrb[z])

    for i in range(group):
        mid_all.append([])
        for j in range(length):
            median,lower,upper = calculate(ct_all[i][j])
            mid_all[i].append(median)

    for i in range(length):
        label.append(np.log(Rt_mean_list[i]))
        elem = []
        if input_size >= group:
            for j in range(group):
                elem.append(mid_all[j][i])
            if input_size == group + 1:
                elem.append(ct_mean_list[i])
            elif input_size == group + 2:
                elem.append(ct_mean_list[i])
                elem.append(ct_skew_list[i])
        else:
            if input_size == 1:
                elem.append(ct_mean_list[i])
            elif input_size == 2:
                elem.append(ct_mean_list[i])
                elem.append(ct_skew_list[i]) 
        data.append(elem)

    smooth = []
    for i in range(input_size):
        smooth.append([])
        for j in range(length):
            smooth[i].append(data[j][i])
        if input_size >= group:
            smooth[i] = (smooth[i] - ct_min[i]) / (ct_max[i] - ct_min[i])
        else:
            smooth[i] = (smooth[i] - ct_min[i + group]) / (ct_max[i + group] - ct_min[i + group])
        smooth[i] = getSmooth(smooth[i],7)
    for i in range(length):
        for j in range(input_size):
            data[i][j] = smooth[j][i]

    return data,label


def create_test_Dataset(args,data,label):
    seq = []
    num_workers=args.workers

    train_data = torch.FloatTensor(data)
    train_label = torch.FloatTensor(label)
    

    seq.append((train_data,train_label))

    seq = MyDataset(seq)

    seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

    return seq

def getTestData(args,num,smooth,ct_min,ct_max):
    date_list,records_list,Rt_mean_list,Rt_lower_list,Rt_upper_list,ct_mean_list,ct_skew_list = readData("test")
    data,label = generate_test_data(args,Rt_mean_list,ct_mean_list,ct_skew_list,num,smooth,ct_min,ct_max)
    seq = create_train_Dataset(args,data,label)
    return seq


        






    



    
