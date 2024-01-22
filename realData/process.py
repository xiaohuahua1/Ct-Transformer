import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import random

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
    
def readData():
    df = pd.read_csv('data_daily_all.csv')
    phase1 = df[36:109]

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

def createDataset(args,Rt_mean_list,ct_mean_list,ct_skew_list):
    Type = args.Dataset_type
    seq = []
    start = 0
    end = 0
    
    m,n = np.max(m),np.min(n)
    Rt_mean_list = (Rt_mean_list - n) / (m - n)

    m_m,n_m = np.max(ct_mean_list),np.min(ct_mean_list)
    m_s,n_s = np.max(ct_skew_list),np.min(ct_skew_list)
    ct_mean_list = (ct_mean_list - n_m) / (m_m - n_m)
    ct_skew_list = (ct_skew_list - n_s) / (m_s - n_s)

    if Type == 5:
        for i in range(0,len(Rt_mean_list) - args.seq_len - args.output_size + 1, args.output_size):
            train_seq = []
            train_label = []
            for j in range(i,i + args.seq_len):
                x = [Rt_mean_list[j]]
                train_seq.append(x)

            for j in range(i + args.seq_len, i + args.seq_len + args.output_size):
                train_label.append(Rt_mean_list[j])
                end += 1
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append(([train_seq, train_label],train_label))

    elif Type == 7:
        for i in range(0,len(Rt_mean_list) - args.seq_len - args.output_size + 1, args.output_size):
            train_seq = []
            train_label = []
            for j in range(i,i + args.seq_len):
                x = []
                x.append(ct_mean_list[j])
                x.append(ct_skew_list[j])
                train_seq.append(x)
                train_label.append(Rt_mean_list[j])
                end += 1
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append(([train_seq, train_label],train_label))

    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return seq,start,end,m,n

    



    
