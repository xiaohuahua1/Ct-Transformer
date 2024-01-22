import os
import random

import numpy as np
import torch
# from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from args import *
import random
import statsmodels.api as sm
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def stable(RtList,num):
    start = 0
    end = 0
    n = len(RtList)
    for i in range(n):
        if RtList[i] >= num:
            start = i
            break
    for i in range(n):
        if RtList[n - i - 1] >= num:
            end = n - i - 1
            break
    return start,end

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

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def process_distrb(distrb,group,i):
    all_group = 25
    elem = []

    if group == 3:
        for j in range(group):
            if j == 0:
                    # elem.append(distrb[all_group*i] + distrb[all_group*i + 1] + distrb[all_group*i + 2] + distrb[all_group*i + 3] + distrb[all_group*i + 4])
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1]+ distrb[all_group*i + 2] + distrb[all_group*i + 3] + distrb[all_group*i + 4] + distrb[all_group*i + 5] + distrb[all_group*i + 6] + distrb[all_group*i + 7])
            elif j == 1:
                elem.append(distrb[all_group*i + 8] + distrb[all_group*i + 9]+ distrb[all_group*i + 10] + distrb[all_group*i + 11] + distrb[all_group*i + 12] + distrb[all_group*i + 13] + distrb[all_group*i + 14] + distrb[all_group*i + 15])
            elif j == 2:
                elem.append(distrb[all_group*i + 16] + distrb[all_group*i + 17]+ distrb[all_group*i + 18] + distrb[all_group*i + 19] + distrb[all_group*i + 20] + distrb[all_group*i + 21] + distrb[all_group*i + 22] + distrb[all_group*i + 23] + distrb[all_group*i + 24])
    elif group == 4:
        for j in range(group):
            if j == 0:
                # elem.append(distrb[all_group*i] + distrb[all_group*i + 1] + distrb[all_group*i + 2] + distrb[all_group*i + 3] + distrb[all_group*i + 4])
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1]+ distrb[all_group*i + 2] + distrb[all_group*i + 3] + distrb[all_group*i + 4] + distrb[all_group*i + 5] + distrb[all_group*i + 6])
            elif j == 1:
                elem.append(distrb[all_group*i + 7] + distrb[all_group*i + 8]+ distrb[all_group*i + 9] + distrb[all_group*i + 10] + distrb[all_group*i + 11] + distrb[all_group*i + 12])
            elif j == 2:
                elem.append(distrb[all_group*i + 13] + distrb[all_group*i + 14]+ distrb[all_group*i + 15] + distrb[all_group*i + 16] + distrb[all_group*i + 17] + distrb[all_group*i + 18])
            elif j == 3:
                elem.append(distrb[all_group*i + 19] + distrb[all_group*i + 20]+ distrb[all_group*i + 21] + distrb[all_group*i + 22] + distrb[all_group*i + 23] + distrb[all_group*i + 24])

    elif group == 5:
        for j in range(group):
            if j == 0:
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1]+ distrb[all_group*i + 2] + distrb[all_group*i + 3] + distrb[all_group*i + 4])
            elif j == 1:
                elem.append(distrb[all_group*i + 5] + distrb[all_group*i + 6]+ distrb[all_group*i + 7] + distrb[all_group*i + 8] + distrb[all_group*i + 9])
            elif j == 2:
                elem.append(distrb[all_group*i + 10] + distrb[all_group*i + 11]+ distrb[all_group*i + 12] + distrb[all_group*i + 13] + distrb[all_group*i + 14])
            elif j == 3:
                elem.append(distrb[all_group*i + 15] + distrb[all_group*i + 16]+ distrb[all_group*i + 17] + distrb[all_group*i + 18] + distrb[all_group*i + 19])
            elif j == 4:
                elem.append(distrb[all_group*i + 20] + distrb[all_group*i + 21]+ distrb[all_group*i + 22] + distrb[all_group*i + 23] + distrb[all_group*i + 24])

    elif group == 6:
        for j in range(group):
            if j == 0:
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1]+ distrb[all_group*i + 2] + distrb[all_group*i + 3] + distrb[all_group*i + 4])
            elif j == 1:
                elem.append(distrb[all_group*i + 5] + distrb[all_group*i + 6]+ distrb[all_group*i + 7] + distrb[all_group*i + 8])
            elif j == 2:
                elem.append(distrb[all_group*i + 9] + distrb[all_group*i + 10]+ distrb[all_group*i + 11] + distrb[all_group*i + 12])
            elif j == 3:
                elem.append(distrb[all_group*i + 13] + distrb[all_group*i + 14]+ distrb[all_group*i + 15] + distrb[all_group*i + 16])
            elif j == 4:
                elem.append(distrb[all_group*i + 17] + distrb[all_group*i + 18]+ distrb[all_group*i + 19] + distrb[all_group*i + 20])
            elif j == 5:
                elem.append(distrb[all_group*i + 21] + distrb[all_group*i + 22]+ distrb[all_group*i + 23] + distrb[all_group*i + 24])

    elif group == 8:
        for j in range(group):
            if j == 0:
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1]+ distrb[all_group*i + 2] + distrb[all_group*i + 3])
            elif j == 1:
                elem.append(distrb[all_group*i + 4] + distrb[all_group*i + 5]+ distrb[all_group*i + 6])
            elif j == 2:
                elem.append(distrb[all_group*i + 7] + distrb[all_group*i + 8]+ distrb[all_group*i + 9])
            elif j == 3:
                elem.append(distrb[all_group*i + 10] + distrb[all_group*i + 11]+ distrb[all_group*i + 12])
            elif j == 4:
                elem.append(distrb[all_group*i + 13] + distrb[all_group*i + 14]+ distrb[all_group*i + 15])
            elif j == 5:
                elem.append(distrb[all_group*i + 16] + distrb[all_group*i + 17]+ distrb[all_group*i + 18])
            elif j == 6:
                elem.append(distrb[all_group*i + 19] + distrb[all_group*i + 20]+ distrb[all_group*i + 21])
            elif j == 7:
                elem.append(distrb[all_group*i + 22] + distrb[all_group*i + 23]+ distrb[all_group*i + 24])

    elif group == 12:
        for j in range(group):
            if j == 0:
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1]+ distrb[all_group*i + 2])
            elif j == 1:
                elem.append(distrb[all_group*i + 3] + distrb[all_group*i + 4])
            elif j == 2:
                elem.append(distrb[all_group*i + 5] + distrb[all_group*i + 6])
            elif j == 3:
                elem.append(distrb[all_group*i + 7] + distrb[all_group*i + 8])
            elif j == 4:
                elem.append(distrb[all_group*i + 9] + distrb[all_group*i + 10])
            elif j == 5:
                elem.append(distrb[all_group*i + 11] + distrb[all_group*i + 12])
            elif j == 6:
                elem.append(distrb[all_group*i + 13] + distrb[all_group*i + 14])
            elif j == 7:
                elem.append(distrb[all_group*i + 15] + distrb[all_group*i + 16])
            elif j == 8:
                elem.append(distrb[all_group*i + 17] + distrb[all_group*i + 18])
            elif j == 9:
                elem.append(distrb[all_group*i + 19] + distrb[all_group*i + 20])
            elif j == 10:
                elem.append(distrb[all_group*i + 21] + distrb[all_group*i + 22])
            elif j == 11:
                elem.append(distrb[all_group*i + 23] + distrb[all_group*i + 24])
                    
    return elem

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
            
def getInfo(group,fold,net,d,R0,Inum):
        
    path = fold + "/" + net + "/d=" + str(d) + "R=" + str(R0) + "/train"
    # path = fold + "d=" + str(d) + "R=" + str(R0) + "\\train"
    max_len = 0

    rF = open(path + "/Rt.txt")

    liner = rF.readline()

    while liner:
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')

        if len(r_str) > 50:
            rt = [float(r_str[i]) for i in range(len(r_str))]
            if len(rt) > max_len:
                max_len = len(rt)
        liner = rF.readline()

    rF.close()

    rt_by_day = []
    ct_by_day = []
    mean_by_day = []
    skew_by_day = []
    I_by_day = []

    for i in range(group):
        ct_by_day.append([])

    rt_mean = []
    ct_mean = []
    mean_mean = []
    skew_mean = []
    I_mean = []

    for i in range(group):
        ct_mean.append([])

    for i in range(max_len):
        rt_by_day.append([])
        mean_by_day.append([])
        skew_by_day.append([])
        I_by_day.append([])
        for j in range(group):
            ct_by_day[j].append([])

    rF = open(path + "/Rt.txt")
    IF = open(path + "/I.txt")
    diF = open(path + "/distrb.txt")
    mF = open(path + "/ctMean.txt")
    sF = open(path + "/ctSkew.txt")

    liner = rF.readline()
    lineI = IF.readline()
    linedi = diF.readline()
    linem = mF.readline()
    lines = sF.readline()

    while liner:
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        lineI = lineI.split(' \n')[0]
        I_str = lineI.split(' ')
        linedi = linedi.split(' \n')[0]
        di_str = linedi.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        if len(r_str) > 50:
            rt = [float(r_str[i]) for i in range(len(r_str))]
            I = [float(I_str[i]) for i in range(len(I_str))]
            distrb = [float(di_str[i]) for i in range(len(di_str))]
            mean = [float(m_str[i]) for i in range(len(m_str))]
            skew= [float(s_str[i]) for i in range(len(s_str))]
            for i in range(len(rt)):
                if rt[i] <= 0.05:
                    rt[i] = 0.05

            for i in range(len(rt)):
                rt_by_day[i].append(rt[i])
                mean_by_day[i].append(mean[i])
                skew_by_day[i].append(skew[i])
                I_by_day[i].append(I[i])
                elem = process_distrb(distrb,group,i)
                for j in range(group):
                    ct_by_day[j][i].append(elem[j])

        liner = rF.readline()
        lineI = IF.readline()
        linedi = diF.readline()
        linem = mF.readline()
        lines = sF.readline()

    rF.close()
    IF.close()
    diF.close()
    mF.close()
    sF.close()

    for i in range(max_len):
        median,lower,upper = calculate(rt_by_day[i])
        rt_mean.append(median)

        median,lower,upper = calculate(mean_by_day[i])
        mean_mean.append(median)

        median,lower,upper = calculate(skew_by_day[i])
        skew_mean.append(median)

        median,lower,upper = calculate(I_by_day[i])
        I_mean.append(median)

        for j in range(group):
            median,lower,upper = calculate(ct_by_day[j][i])
            ct_mean[j].append(median)
            

    start,end = stable(I_mean,Inum) 

    rt_mean = rt_mean[start:end]
    rt_mean = np.log(rt_mean)
    mean_mean = mean_mean[start:end]
    skew_mean = skew_mean[start:end]

    for i in range(group):
        ct_mean[i] = ct_mean[i][start:end]
        
    return start,end,rt_mean,mean_mean,skew_mean,ct_mean

def getInfo_s(group,fold,net,d,R0,Inum):
    path = fold + "/" + net + "/d=" + str(d) + "R=" + str(R0) + "/train"

    rt_all = []
    mean_all = []
    skew_all = []
    ct_all = []

    for i in range(group):
        ct_all.append([])
    
    rF = open(path + "\\Rt.txt")
    IF = open(path + "\\I.txt")
    diF = open(path + "\\distrb.txt")
    mF = open(path + "\\ctMean.txt")
    sF = open(path + "\\ctSkew.txt")

    liner = rF.readline()
    lineI = IF.readline()
    linedi = diF.readline()
    linem = mF.readline()
    lines = sF.readline()

    while liner:
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        lineI = lineI.split(' \n')[0]
        I_str = lineI.split(' ')
        linedi = linedi.split(' \n')[0]
        di_str = linedi.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        if len(r_str) > 50:
            rt = [float(r_str[i]) for i in range(len(r_str))]
            I = [float(I_str[i]) for i in range(len(I_str))]
            distrb = [float(di_str[i]) for i in range(len(di_str))]
            mean = [float(m_str[i]) for i in range(len(m_str))]
            skew= [float(s_str[i]) for i in range(len(s_str))]
            for i in range(len(rt)):
                if rt[i] <= 0.05:
                    rt[i] = 0.05
            start,end = stable(I,Inum)
            if start != end:
                for i in range(start,end):
                    rt_all.append(rt[i])
                    mean_all.append(mean[i])
                    skew_all.append(skew[i])
                    elem = process_distrb(distrb,group,i)
                    for j in range(group):
                        ct_all[j].append(elem[j])
        liner = rF.readline()
        lineI = IF.readline()
        linedi = diF.readline()
        linem = mF.readline()
        lines = sF.readline()

    rF.close()
    IF.close()
    diF.close()
    mF.close()
    sF.close()
    return rt_all,mean_all,skew_all,ct_all



def getBound(args,fold,d,R0,Inum,type):
    
    max_list = []
    min_list = []
    min_rt = 0
    max_rt = 0
    group = args.group

    ct_all = []
    for i in range(group):
        ct_all.append([])
    mean_all = []
    skew_all = []
    rt_all = []

    net = "ER"
    if type == 1:
        for i in range(len(d)):
            for j in range(len(R0)):
                start,end,rt_mean,mean_mean,skew_mean,ct_mean = getInfo(group,fold,net,d[i],R0[j],Inum)
                mean_all.extend(mean_mean)
                skew_all.extend(skew_mean)
                rt_all.extend(rt_mean)

                for z in range(group):
                    ct_all[z].extend(ct_mean[z])
    elif type == 2:
        for i in range(len(d)):
            for j in range(len(R0)):
                rt_s,mean_s,skew_s,ct_s = getInfo_s(group,fold,net,d[i],R0[j],Inum)
                rt_all.extend(rt_s)
                mean_all.extend(mean_s)
                skew_all.extend(skew_s)
                for z in range(group):
                    ct_all[z].extend(ct_s[z])
    for i in range(group):
        min_mean,max_mean = np.min(ct_all[i]),np.max(ct_all[i])
        max_list.append(max_mean)
        min_list.append(min_mean)

    min_mean,max_mean = np.min(mean_all),np.max(mean_all)
    max_list.append(max_mean)
    min_list.append(min_mean)

    min_mean,max_mean = np.min(skew_all),np.max(skew_all)
    max_list.append(max_mean)
    min_list.append(min_mean)

    min_rt,max_rt = np.min(rt_all),np.max(rt_all)
    return max_list,min_list,max_rt,min_rt
            


def process(args,distrb,mean,skew,Rt,start,end,max_list,min_list,max_rt,min_rt):
    t_data = []
    t_label = []


    group = args.group
    input_size = args.input_size

    for i in range(start,end):
        t_label.append(Rt[i])
        # elem = []
        if input_size >= group:
            elem = process_distrb(distrb,group,i)
            if input_size == group + 1:
                elem.append(mean[i])
            elif input_size == group + 2:
                elem.append(mean[i])
                elem.append(skew[i])
        else:
            elem = []
            if input_size == 1:
                elem.append(mean[i])
            elif input_size == 2:
                elem.append(mean[i])
                elem.append(skew[i])
        t_data.append(elem)
    # t_data = np.array(t_data)
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(t_data)
    # pca = PCA(n_components=2)  # 设置要保留的主成分数量为2
    # data_pca = pca.fit_transform(data_scaled)
    # data_pca = data_pca.tolist()
    smooth = []
    for i in range(input_size):
        smooth.append([])
        for j in range(len(t_data)):
            smooth[i].append(t_data[j][i])
        # if input_size >= group:
        #     smooth[i] = (smooth[i] - min_list[i]) / (max_list[i] - min_list[i])
        # else:
        #     smooth[i] = (smooth[i] - min_list[i + group]) / (max_list[i + group] - min_list[i + group])
        smooth[i] = getSmooth(smooth[i],7)
    for i in range(len(t_data)):
        for j in range(len(t_data[0])):
            t_data[i][j] = smooth[j][i]

    for i in range(len(t_label)):
        if t_label[i] <= 0.05:
            t_label[i] = 0.05    
    t_label = np.log(t_label)
    # t_label = (t_label - min_rt) / (max_rt - min_rt)
    
    return t_data,t_label

def read_train_data(path,args,Inum,smooth,max_list,min_list,max_rt,min_rt):
    data = []
    label = []
    dF = open(path + "\\distrb.txt")
    rF = open(path + "\\Rt.txt")
    iF = open(path + "\\I.txt")
    mF = open(path + "\\ctMean.txt")
    sF = open(path + "\\ctSkew.txt")

    lined = dF.readline()
    linei = iF.readline()
    liner = rF.readline()
    linem = mF.readline()
    lines = sF.readline()

    count = 0
    
    while lined:
        lined = lined.split(' \n')[0]
        d_str = lined.split(' ')
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ')
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        I = [float(i_str[i]) for i in range(len(i_str))]
        distrb = [float(d_str[i]) for i in range(len(d_str))]
        Rt = [float(r_str[i]) for i in range(len(r_str))]
        mean = [float(m_str[i]) for i in range(len(m_str))]
        skew= [float(s_str[i]) for i in range(len(s_str))]

        if len(I) > 50:
            # count += 1
            if smooth:
                Rt = getSmooth(Rt,7)
            start,end = stable(I,Inum)
            if start != end:
                count += 1
                t_data,t_label = process(args,distrb,mean,skew,Rt,start,end,max_list,min_list,max_rt,min_rt)
                data.append(t_data)
                label.append(t_label)

        lined = dF.readline()
        linei = iF.readline()
        liner = rF.readline()
        linem = mF.readline()
        lines = sF.readline()
    
    dF.close()
    iF.close()
    rF.close()
    mF.close()
    sF.close()

    return data,label,count

def create_train_dataset(args,data,label,count):
    type = args.Dataset_type
    group = args.group
    batch_size = args.batch_size
    min_len = 100000

    seq = []

    n = len(data)
    if count != 0:
        n = count - 1
        # print(n)
        # print(len(data))
    if type == 1:
        output_size = args.output_size
        for z in range(n):
            data_s = data[z]

            length = len(data_s)
            if length < min_len:
                min_len = length

        for z in range(n):
            data_s = data[z]
            label_s = label[z]

            train_seq = data_s[:min_len]
            train_label = label_s[:min_len]

            # label_high = []
            # for i in range(len(train_label)):
            #     label_value = train_label[i]
            #     label_value_list = []
            #     for j in range(output_size):
            #         label_value_list.append(np.power(label_value,j + 1))
            #     label_high.append(label_value_list)

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)

            seq.append((train_seq,train_label))

            # print(train_seq.shape)
            # print(train_label.shape)
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
        
    elif type == 2:
        for z in range(n):
            data_s = data[z]

            length = len(data_s)
            if length < min_len:
                min_len = length

        for z in range(n):
            data_s = data[z]
            label_s = label[z]

            train_seq = data_s[:min_len]
            train_label = label_s[:min_len]

            train_label_input = [1.0]
            for x in range(len(train_label)):
                train_label_input.append(train_label[x])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            train_label_input = torch.FloatTensor(train_label_input)

            seq.append(([train_seq,train_label_input],train_label))
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
        
    elif type == 3:
        pre_step = args.num_encoder_steps
        for z in range(n):
            data_s = data[z]

            length = len(data_s)
            if length < min_len:
                min_len = length

        for z in range(n):
            data_s = data[z]
            label_s = label[z]

            # train_seq = data_s[:20]
            # train_label = label_s[:20]

            train_seq = data_s[:min_len]
            train_label = label_s[:min_len]
            train_label = train_label[pre_step:]
            
            # min_num,max_num = np.min(train_label),np.max(train_label)
            # train_label = (train_label - min_num) / (max_num - min_num)

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)

            seq.append((train_seq,train_label))

            # print(train_seq.shape)
            # print(train_label.shape)
        
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
    
    return seq

class GroupBatchRandomSampler(object):
    def __init__(self, data_groups):
        self.batch_indices = []
        for data_group in data_groups:
            self.batch_indices.extend(list(data_group))

    def __iter__(self):
        return (self.batch_indices[i] for i in torch.randperm(len(self.batch_indices)))

    def __len__(self):
        return len(self.batch_indices)
            
def get_Train_range(args,pre,net,d,R,Inum,smooth,max_list,min_list,max_rt,min_rt):

    Dtr_all = []
    Dva_all = []

    train_data_all = []
    train_label_all = []

    count_max = 10000

    for i in range(len(net)):
        for j in range(len(d)):
            for z in range(len(R)):
                fold = pre + "\\" + net[i] + "\\d=" + str(d[j]) + "R=" + str(R[z])
                val_path = fold + "\\val"
                train_path = fold + "\\train"
                val_data,val_label,_ = read_train_data(val_path,args,Inum,smooth,max_list,min_list,max_rt,min_rt)
                Dva_s = create_train_dataset(args,val_data,val_label,0)
                # Dtr_s,Dva_s = getTrainData(args,train_path,val_path,Inum,smooth)
                # Dtr_all.append(Dtr_s)
                Dva_all.append(Dva_s)

                train_data,train_label,count = read_train_data(train_path,args,Inum,smooth,max_list,min_list,max_rt,min_rt)
                # print(count)
                if count < count_max:
                    count_max = count
                train_data_all.append(train_data)
                train_label_all.append(train_label)
    # print(count_max)
    for i in range(len(train_data_all)):
        Dtr_s = create_train_dataset(args,train_data_all[i],train_label_all[i],count_max)
        Dtr_all.append(Dtr_s)

    Dtr = GroupBatchRandomSampler(Dtr_all)
    Dva = GroupBatchRandomSampler(Dva_all)

    return Dtr,Dva,Dtr_all,Dva_all


def read_test_data(path,args,Inum,smooth,text,max_list,min_list,max_rt,min_rt):
    data = []
    label = []
    valid_text = []
    duration = []
    dF = open(path + "\\distrb.txt")
    rF = open(path + "\\Rt.txt")
    iF = open(path + "\\I.txt")
    mF = open(path + "\\ctMean.txt")
    sF = open(path + "\\ctSkew.txt")

    lined = dF.readline()
    linei = iF.readline()
    liner = rF.readline()
    linem = mF.readline()
    lines = sF.readline()

    count = 0

    while lined:
        lined = lined.split(' \n')[0]
        d_str = lined.split(' ')
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ')
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        I = [float(i_str[i]) for i in range(len(i_str))]
        distrb = [float(d_str[i]) for i in range(len(d_str))]
        Rt = [float(r_str[i]) for i in range(len(r_str))]
        mean = [float(m_str[i]) for i in range(len(m_str))]
        skew= [float(s_str[i]) for i in range(len(s_str))]

        if len(I) > 50:
            valid_text.append(text[count])
            if smooth:
                Rt = getSmooth(Rt,7)
            start,end = stable(I,Inum)
            if start != end:
                t_data,t_label = process(args,distrb,mean,skew,Rt,start,end,max_list,min_list,max_rt,min_rt)
                data.append(t_data)
                label.append(t_label)
                duration.append([start,end])

        count += 1
        
        lined = dF.readline()
        linei = iF.readline()
        liner = rF.readline()
        linem = mF.readline()
        lines = sF.readline()
    
    dF.close()
    iF.close()
    rF.close()
    mF.close()
    sF.close()
    
    return data,label,valid_text,duration

def read_testRate_data(path,args,Inum,smooth,text,max_list,min_list,max_rt,min_rt,id):
    data = []
    label = []
    valid_text = []
    duration = []

    meanList = []
    skewList = []
    distrbList = []

    means = []
    skews = []
    distrbs = []

    dF = open(path + "\\distrb.txt")
    mF = open(path + "\\ctMean.txt")
    sF = open(path + "\\ctSkew.txt")

    lined = dF.readline()
    linem = mF.readline()
    lines = sF.readline()

    count = 0
    while lined:
        lined = lined.split(' \n')[0]
        d_str = lined.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        distrb = [float(d_str[i]) for i in range(len(d_str))]
        mean = [float(m_str[i]) for i in range(len(m_str))]
        skew= [float(s_str[i]) for i in range(len(s_str))]

        if len(mean) > 50:
            means.append(mean)
            skews.append(skew)
            distrbs.append(distrb)

            if count % 5 == 4:
                meanList.append(means)
                skewList.append(skews)
                distrbList.append(distrbs)

                means = []
                skews = []
                distrbs = []

        count += 1
        lined = dF.readline()
        linem = mF.readline()
        lines = sF.readline()

    dF.close()
    mF.close()
    sF.close()

    rF = open(path + "\\Rt.txt")
    iF = open(path + "\\I.txt")

    linei = iF.readline()
    liner = rF.readline()

    count = 0
    while linei:
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ')
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')

        I = [float(i_str[i]) for i in range(len(i_str))]
        Rt = [float(r_str[i]) for i in range(len(r_str))]

        if len(I) > 50:
            valid_text.append(text[count])
            if smooth:
                Rt = getSmooth(Rt,7)
            start,end = stable(I,Inum)
            if start != end:
                t_data,t_label = process(args,distrbList[count][id],meanList[count][id],skewList[count][id],Rt,start,end,max_list,min_list,max_rt,min_rt)
                data.append(t_data)
                label.append(t_label)
                duration.append([start,end])
        count += 1
        linei = iF.readline()
        liner = rF.readline()
    iF.close()
    rF.close()

    
    return data,label,valid_text,duration

  
def create_test_dataset(args,data,label,duration):
    type = args.Dataset_type
    # duration_t = []

    seq_all = []
    
    if type == 1:
        output_size = args.output_size
        pre_step = args.num_encoder_steps
        for z in range(len(data)):
            seq = []
            train_seq = data[z]
            train_label = label[z]

            train_seq = train_seq[pre_step:]
            train_label = train_label[pre_step:]
            duration[z][0]  = duration[z][0] + pre_step
            # duration_t.append(duration[z])

            # label_high = []
            # for i in range(len(train_label)):
            #     label_value = train_label[i]
            #     label_value_list = []
            #     for j in range(output_size):
            #         label_value_list.append(np.power(label_value,j + 1))
            #     label_high.append(label_value_list)

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq,train_label))
            
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=True)
        
            seq_all.append(seq)
            
    elif type == 2:
        for z in range(len(data)):
            seq = []
            train_seq = data[z]
            train_label = label[z]
            # duration_t.append(duration[z])
            
            train_label_input = [1.0]
            for x in range(len(train_label)):
                train_label_input.append(train_label[x])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            train_label_input = torch.FloatTensor(train_label_input)
            seq.append(([train_seq,train_label_input],train_label))
            
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=True)
        
            seq_all.append(seq)
            
    elif type == 3:
        pre_step = args.num_encoder_steps
        for z in range(len(data)):
            seq = []
            train_seq = data[z]
            train_label = label[z]
            # duration_t.append(duration[z])

            # train_seq = train_seq[:20]
            # train_label = train_label[:20]

            train_label = train_label[pre_step:]
            duration[z][0]  = duration[z][0] + pre_step
            
            # min_num,max_num = np.min(train_label),np.max(train_label)
            # train_label = (train_label - min_num) / (max_num - min_num)
            
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label)
            seq.append((train_seq,train_label))
            
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=args.workers, drop_last=True)
        
            seq_all.append(seq)

    return seq_all,duration


def getTestData(args,test_path,Inum,smooth,text,max_list,min_list,max_rt,min_rt):

    test_data,test_label,valid_text,duration = read_test_data(test_path,args,Inum,smooth,text,max_list,min_list,max_rt,min_rt)
    # print(duration[0][0])
    Dte,duration = create_test_dataset(args,test_data,test_label,duration)
    # print(duration[0][0])
    # print(duration_t[0][0])
    
    return Dte,valid_text,duration

def getTestRateData(args,test_path,Inum,smooth,text,max_list,min_list,max_rt,min_rt,id):

    test_data,test_label,valid_text,duration = read_testRate_data(test_path,args,Inum,smooth,text,max_list,min_list,max_rt,min_rt,id)
    # print(duration[0][0])
    Dte,duration = create_test_dataset(args,test_data,test_label,duration)
    # print(duration[0][0])
    # print(duration_t[0][0])
    
    return Dte,valid_text,duration

def getRegression(test_path,Inum,pre_step):
    Rpre = []
    duration = []

    rF = open(test_path + "\\Rt.txt")
    iF = open(test_path + "\\I.txt")
    mF = open(test_path + "\\ctMean.txt")
    sF = open(test_path + "\\ctSkew.txt")

    liner = rF.readline()
    linei = iF.readline()
    linem = mF.readline()
    lines = sF.readline()

    while liner:
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        I = [float(i_str[i]) for i in range(len(i_str))]

        if len(I) > 50:
            Rt = [float(r_str[i]) for i in range(len(r_str))]
            mean = [float(m_str[i]) for i in range(len(m_str))]
            skew = [float(s_str[i]) for i in range(len(s_str))]
            Rt = getSmooth(Rt,7)
            for i in range(len(Rt)):
                if Rt[i] <= 0.05:
                    Rt[i] = 0.05
            Rt_log = np.log(Rt)
            # Rt_log = Rt
            start,end = stable(I,Inum)
            if start != end:
                result = []
                # Rt = Rt[start:end]
                Rt_log = Rt_log[start:end]
                mean = mean[start:end]
                skew = skew[start:end]
                n = len(Rt_log)
                train_n = int(0.4*n)
                # Rt_t = Rt[start:start+train_n]
                Rt_log_t = Rt_log[start:start+train_n]
                mean_t = mean[start:start+train_n]
                skew_t = skew[start:start+train_n]

                # rt_train = pd.Series(Rt)
                # mean_train = pd.Series(mean)
                # skew_train = pd.Series(skew)
                rt_train = pd.Series(Rt_log_t)
                mean_train = pd.Series(mean_t)
                skew_train = pd.Series(skew_t)

                x_train = np.column_stack((mean_train,skew_train))
                x11_train = sm.add_constant(x_train)
                model = sm.OLS(rt_train,x11_train).fit()

                for j in range(n):
                    rt_p = model.params[0] + mean[j]*model.params[1] + skew[j]*model.params[2]
                    result.append(np.exp(rt_p))
                    # if rt_p < 0:
                    #     rt_p = 0
                    # result.append(rt_p)
                result = result[pre_step:]
                result = getSmooth(result,7)
                Rpre.append(result)
                duration.append((start + pre_step,end))
        
        liner = rF.readline()
        linei = iF.readline()
        linem = mF.readline()
        lines = sF.readline()
    rF.close()
    iF.close()
    mF.close()
    sF.close()
    return Rpre,duration

def getBayes(test_path,Inum,pre_step):

    Ts = []
    M = []
    U = []
    L = []
    R = []

    tF = open(test_path + "\\Tstart.txt")
    iF = open(test_path + "\\I.txt")
    mF = open(test_path + "\\baseMean.txt")
    uF = open(test_path + "\\baseUpper.txt")
    lF = open(test_path + "\\baseLower.txt")
    rF = open(test_path + "\\Rt.txt")

    linet = tF.readline()
    linei = iF.readline()
    linem = mF.readline()
    lineu = uF.readline()
    linel = lF.readline()
    liner = rF.readline()

    while linet:
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ') 
        linet = linet.split(' \n')[0]
        t_str = linet.split(' ') 
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lineu = lineu.split(' \n')[0]
        u_str = lineu.split(' ')
        linel = linel.split(' \n')[0]
        l_str = linel.split(' ')
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')

        I = [float(i_str[i]) for i in range(len(i_str))] 
        if len(I) > 50:
            mean = [float(m_str[i]) for i in range(len(m_str))] 
            lower = [float(l_str[i]) for i in range(len(l_str))]
            upper =  [float(u_str[i]) for i in range(len(u_str))]
            tstart = [int(t_str[i]) for i in range(len(t_str))]
            Rt = [float(r_str[i]) for i in range(len(r_str))]
            start,end = stable(I,Inum)
            if start != end:
                start_s = start - tstart[0]
                end_s = end - tstart[0]
                Rt = Rt[start + pre_step:end]
                mean = mean[start_s + pre_step:end_s]
                upper = upper[start_s + pre_step:end_s]
                lower = lower[start_s + pre_step:end_s]


                M.append(mean)
                U.append(upper)
                L.append(lower)
                R.append(Rt)
                Ts.append((start + pre_step,end))

        linet = tF.readline()
        linei = iF.readline()
        linem = mF.readline()
        lineu = uF.readline()
        linel = lF.readline()
        liner = rF.readline()

    tF.close()
    iF.close()
    mF.close()
    uF.close()
    lF.close()
    rF.close()  
    return Ts,M,U,L,R

def getViro(test_path,Inum,pre_step):
    Vpre = []
    duration = []

    rF = open(test_path + "\\Rt.txt")
    iF = open(test_path + "\\I.txt")
    vF = open(test_path + "\\viroRt.txt")

    liner = rF.readline()
    linei = iF.readline()
    linev = vF.readline()

    while liner:
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ')
        linev = linev.split(' \n')[0]
        v_str = linev.split(' ')

        I = [float(i_str[i]) for i in range(len(i_str))] 
        if len(I) > 50:
            Rt = [float(r_str[i]) for i in range(len(r_str))]
            viroRt = [float(v_str[i]) for i in range(len(v_str))]
            start,end = stable(I,Inum)
            if start != end:
                Rt = Rt[start:end]
                viroRt = viroRt[start:end]

                Vpre.append(viroRt)
                duration.append((start + pre_step,end))
        liner = rF.readline()
        linei = iF.readline()
        linev = vF.readline()

    rF.close()
    iF.close()
    vF.close()

    return Vpre,duration




