import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from args import *
import random
import statsmodels.api as sm

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
    all_group = 23
    elem = []
    if group == 6:
        for j in range(group):
            if j == 0:
                elem.append(distrb[all_group*i] + distrb[all_group*i + 1] + distrb[all_group*i + 2])
            else:
                elem.append(distrb[all_group*i + 3 + 4*(j - 1)] + distrb[all_group*i + 3 + 4*(j - 1) + 1] + distrb[all_group*i + 3 + 4*(j - 1) + 2] + distrb[all_group*i + 3 + 4*(j - 1) + 3])
    elif group == 12:
        for j in range(group):
            if j == 0:
                elem.append(distrb[all_group*i])
            else:
                elem.append(distrb[all_group*i + 1 + 2*(j - 1)] + distrb[all_group*i + 1 + 2*(j - 1) + 1])
    elif group == 18:
        for j in range(group):
            if j == 0 or j == 1:
                elem.append(distrb[all_group*i + 2*j] + distrb[all_group*i + 2*j + 1])
            elif j >= 2 and j <= 15:
                elem.append(distrb[all_group*i + j + 2])
            elif j == 16:
                elem.append(distrb[all_group*i + 18] + distrb[all_group*i + 19])
            elif j == 17:
                elem.append(distrb[all_group*i + 20] + distrb[all_group*i + 21]  + distrb[all_group*i + 22])
    return elem
            

def process(Rt,distrb,mean,skew,start,end,args,label_all):
    t_data = []
    t_label = []
    m_m,n_m = np.max(mean),np.min(mean)
    m_s,n_s = np.max(skew),np.min(skew)
    mean = (mean - n_m) / (m_m - n_m)
    skew = (skew - n_s) / (m_s - n_s)
    for i in range(start,end):
        label_all.append(Rt[i])
        t_label.append(Rt[i])
        if args.readDate_type == 2:
            # elem = []
            # for j in range(args.group):
            #     elem.append(distrb[args.group*i + j])
            elem = process_distrb(distrb,args.group,i)
            print(len(elem))
            t_data.append(elem)
        if args.readDate_type == 3:
            elem = [mean[i],skew[i]]
            t_data.append(elem)    
    return t_data,t_label

def avertByweek(Rt,distrb,mean,skew):
    all_group = 23
    Rt_week = []
    distrb_week = []
    mean_week = []
    skew_week = []

    weekNum = int(len(Rt)/7)
    for i in range(weekNum):
        ave_rt = 0.0
        ave_mean = 0.0
        ave_skew = 0.0
        ave_distrb = []
        for z in range(all_group):
            ave_distrb.append(0.0)

        for j in range(7):
            index = i*7 + j
            ave_rt += Rt[index]
            ave_mean += mean[index]
            ave_skew += skew[index]

            for z in range(all_group):
                ave_distrb[z] += distrb[index*all_group + z]

        ave_rt /= 7.0
        ave_mean /= 7.0
        ave_skew /= 7.0

        Rt_week.append(ave_rt)
        mean_week.append(ave_mean)
        skew_week.append(ave_skew)

        for z in range(all_group):
            ave_distrb[z] /= 7.0
            distrb_week.append(ave_distrb[z])

    return Rt_week,distrb_week,mean_week,skew_week


def read_data_fixed(path,args,text,Inum,smooth):
    data = []
    label = []
    label_all = []
    R0 = []
    duration = []
    flag = True
    if len(text) == 0:
        flag = False

    dF = open(path + "\\distrb.txt")
    rF = open(path + "\\Rt.txt")
    iF = open(path + "\\I.txt")
    mF = open(path + "\\ctMean.txt")
    sF = open(path + "\\ctSkew.txt")

    count = 0

    lined = dF.readline()
    liner = rF.readline()
    linei = iF.readline()
    linem = mF.readline()
    lines = sF.readline()

    while lined:
        lined = lined.split(' \n')[0]
        d_str = lined.split(' ')
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        linei = linei.split(' \n')[0]
        i_str = linei.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')

        Rt = [float(r_str[i]) for i in range(len(r_str))]
        I = [float(i_str[i]) for i in range(len(i_str))]
        distrb = [float(d_str[i]) for i in range(len(d_str))]
        mean = [float(m_str[i]) for i in range(len(m_str))]
        skew = [float(s_str[i]) for i in range(len(s_str))]

        if len(I) > 50 :
            if flag:
                R0.append(text[count])
            if smooth:
                Rt = getSmooth(Rt,3)
                mean = getSmooth(mean,3)
                skew = getSmooth(skew,3)
            # print(str(len(I)) +' ' + str(len(distrb)) + ' ' + str(6*len(I)))
            start,end = stable(I,Inum)
            if start != end:
                t_data,t_label = process(Rt,distrb,mean,skew,start,end,args,label_all)
                data.append(t_data)
                label.append(t_label)
                duration.append((start,end))

        count += 1
        lined = dF.readline()
        liner = rF.readline()
        linei = iF.readline()
        linem = mF.readline()
        lines = sF.readline()

    dF.close()
    rF.close()
    iF.close()
    mF.close()
    sF.close()

    return data,label,label_all,R0,duration

def createDataset(args,dataset,dataset_l,step_size, shuffle,m,n,duration):
    type = args.Dataset_type
    group = args.group
    seq = []
    l = []
    start = 0
    end = 0
    d = []
    flag = True
    if len(duration) == 0:
        flag = False
    if type == 1:
        for z in range(len(dataset_l)):
            data = dataset_l[z]

            data = (data - n) / (m - n)
            cnt = 0

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                start += args.seq_len
                end = start

            for i in range(0,len(data) - args.seq_len - args.output_size + 1, step_size):
                train_seq = []
                train_label = []
                cnt += step_size
                for j in range(i,i + args.seq_len):
                    x = [data[j]]
                    train_seq.append(x)

                for j in range(i + args.seq_len, i + args.seq_len + args.output_size):
                    train_label.append(data[j])
                    end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append((train_seq, train_label))
            l.append(cnt)
            if flag:
                d.append((start,end))

    elif type == 2:
        for z in range(len(dataset)):
            cnt = 0
            data = dataset[z]
            label = dataset_l[z]

            label = (label - n) / (m - n)

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                start += args.seq_len
                end = start

            for i in range(0,len(data) - args.seq_len - args.output_size + 1, step_size):
                train_seq = []
                train_label = []
                cnt += step_size
                for j in range(i,i + args.seq_len):
                    x = [label[j]]
                    for c in range(group):
                        x.append(data[j][c])
                    train_seq.append(x)

                for j in range(i + args.seq_len, i + args.seq_len + args.output_size):
                    train_label.append(label[j])
                    end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append((train_seq, train_label))
            l.append(cnt)
            if flag:
                d.append((start,end))
    
    elif type == 3:
        for z in range(len(dataset)):
            data = dataset[z]
            label = dataset_l[z]
            cnt = 0

            label = (label - n) / (m - n)

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                start += args.seq_len
                end = start

            for i in range(0,len(data)):
                train_seq = []
                train_label = []
                cnt += step_size

                train_seq = data[i]
                train_label.append(label[i])
                end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append((train_seq, train_label))
            l.append(cnt)

            if flag:
                d.append((start,end))

    elif type == 4:

        for z in range(len(dataset)):
            data = dataset[z]
            label = dataset_l[z]
            cnt = 0

            label = (label - n) / (m - n)

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                start += args.seq_len
                end = start

            for i in range(0,len(data) - args.seq_len - args.output_size + 1, step_size):
                train_seq = []
                train_label = []
                cnt += step_size
                for j in range(i,i + args.seq_len):
                    x = []
                    for c in range(group):
                        x.append(data[j][c])
                    train_seq.append(x)

                for j in range(i + args.seq_len, i + args.seq_len + args.output_size):
                    train_label.append(label[j])
                    end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append((train_seq, train_label))
            l.append(cnt)

            if flag:
                d.append((start,end))
    elif type == 5:
        for z in range(len(dataset_l)):
            data = dataset_l[z]

            data = (data - n) / (m - n)
            cnt = 0

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                start += args.seq_len
                end = start

            for i in range(0,len(data) - args.seq_len - args.output_size + 1, step_size):
                train_seq = []
                train_label = []
                cnt += step_size
                for j in range(i,i + args.seq_len):
                    x = [data[j]]
                    train_seq.append(x)

                for j in range(i + args.seq_len, i + args.seq_len + args.output_size):
                    train_label.append(data[j])
                    end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append(((train_seq, train_label),train_label))
            l.append(cnt)
            if flag:
                d.append((start,end))

    elif type == 6:

        for z in range(len(dataset)):
            data = dataset[z]
            label = dataset_l[z]
            cnt = 0
            dim = len(data[0])

            label = (label - n) / (m - n)

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                start += args.seq_len
                end = start

            for i in range(0,len(data) - args.seq_len - args.output_size + 1, step_size):
                train_seq = []
                train_label = []
                train_ct = []
                cnt += step_size
                for j in range(i,i + args.seq_len):
                    x = [label[j]]
                    train_seq.append(x)

                for j in range(i + args.seq_len, i + args.seq_len + args.output_size):
                    train_label.append(label[j])
                    x = []
                    for c in range(dim):
                        x.append(data[j][c])
                    train_ct.append(x)
                    end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_ct = torch.FloatTensor(train_ct)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append(([train_seq, train_label,train_ct],train_label))
            l.append(cnt)

            if flag:
                d.append((start,end))

    elif type == 7:

        for z in range(len(dataset)):
            data = dataset[z]
            label = dataset_l[z]
            cnt = 0
            dim = len(data[0])

            label = (label - n) / (m - n)

            if flag:
                start = duration[z][0]
                end = duration[z][1]
                # start += args.seq_len
                end = start

            for i in range(0,len(data) - args.seq_len + 1, step_size):
                train_seq = []
                train_label = []
                cnt += step_size
                for j in range(i,i + args.seq_len):
                    x = []
                    for c in range(dim):
                        x.append(data[j][c])
                    train_seq.append(x)
                    train_label.append(label[j])
                    end += 1

                train_seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seq.append(([train_seq, train_label],train_label))
            l.append(cnt)

            if flag:
                d.append((start,end))
        


    seq = MyDataset(seq)
    seq = DataLoader(dataset=seq, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, drop_last=False)

    return seq,d,l

def getTrainData(args,train_path,val_path,text,Inum,smooth):

    train,train_l,label_all,_,_ = read_data_fixed(train_path,args,text,Inum,smooth)
    val,val_l,_,_,_ = read_data_fixed(val_path,args,text,Inum,smooth)

    m, n = np.max(label_all), np.min(label_all)

    Dtr,_,_ = createDataset(args,train,train_l ,step_size=1,shuffle=True, m=m, n=n,duration=[])
    Val,_,_ = createDataset(args,val, val_l, step_size=1, shuffle=True, m=m, n=n,duration=[])


    return Dtr,Val,m,n

def getTestData_fixed(args,test_path,m,n,text,Inum):
    test,test_l,_,R0,duration = read_data_fixed(test_path,args,text,Inum,False)


    Dte,d,l = createDataset(args,test, test_l, step_size=args.output_size, shuffle=False, m=m, n=n,duration=duration)

    return Dte,l,R0,d

def getBayes(test_path):

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
            Rt = Rt[tstart[0]:]

            M.append(mean)
            U.append(upper)
            L.append(lower)
            R.append(Rt)
            Ts.append((tstart[0],tstart[0] + len(mean)))

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

def getRegression(test_path,Inum):
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
            start,end = stable(I,Inum)
            if start != end:
                result = []
                Rt = Rt[start:end]
                mean = mean[start:end]
                skew = skew[start:end]

                rt_train = pd.Series(Rt)
                mean_train = pd.Series(mean)
                skew_train = pd.Series(skew)

                x_train = np.column_stack((mean_train,skew_train))
                x11_train = sm.add_constant(x_train)
                model = sm.OLS(rt_train,x11_train).fit()

                for j in range(start,end):
                    rt_p = model.params[0] + mean[j]*model.params[1] + skew[j]*model.params[2]
                    result.append(rt_p)
                Rpre.append(result)
                duration.append((start,end))
        
        liner = rF.readline()
        linei = iF.readline()
        linem = mF.readline()
        lines = sF.readline()
    rF.close()
    iF.close()
    mF.close()
    sF.close()
    return Rpre,duration
            






# def getTestData_NC(args,test_path,m,n,start,end):

#     test,test_l,_,method = read_data_NC(test_path,args.group,start,end,args.readDate_type)
    
#     l = []
#     if args.Dataset_type == 1:
#         for z in range(len(test_l)):
#             cnt = 0
#             for i in range(0,len(test_l[z]) - args.seq_len - args.output_size + 1, args.output_size):
#                 cnt += args.output_size
#             l.append(cnt)
            
#     elif args.Dataset_type == 3:
#         for z in range(len(test)):
#             cnt = 0
#             for i in range(0,len(test[z])):
#                 cnt += args.output_size
#             l.append(cnt)
#     else:
#         for z in range(len(test)):
#             cnt = 0
#             for i in range(0,len(test[z]) - args.seq_len - args.output_size + 1, args.output_size):
#                 cnt += args.output_size
#             l.append(cnt)

#     Dte,_ = createDataset(args,test, test_l, step_size=args.output_size, shuffle=False, m=m, n=n,duration=[])

#     return Dte,l,method



# def readNC(path):
#     regress = []
#     mean = []
#     low = []
#     upper = []
#     truth = []

#     rF = open(path + "\\test.txt")
#     mF = open(path + "\\mean.txt")
#     lF = open(path + "\\low.txt")
#     uF = open(path + "\\upper.txt")
#     tF = open(path + "\\truth.txt")
    
#     liner = rF.readline()
#     linem = mF.readline()
#     linel = lF.readline()
#     lineu = uF.readline()
#     linet = tF.readline()

#     while linem:
#         linem = linem.split(' \n')[0]
#         m_str = linem.split(' ')
#         linel = linel.split(' \n')[0]
#         l_str = linel.split(' ')
#         lineu = lineu.split(' \n')[0]
#         u_str = lineu.split(' ')
#         linet = linet.split(' \n')[0]
#         t_str = linet.split(' ')

#         mean = [float(m_str[i]) for i in range(len(m_str))]
#         low = [float(l_str[i]) for i in range(len(l_str))]
#         upper = [float(u_str[i]) for i in range(len(u_str))]
#         truth = [float(t_str[i]) for i in range(len(t_str))]

#         linem = mF.readline()
#         linel = lF.readline()
#         lineu = uF.readline()
#         linet = tF.readline()

#     while liner:
#         liner = liner.split(' \n')[0]
#         r_str = liner.split(' ')

#         R = [float(r_str[i]) for i in range(len(r_str))]
#         regress.append(R)
#         liner = rF.readline()

#     rF.close()
#     mF.close()
#     uF.close()
#     lF.close()

#     return truth,regress,mean,low,upper



# def read_data_NC(path,group,start,end,type):
    
#     data = []
#     label = []
#     label_all = []
#     Rt = []
#     method = []
#     count = 0
#     rF = open(path + "\\Rt.txt")
#     liner = rF.readline()

#     while liner:
#         liner = liner.split(' \n')[0]
#         r_str = liner.split(' ')
#         Rt = [float(r_str[i]) for i in range(len(r_str))]
#         liner = rF.readline()
#     rF.close()

#     dF = open(path + "\\distrb.txt")
#     lined = dF.readline()
#     while lined:
#         count += 1
#         lined = lined.split(' \n')[0]
#         d_str = lined.split(' ')
#         distrb = [float(d_str[i]) for i in range(len(d_str))]

#         if len(distrb) % group == 0 :
#             method.append(count)
#             t_data,t_label = process(Rt,distrb,start,end,group,type,label_all)
#             data.append(t_data)
#             label.append(t_label)
#         lined = dF.readline()
#     dF.close()

#     return data,label,label_all,method

        


