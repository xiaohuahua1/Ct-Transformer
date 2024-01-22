import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import math
import pandas as pd
import seaborn as sns
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

def mse(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    for i in range(num):
        result += math.pow((pred[i] - y[i]),2)
    result /= num
    return result

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
    
def getInfo_align(fold,d,R0,onset):
    path = fold + "d=" + str(d) + "R=" + str(R0) + "\\train"
    min_bound = 0
    max_bound = 0

    dailyF = open(path + "\\dailyInfection.txt")
    lineda = dailyF.readline()

    while lineda:
        lineda = lineda.split(' \n')[0]
        da_str = lineda.split(' ')

        if len(da_str) > 50:
            daily = [float(da_str[i]) for i in range(len(da_str))]
            length = len(daily)
            length_mid = daily.index(max(daily))

            if -length_mid < min_bound:
                min_bound = -length_mid
            if length - length_mid > max_bound:
                max_bound = length - length_mid
        lineda = dailyF.readline()
    dailyF.close()

    mean_dict = {}
    skew_dict = {}
    rt_dict = {}
    daily_dict = {}
    I_dict = {}
    ct_dict = []
    for i in range(group):
        ct_dict.append({})

    for i in range(min_bound,max_bound):
        mean_dict[i] = []
        skew_dict[i] = []
        rt_dict[i] = []
        daily_dict[i] = []
        I_dict[i] = []
        for j in range(group):
            ct_dict[j][i] = []

    rF = open(path + "\\Rt.txt")
    IF = open(path + "\\I.txt")
    mF = open(path + "\\ctMean.txt")
    dailyF = open(path + "\\dailyInfection.txt")
    sF = open(path + "\\ctSkew.txt")
    diF = open(path + "\\distrb.txt")

    if onset:
        diF = open(path + "\\distrbOnset.txt")
        mF = open(path + "\\ctMeanOnset.txt")
        sF = open(path + "\\ctSkewOnset.txt")

    liner = rF.readline()
    lineI = IF.readline()
    linem = mF.readline()
    lineda = dailyF.readline()
    lines = sF.readline()
    linedi = diF.readline()

    while liner:
        liner = liner.split(' \n')[0]
        r_str = liner.split(' ')
        lineI = lineI.split(' \n')[0]
        I_str = lineI.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        lineda = lineda.split(' \n')[0]
        da_str = lineda.split(' ')
        lines = lines.split(' \n')[0]
        s_str = lines.split(' ')
        linedi = linedi.split(' \n')[0]
        di_str = linedi.split(' ')

        if len(r_str) > 50:
            rt = [float(r_str[i]) for i in range(len(r_str))]
            I = [float(I_str[i]) for i in range(len(I_str))]
            mean = [float(m_str[i]) for i in range(len(m_str))]
            skew= [float(s_str[i]) for i in range(len(s_str))]
            daily = [float(da_str[i]) for i in range(len(da_str))]
            distrb = [float(di_str[i]) for i in range(len(di_str))]
            for i in range(len(rt)):
                if rt[i] <= 0.05:
                    rt[i] = 0.05
            length = len(daily)
            length_mid = daily.index(max(daily))
            for i in range(-length_mid,length - length_mid):
                mean_dict[i].append(mean[i + length_mid])
                skew_dict[i].append(skew[i + length_mid])
                rt_dict[i].append(rt[i + length_mid])
                I_dict[i].append(I[i + length_mid])
                elem = process_distrb(distrb,group,i + length_mid)
                for j in range(group):
                    ct_dict[j][i].append(elem[j])

        liner = rF.readline()
        lineI = IF.readline()
        linem = mF.readline()
        lineda = dailyF.readline() 
        lines = sF.readline()
        linedi = diF.readline()

    rF.close()
    IF.close()
    mF.close()
    dailyF.close()  
    sF.close()
    diF.close()

    rt_mean = []
    mean_mean = []
    skew_mean = []
    I_mean = []
    ct_mean = []

    for i in range(group):
        ct_mean.append([])

    for i in range(min_bound,max_bound):
        median_rt,lower,upper = calculate(rt_dict[i])
        rt_mean.append(median_rt)

        median_mean,lower,upper = calculate(mean_dict[i])
        mean_mean.append(median_mean)

        median_skew,lower,upper = calculate(skew_dict[i])
        skew_mean.append(median_skew)

        median_I,lower,upper = calculate(I_dict[i])
        I_mean.append(median_I)

        for j in range(group):
            median_ct,lower,upper = calculate(ct_dict[j][i])
            ct_mean[j].append(median_ct)

    start,end = stable(I_mean,Inum) 
    rt_mean = rt_mean[start:end]
    mean_mean = mean_mean[start:end]
    skew_mean = skew_mean[start:end]

    for i in range(group):
        ct_mean[i] = ct_mean[i][start:end]

    start += min_bound
    end += min_bound

    return start,end,rt_mean,mean_mean,skew_mean,ct_mean



def getInfo(fold,d,R0,onset):
    path = fold + "d=" + str(d) + "R=" + str(R0) + "\\train"
    max_len = 0

    rF = open(path + "\\Rt.txt")

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

    rF = open(path + "\\Rt.txt")
    IF = open(path + "\\I.txt")
    diF = open(path + "\\distrb.txt")
    mF = open(path + "\\ctMean.txt")
    sF = open(path + "\\ctSkew.txt")

    if onset:
        diF = open(path + "\\distrbOnset.txt")
        mF = open(path + "\\ctMeanOnset.txt")
        sF = open(path + "\\ctSkewOnset.txt")

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

        # if len(r_str) > 50:
        rt = [float(r_str[i]) for i in range(len(r_str))]
        I = [float(I_str[i]) for i in range(len(I_str))]
        distrb = [float(di_str[i]) for i in range(len(di_str))]
        mean = [float(m_str[i]) for i in range(len(m_str))]
        skew= [float(s_str[i]) for i in range(len(s_str))]
                # for i in range(len(rt)):
                #     if rt[i] <= 0.05:
                #         rt[i] = 0.05
        for i in range(len(mean)):
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
    mean_mean = mean_mean[start:end]
    skew_mean = skew_mean[start:end]

    for i in range(group):
        ct_mean[i] = ct_mean[i][start:end]
        
    return start,end,rt_mean,mean_mean,skew_mean,ct_mean

if __name__ == '__main__':
    
    figure = "VSN weight"

    # fig, ax = plt.subplots(2,6,figsize=(25, 10))
    if figure == "more CtRt":
        
        Inum = 100
        group = 8
        show_type = "noalign"
        isLog = False
        isMinMax = False
        show_feature = "Rt"
        onset = True
        label = "Rt Value"

        fold = "..\\results\\ER\\"
        net = ["ER"]
        d = [10]
        # R0 = [1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6]
        # R0 = [1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6]
        R0 = [1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5]

        color = ["rosybrown","lightcoral","maroon","tomato","chocolate","sandybrown","orange","tan","goldenrod","gold","olive","y","greenyellow","green","lightseagreen","teal","c","deepskyblue","royalblue","blue","blueviolet","indigo","m","deeppink","crimson"]
        count = 0
        count_c = 0

        fig = plt.figure(figsize=(12, 6))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        skew_all = []
        skew_list = []
        mean_all = []
        mean_list = []
        rt_all = []
        rt_list = []
        start_list = []
        end_list = []

        ct_all = []
        ct_list = []
        for i in range(group):
            ct_all.append([])
            ct_list.append([])

        if show_feature == "ct":
            fig, ax = plt.subplots(2,3,figsize=(25, 10))
        
        for i in range(len(d)):
            for j in range(len(R0)):
                if show_type == "align":
                    start,end,rt_mean,mean_mean,skew_mean,ct_mean = getInfo_align(fold,d[i],R0[j],onset)
                else:
                    start,end,rt_mean,mean_mean,skew_mean,ct_mean = getInfo(fold,d[i],R0[j],onset)
                if isLog:
                    rt_mean  = np.log(rt_mean)
                if isMinMax == False:
                    if show_type == "align":
                        x = [i for i in range(start,end)]
                    else:
                        x = [i for i in range(end - start)]
                    if show_feature == "mean":
                        mean_mean = getSmooth(mean_mean,7)
                        plt.plot(x,mean_mean,color=color[count],label=str(R0[j]),linewidth=2)  
                    elif show_feature == "skew":
                        plt.plot(x,skew_mean,color=color[count],label=str(R0[j]),linewidth=2)  
                    elif show_feature == "Rt":
                        plt.plot(x,rt_mean,color=color[count],label=str(R0[j]),linewidth=2)  
                    elif show_feature == "ct":
                        ax[0,0].plot(x,ct_mean[0],color=color[count])
                        ax[0,0].set_title('[16-20]')
                        
                        ax[0,1].plot(x,ct_mean[1],color=color[count])
                        ax[0,1].set_title('[21-24]')

                        ax[0,2].plot(x,ct_mean[2],color=color[count])
                        ax[0,2].set_title('[25-28]')

                        ax[1,0].plot(x,ct_mean[3],color=color[count])
                        ax[1,0].set_title('[29-32]')
                        
                        ax[1,1].plot(x,ct_mean[4],color=color[count])
                        ax[1,1].set_title('[33-36]')

                        ax[1,2].plot(x,ct_mean[5],color=color[count])
                        ax[1,2].set_title('[37-40]')

                        fig.text(0.15 + 0.03*count, 0.95, str(R0[j]), ha='center', va='top', color=color[count], fontsize=16)
                    count += 2
                mean_all.extend(mean_mean)
                mean_list.append(mean_mean)

                skew_all.extend(skew_mean)
                skew_list.append(skew_mean)

                rt_all.extend(rt_mean)
                rt_list.append(rt_mean)

                start_list.append(start)
                end_list.append(end)

                for z in range(group):
                    ct_all[z].extend(ct_mean[z])
                    ct_list[z].append(ct_mean[z])


        if isMinMax:
            count = 0
            count_c = 0

            ct_min,ct_max = [],[]
            rt_min,rt_max = 0,0
            mean_min,mean_max = 0,0
            skew_min,skew_max = 0,0

            for i in range(group):
                c_min,c_max = np.min(ct_all[i]),np.max(ct_all[i])
                ct_min.append(c_min)
                ct_max.append(c_max)

            mean_min,mean_max = np.min(mean_all),np.max(mean_all)
            rt_min,rt_max = np.min(rt_all),np.max(rt_all)
            skew_min,skew_max = np.min(skew_all),np.max(skew_all)

            for i in range(len(d)):
                for j in range(len(R0)):
                    if show_type == "align":
                        x = [z for z in range(start_list[count],end_list[count])]
                    else:
                        x = [z for z in range(end_list[count]-start_list[count])]
                    mean_list[count] = (mean_list[count] - mean_min) / (mean_max - mean_min)
                    skew_list[count] = (skew_list[count] - skew_min) / (skew_max - skew_min)
                    rt_list[count] = (rt_list[count] - rt_min) / (rt_max - rt_min)

                    for z in range(group):
                        ct_list[z][count] = (ct_list[z][count] - ct_min[z]) / (ct_max[z] - ct_min[z])
                    if show_feature == "mean":
                        plt.plot(x,mean_list[count],color=color[count_c],label=str(R0[count]))
                    elif show_feature == "skew":
                        plt.plot(x,skew_list[count],color=color[count_c],label=str(R0[count]))
                    elif show_feature == "rt":
                        plt.plot(x,rt_list[count],color=color[count_c],label=str(R0[count]))
                    elif show_feature == "ct":
                        ax[0,0].plot(x,ct_list[0][count],color=color[count_c])
                        ax[0,0].set_title('[16-20]')
                        
                        ax[0,1].plot(x,ct_list[1][count],color=color[count_c])
                        ax[0,1].set_title('[21-24]')

                        ax[0,2].plot(x,ct_list[2][count],color=color[count_c])
                        ax[0,2].set_title('[25-28]')

                        ax[1,0].plot(x,ct_list[3][count],color=color[count_c])
                        ax[1,0].set_title('[29-32]')
                        
                        ax[1,1].plot(x,ct_list[4][count],color=color[count_c])
                        ax[1,1].set_title('[33-36]')

                        ax[1,2].plot(x,ct_list[5][count],color=color[count_c])
                        ax[1,2].set_title('[37-40]')

                        fig.text(0.3 + 0.03*count, 0.95, str(R0[j]), ha='center', va='top', color=color[count_c], fontsize=16)
                    count += 1
                    count_c += 2

        plt.xlabel("Days since start of outbreak",fontsize=25,fontweight='bold')
        # plt.ylabel(show_feature)
        plt.ylabel(label,fontsize=25,fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()

    elif figure == "more CtRt test":
        fold = "..\\results\\ER\\"
        net = ["ER"]
        d = [10]
        R = [1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5]
        onset = True
        group = 4
        Inum = 100
        write = True
        value_type = "Ct"

        for i in range(len(R)):
            fig = plt.figure(figsize=(12, 6))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rcParams['grid.linewidth'] = 2.0
            plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
            plt.rcParams['ytick.labelsize'] = 20  
            plt.rcParams['font.family'] = 'Times New Roman'

            start,end,rt_mean,mean_mean,skew_mean,ct_mean = getInfo(fold,d[0],R[i],onset)
            
            if value_type == "Ct":
                label = "Mean Ct Value"
                value = mean_mean
                value = getSmooth(value,7)
            elif value_type == "Rt":
                label = "Rt Value"
                value = rt_mean
                value = getSmooth(value,7)
            # end -= 30
            # value = value[:-30]
            x = [i for i in range(end - start)]
            plt.plot(x,value,color="rosybrown",label=str(R[i]),linewidth=2) 
            plt.xlabel("Days since start of outbreak",fontsize=25,fontweight='bold')
            # plt.ylabel(show_feature)
            plt.ylabel(label,fontsize=25,fontweight='bold')
            plt.legend(fontsize=12)
            plt.grid()
            plt.show() 

            if write == True:
                if value_type == "Ct" and onset == False:
                    f = open(fold + "d=" + str(d[0]) + "R=" + str(R[i]) + "\\" + value_type + "1_mean.txt",'w')
                else:
                    f = open(fold + "d=" + str(d[0]) + "R=" + str(R[i]) + "\\" + value_type + "_mean.txt",'w')
                for i in range(len(value)):
                    f.write(str(value[i]) + " ")
                f.write("\n")

    elif figure == "CtRt relation":
        fold = "..\\results\\ER\\"
        net = ["ER"]
        d = [10]
        R = [1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5]
        length = [247,182,153,131,122,109,101,92,89,80,75,74]
        onset = True
        value_type = "Rt"
        color = ["rosybrown","lightcoral","maroon","tomato","chocolate","sandybrown","orange","tan","goldenrod","gold","olive","y","greenyellow","green","lightseagreen","teal","c","deepskyblue","royalblue","blue","blueviolet","indigo","m","deeppink","crimson"]
        count = 0

        fig = plt.figure(figsize=(12, 7))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        for i in range(len(d)):
            for j in range(len(R)):
                if value_type == "Ct" and onset == False:
                    f = open(fold + "d=" + str(d[i]) + "R=" + str(R[j]) + "\\" + value_type + "1_mean.txt")
                else:
                    f = open(fold + "d=" + str(d[i]) + "R=" + str(R[j]) + "\\" + value_type + "_mean.txt")
                line = f.readline()

                while line:
                    line = line.split(' \n')[0]
                    l_str = line.split(' ')
                    value = [float(l_str[i]) for i in range(len(l_str))]
                    # for z in range(len(value)):
                    #     value[z] = 49.8 - value[z]
                    x = [z for z in range(length[j])]
                    value = value[0:length[j]]
                    plt.plot(x,value,color=color[count],label=str(R[j]),linewidth=2.5) 
                    # plt.xlim(0,300)
                    if value_type == "Rt":
                        plt.axis([0, 300, 0, 4.0])
                        plt.ylim(0,4.0)
                    if value_type == "Ct":
                        plt.axis([0, 300, 21.20, 23.00])
                        # plt.ylim(21.20,23.00)
                    line = f.readline()
                f.close()
                count += 2

        if value_type == "Ct":
            label = "Mean Ct Value"
            
        elif value_type == "Rt":
            label = "Rt Value"

        plt.xlabel("Days since start of outbreak",fontsize=28,fontweight='bold')
        # plt.ylabel(show_feature)
        plt.ylabel(label,fontsize=28,fontweight='bold')
        plt.legend(fontsize=18)
        # plt.grid()
        plt.show() 




    elif figure == "more Ct trajectory":
        num = 100
        random_num = random.random()
        num_id = int(num*random_num)
        count = 0

        fig = plt.figure(figsize=(14, 7))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        ctF = open("..\\results\\trajectory.txt")
        linec = ctF.readline()
        
        x_black = []
        ct_black = []

        while linec:
            linec = linec.split(' \n')[0]
            c_str = linec.split(' ')

            ct = [float(c_str[i]) for i in range(len(c_str))]
            x = [i for i in range(len(c_str))]
            if count == num_id:
                x_black = x
                ct_black = ct
            else:
                if count % 3 == 0:
                    plt.plot(x,ct,color="lightgray")
            linec = ctF.readline()
            count += 1
        ctF.close()
        plt.plot(x_black,ct_black,color="black")
        plt.scatter(x_black,ct_black,color="black",marker="o",label='Viral load')
        plt.ylim(15.00,45.0)
        plt.xlim(0,30)
        plt.xlabel("Days since Infection",fontsize=28,fontweight='bold')
        plt.ylabel("Ct value",fontsize=28,fontweight='bold')
        # plt.legend(fontsize=18)
        plt.show()

    elif figure == "one Ct trajectory":
        num = 100
        random_num = random.random()
        num_id = int(num*random_num)
        count = 0

        fig = plt.figure(figsize=(10, 6))
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 'large'
        plt.rcParams['ytick.labelsize'] = 'large'
        plt.rcParams['font.weight'] = 'bold'
        ctF = open("..\\results\\trajectory.txt")
        linec = ctF.readline()
        
        x_black = []
        ct_black = []

        while linec:
            linec = linec.split(' \n')[0]
            c_str = linec.split(' ')

            ct = [float(c_str[i]) for i in range(len(c_str))]
            x = [i for i in range(len(c_str))]
            if count == num_id:
                x_black = x
                ct_black = ct
            linec = ctF.readline()
            count += 1
        ctF.close()
        plt.plot(x_black,ct_black,color="darkblue")
        plt.scatter(x_black,ct_black,color="darkblue",marker="o")
        # plt.xlabel("time since infection",fontsize=24,fontname='Times New Roman',weight = 'bold')
        # plt.ylabel("Ct Value",fontsize=24,fontname='Times New Roman',weight = 'bold')
        plt.grid()
        plt.show()


    elif figure == "Ct ln(Rt)":

        Inum = 100
        group = 8
        onset = True
        fold = "..\\results\\ER\\"
        net = ["ER"]
        d = [10]
        # R0 = [1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.1,3.2,3.3,3.4,3.5,3.6]
        # R0 = [1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6]
        R0 = [1.3,1.5,1.7,1.9,2.1,2.3,2.5,2.7,2.9,3.1,3.3,3.5]
        Ct = [28.136,28.362,28.690,28.970,29.230,29.335,29.547,29.780,30.069,30.197,30.397,30.587]
        Ct_onset = [21.758,21.674,21.616,21.553,21.486,21.423,21.377,21.346,21.324,21.300,21.282,21.268]

        if onset:
            value = Ct_onset
        else:
            value = Ct

        Rt0 = []
        Rt0_log = []

        for i in range(len(R0)):
            Rt0.append(1.2**R0[i])
            Rt0_log.append(np.log(R0[i]))

        r_max,r_min = np.max(Rt0),np.min(Rt0)
        l_max,l_min = np.max(Rt0_log),np.min(Rt0_log)

        Rt0 = (Rt0 - r_min) / (r_max - r_min)
        Rt0_log = (Rt0_log - l_min) / (l_max - l_min)

        fig = plt.figure(figsize=(12, 7))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.plot(value,Rt0,color="sandybrown",label="Rt",linewidth=2.5)
        plt.scatter(value,Rt0,color="sandybrown",marker='*',s=200)
        plt.plot(value,Rt0_log,color="olive",label="ln(Rt)",linewidth=2.5)
        plt.scatter(value,Rt0_log,color="olive",marker='*',s=200)

        for i in range(len(R0)):
            plt.annotate(R0[i], (value[i], Rt0_log[i]),fontsize=20) 
        
        plt.ylim(0,1)
        # plt.xlim(21.25,21.8)
        plt.xlabel("the mean Ct value at the begining of the outbreak",fontsize=28,fontweight='bold')
        plt.ylabel("Normalized Rt or ln(Rt) value",fontsize=28,fontweight='bold')
        # plt.xticks([21.2,21.3,21.4,21.5,21.6,21.7,21.8,21.9])
        
        # plt.grid()
        plt.legend(fontsize=20)
        plt.show()

        

    elif figure == "one CtRt":
        Inum = 100
        group = 8

        fold = "..\\results\\ER\\d=10R=3\\val"
        rF = open(fold + "\\Rt.txt")
        IF = open(fold + "\\I.txt")
        diF = open(fold + "\\distrb.txt")

        liner = rF.readline()
        lineI = IF.readline()
        linedi = diF.readline()

        while liner:
            liner = liner.split(' \n')[0]
            r_str = liner.split(' ')
            lineI = lineI.split(' \n')[0]
            I_str = lineI.split(' ')
            linedi = linedi.split(' \n')[0]
            di_str = linedi.split(' ')

            if len(r_str) > 50:
                ct_all = []
                rt = [float(r_str[i]) for i in range(len(r_str))]
                I = [float(I_str[i]) for i in range(len(I_str))]
                distrb = [float(di_str[i]) for i in range(len(di_str))]
                for i in range(len(rt)):
                    if rt[i] <= 0.05:
                        rt[i] = 0.05
                for i in range(group):
                    ct_all.append([])

                start,end = stable(I,Inum)
                if start != end:
                    rt = rt[start:end]
                for i in range(start,end):
                    elem = process_distrb(distrb,group,i)
                    for j in range(group):
                        ct_all[j].append(elem[j])
                for i in range(group):
                    ct_all[i] = getSmooth(ct_all[i],4)

                plt.rcParams['font.weight'] = 'bold'
                plt.rcParams['axes.linewidth'] = 2.0
                plt.rcParams['grid.linewidth'] = 2.0
                plt.rcParams['xtick.labelsize'] = 25  # 设置x轴刻度标签的字体大小为12
                plt.rcParams['ytick.labelsize'] = 25  
                plt.rcParams['font.family'] = 'Times New Roman'
                fig = plt.figure(figsize=(12, 6))
                x = [i for i in range(start,end)]
                plt.plot(x,ct_all[0],'coral',linewidth=4,label='Ct1')
                # plt.plot(x,ct_all[1],'goldenrod',linewidth=2.5)
                plt.plot(x,ct_all[2],'yellowgreen',linewidth=4,label='Ct2')
                # plt.plot(x,ct_all[3],'turquoise',linewidth=2.5)
                plt.plot(x,ct_all[4],'cornflowerblue',linewidth=4,label='Ct3')
                # plt.plot(x,ct_all[5],'darkviolet',linewidth=2.5)
                plt.plot(x,ct_all[6],'magenta',linewidth=4,label='Ct4')
                # plt.plot(x,ct_all[7],'lightpink',linewidth=2.5)
                plt.grid()
                legend = plt.legend(fontsize=21,loc='upper left')
                for label in legend.get_texts():
                    label.set_rotation('vertical')
                # 设置x轴和y轴刻度标签，并将它们竖着放置
                plt.xticks(rotation='vertical')
                plt.yticks(rotation='vertical')
                plt.show()

                fig = plt.figure(figsize=(12, 6))
                x = [i for i in range(start,end)]
                plt.plot(x,rt,'r',linewidth=4,label='Rt')
                plt.grid()
                legend = plt.legend(fontsize=21,loc='upper right')
                for label in legend.get_texts():
                    label.set_rotation('vertical')
                # 设置x轴和y轴刻度标签，并将它们竖着放置
                plt.xticks(rotation='vertical')
                plt.yticks(rotation='vertical')
                plt.show()

                break
            liner = rF.readline()
            lineI = IF.readline()
            linedi = diF.readline()

            rF.close()
            IF.close()
            diF.close()

    elif figure == "Rt dailyInfection":
        Inum = 100
        fold = "..\\results\\ER\\d=10R=3\\val"
        rF = open(fold + "\\Rt.txt")
        IF = open(fold + "\\I.txt")
        dF = open(fold + "\\dailyInfection.txt")

        liner = rF.readline()
        lineI = IF.readline()
        lined = dF.readline()

        while liner:
            liner = liner.split(' \n')[0]
            r_str = liner.split(' ')
            lineI = lineI.split(' \n')[0]
            I_str = lineI.split(' ')
            lined = lined.split(' \n')[0]
            d_str = lined.split(' ')

            if len(r_str) > 50:
                rate = []
                rt = [float(r_str[i]) for i in range(len(r_str))]
                I = [float(I_str[i]) for i in range(len(I_str))]
                daily = [float(d_str[i]) for i in range(len(d_str))]
                for i in range(len(rt)):
                    if rt[i] <= 0.05:
                        rt[i] = 0.05
                start,end = stable(I,Inum)
                print(start)
                print(end)
                if start != end:
                    rt = rt[start:end]
                    I = I[start:end]
                    daily = daily[start:end]
                for i in range(len(rt)):
                    rate.append(daily[i]/I[i])

                plt.rcParams['font.weight'] = 'bold'
                plt.rcParams['axes.linewidth'] = 2.0
                plt.rcParams['grid.linewidth'] = 2.0
                plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
                plt.rcParams['ytick.labelsize'] = 20  
                plt.rcParams['font.family'] = 'Times New Roman'
                fig, ax1 = plt.subplots(figsize=(12, 7))
                # x = [i for i in range(start,end)]
                x = [i for i in range(0,end-start)]
                # plt.bar(x,I)
                # plt.bar(x,daily)
                ax1.bar(x, I,color='darkgrey',label='Infection')
                ax1.bar(x,daily,color='dimgrey',label='New Infection')
                ax1.set_xlabel('Days since start of outbreak',fontsize=28,fontweight='bold')
                ax1.set_ylabel('Infection',fontsize=28,fontweight='bold')
                ax1.tick_params('y')

                ax2 = ax1.twinx()
                ax2.plot(x, rate, 'darkblue',linewidth=2.5,label='New Infection / Infection')
                ax2.set_xlim(0,75)
                ax2.set_ylim(0,0.30)
                ax2.set_ylabel('Proportion of new infection',fontsize=28,fontweight='bold')
                ax2.tick_params('y')
                
                ax1.legend(fontsize=16,bbox_to_anchor=(0.64, 0.9))
                ax2.legend(fontsize=16)
                plt.show()

                break
        liner = rF.readline()
        lineI = IF.readline()
        linedi = dF.readline()

        rF.close()
        IF.close()
        dF.close()

    elif figure == "conparison of C Rt and Epi Rt":
        Inum = 100
        fold = "..\\results\\ER\\changeRviro"
        rF = open(fold + "\\Rt.txt")
        cF = open(fold + "\\baseMean.txt")
        eF = open(fold + "\\EpiRt.txt")
        tF = open(fold + "\\Tstart.txt")
        iF = open(fold + "\\I.txt")

        liner = rF.readline()
        linec = cF.readline()
        linee = eF.readline()
        linet = tF.readline()
        linei = iF.readline()

        while liner:
            linet = linet.split(' \n')[0]
            t_str = linet.split(' ') 
            liner = liner.split(' \n')[0]
            r_str = liner.split(' ') 
            linee = linee.split(' \n')[0]
            e_str = linee.split(' ') 
            linec = linec.split(' \n')[0]
            c_str = linec.split(' ') 
            linei = linei.split(' \n')[0]
            i_str = linei.split(' ') 

            tstart = [int(t_str[i]) for i in range(len(t_str))]
            Rt = [float(r_str[i]) for i in range(len(r_str))]
            eRt = [float(e_str[i]) for i in range(len(e_str))]
            cRt = [float(c_str[i]) for i in range(len(c_str))]
            I = [float(i_str[i]) for i in range(len(i_str))] 

            start,end = stable(I,Inum)
            if start != end:
                start_s = start - tstart[0]
                end_s = end - tstart[0]

                Rt = Rt[start:end]
                eRt = eRt[start:end]
                cRt = cRt[start_s:end_s]

                e_mse = mse(Rt,eRt)
                c_mse = mse(Rt,cRt)

                fig = plt.figure(figsize=(12, 6))
                x = [i for i in range(start,end)]
                plt.plot(x,Rt,color='black',label='truth')
                plt.plot(x,cRt,color='brown',label='C Rt=' + "{:.4f}".format(c_mse))
                plt.plot(x,eRt,color='c',label='Epi Rt=' + "{:.4f}".format(e_mse))
                plt.xlabel("Days since start of outbreak",fontsize=25,fontweight='bold')
                plt.ylabel("Rt Value",fontsize=25,fontweight='bold')
                plt.grid()
                plt.legend()
                plt.show()
        
            linet = tF.readline()
            linei = iF.readline()
            liner = rF.readline()
            linec = cF.readline()
            linee = eF.readline()
        tF.close()
        iF.close()
        rF.close()
        cF.close()
        eF.close()

    elif figure == "Length of Patch":
        net = "The Average MAE Error on Synthetic SF Dataset"
        P = [2,4,6,8,10,12,14,16,18]
        s = [100,100,100,100,100,100,100,100,100]
        if net == "The Average MAE Error on Synthetic ER Dataset":
            # MAE = [0.074,0.070,0.071,0.074,0.075,0.074,0.087,0.078,0.088]
            MAE = [0.063,0.060,0.061,0.063,0.064,0.063,0.075,0.067,0.075]
        if net == "The Average MAE Error on Synthetic SF Dataset":
            # MAE = [0.124,0.114,0.117,0.108,0.126,0.119,0.129,0.141,0.125]
            MAE = [0.099,0.091,0.093,0.086,0.101,0.095,0.103,0.113,0.100]
        
        fig = plt.figure(figsize=(14, 7))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.plot(P,MAE,color='steelblue',linewidth=3)
        plt.scatter(P,MAE,color='steelblue',s=s)
        plt.gca().yaxis.set_major_formatter('{:.3f}'.format)
        if net == "The Average MAE Error on Synthetic SF Dataset":
            plt.ylim(0.085,0.115)
        if net == "The Average MAE Error on Synthetic ER Dataset":
            plt.ylim(0.058,0.076)
        plt.xlim(2,18)
        plt.xlabel("P",fontsize=28,fontweight='bold')
        plt.ylabel("MAE",fontsize=28,fontweight='bold')
        plt.title(net,fontsize=28,fontweight='bold')
        plt.show()

    elif figure == "VSN weight":
        vsn_type = "more Rt"
        patch_len = 1
        input_size = 9

        def getWeight(patch,input,path,num):

            weight_all = []
            for i in range(input):
                weight_all.append([])

            wF = open(path)
            linew = wF.readline()

            while linew:
                
                linew = linew.split('\n')[0]
                w_str = linew.split(' ') 

                weight = [float(w_str[i]) for i in range(len(w_str))]
                
                for i in range(len(weight)):
                    id = int(i / patch)
                    weight_all[id].append(weight[i])


                linew = wF.readline()
            wF.close()
            result = getSmooth(weight_all[num],7)
            return result

        if vsn_type == "one Rt":
            path = "..\\results\\ER\\d=10R=2.2\\test\\draw\\ablation\\flag_patch.txt"
            weight1 = getWeight(patch_len,input_size,path,0)
            weight2 = getWeight(patch_len,input_size,path,2)
            weight3 = getWeight(patch_len,input_size,path,4)
            weight4 = getWeight(patch_len,input_size,path,6)
            weight5 = getWeight(patch_len,input_size,path,8)
            fig = plt.figure(figsize=(14, 7))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rcParams['grid.linewidth'] = 2.0
            plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
            plt.rcParams['ytick.labelsize'] = 20  
            plt.rcParams['font.family'] = 'Times New Roman'

            x = [i+20 for i in range(len(weight1))]
            plt.plot(x,weight1,color='indianred',linewidth=3,label="Ct=[16,23)",marker="s",markevery=12,linestyle="-.")
            plt.plot(x,weight2,color='goldenrod',linewidth=3,label="Ct=[23,29)",marker="s",markevery=12,linestyle="-.")
            plt.plot(x,weight3,color='olivedrab',linewidth=3,label="Ct=[29,35)",marker="s",markevery=12,linestyle="-.")
            plt.plot(x,weight4,color='teal',linewidth=3,label="Ct=[35,40]",marker="s",markevery=12,linestyle="-.")
            plt.plot(x,weight5,color='slateblue',linewidth=3,label="Mean",marker="s",markevery=12,linestyle="-.")
            # plt.plot(x,weight_all[8],color='darkcyan',linewidth=3)
            plt.ylim(0,0.1)
            plt.xlim(20,160)
            plt.title("Importance of Each Ct feature ",fontsize=28,fontweight='bold')
            plt.xlabel("Days since start of outbreak",fontsize=28,fontweight='bold')
            plt.ylabel("weight",fontsize=28,fontweight='bold')
            
            plt.legend(fontsize=18)
            plt.show()
        elif vsn_type == "more Rt":
            path1 = "..\\results\\SF\\d=10R=1.8\\test\\draw\\ablation\\flag_patch.txt"
            path2 = "..\\results\\ER\\d=10R=1.8\\test\\draw\\ablation\\flag_patch.txt"
            path4 = "..\\results\\ER\\d=10R=2.2\\test\\draw\\ablation\\flag_patch.txt"
            path3 = "..\\results\\SF\\d=10R=2.4\\test\\draw\\ablation\\flag_patch.txt"

            weight1 = getWeight(patch_len,input_size,path1,8)
            weight2 = getWeight(patch_len,input_size,path2,8)
            weight3 = getWeight(patch_len,input_size,path3,8)
            weight4 = getWeight(patch_len,input_size,path4,8)

            fig = plt.figure(figsize=(14, 7))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rcParams['grid.linewidth'] = 2.0
            plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
            plt.rcParams['ytick.labelsize'] = 20  
            plt.rcParams['font.family'] = 'Times New Roman'

            x1 = [i+20 for i in range(len(weight1))]
            x2 = [i+20 for i in range(len(weight2))]
            x3 = [i+20 for i in range(len(weight3))]
            x4 = [i+20 for i in range(len(weight4))]
            plt.plot(x1,weight1,color='indianred',linewidth=3,label="net=SF,R0=1.8",marker="s",markevery=8,linestyle="-.")
            plt.plot(x2,weight2,color='goldenrod',linewidth=3,label="net=ER,R0=1.8",marker="s",markevery=8,linestyle="-.")
            plt.plot(x3,weight3,color='olivedrab',linewidth=3,label="net=SF,R0=2.4",marker="s",markevery=8,linestyle="-.")
            plt.plot(x4,weight4,color='teal',linewidth=3,label="net=ER,R0=2.4",marker="s",markevery=8,linestyle="-.")
            plt.ylim(0.02,0.08)
            plt.xlim(20,160)
            # plt.plot(x,weight5,color='slateblue',linewidth=3,label="ct mean")
            # plt.plot(x,weight_all[8],color='darkcyan',linewidth=3)
            plt.title("Importance of Each Ct feature ",fontsize=28,fontweight='bold')
            plt.xlabel("Days since start of outbreak",fontsize=28,fontweight='bold')
            plt.ylabel("weight",fontsize=28,fontweight='bold')
            
            plt.legend(fontsize=18)
            plt.show()
        

    elif figure == "QK weight":
        head = 8
        path_ER = "..\\results\\ER\\d=10R=2.4\\test\\draw\\ablation\\attn_patch.txt"
        # path_SF = "..\\results\\SF\\d=10R=2.4\\test\\draw\\ablation\\attn_all.txt"

        def get_attn(path):

            attn_all = []
            attn = []

            aF = open(path)
            linea = aF.readline()

            while linea:
                linea = linea.split('\n')[0]
                a_str = linea.split(' ') 
                
                if len(a_str) != 1:
                    att_line = [float(a_str[i]) for i in range(len(a_str))]
                    attn.append(att_line)
                else:
                    attn_all.append(attn)
                    attn = []

                linea = aF.readline()
            aF.close()
            
            id = 0
            weight = attn_all[id]
            
            # for i in range(1,head):
            #     weights = attn_all[i]
            #     for j in range(len(weights)):
            #         for z in range(len(weights[0])):
            #             weight[j][z] = weight[j][z] + weights[j][z]
            # for i in range(len(weight)):
            #     for j in range(len(weight[0])):
            #         weight[i][j] = weight[i][j] / head
            weight_matrix = np.array(weight)
            return weight_matrix

        weight_matrix_ER = get_attn(path_ER)
        # weight_matrix_SF = get_attn(path_SF)

        # 创建一个包含两个子图的Figure对象，并指定子图的大小
        fig = plt.figure(figsize=(8, 8))
        # fig = plt.figure(figsize=(8, 8))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'
        
        plt.imshow(weight_matrix_ER, cmap='magma', interpolation='nearest')
        plt.colorbar()
        # plt.title("Attention Map(Patch)",fontsize=25,fontweight='bold')
        
        plt.show()
    elif figure == "ablation":
        loss_type = "r2"
        ablation_type = ["No TP","No patch","No ISA","No GRN","No CFS"]


        ER_mae = [0.636098981,0.197962154,0.027656477,0.048034934,0.034934498]
        ER_rmse = [0.517826825,0.158743633,0.009337861,0.060271647,0.012733447]
        ER_r2 = [-0.029523408,-0.023724167,0.003479544,-0.002530578,0.002003374]

        SF_mae = [0.458178439,0.119888476,0.30204461,0.207249071,0.002788104]
        SF_rmse = [0.478790614,0.0816787,0.279783394,0.181859206,0.031588448]
        SF_r2 = [-0.030912863,-0.00560166,-0.020539419,-0.016390041,-0.002593361]

        ave_mae = [0.54713871,0.158925315,0.164850544,0.127642003,0.018861301]
        # ave_mae = [54.713871,15.8925315,16.4850544,12.7642003,1.8861301]
        ave_rmse = [0.498308719,0.120211167,0.144560627,0.121065426,0.022160947]
        ave_r2 = [-0.030218135,-0.014662913,-0.008529937,-0.00946031,-0.000294993]

        fig = plt.figure(figsize=(12, 8))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        def percent_formatter(x, pos):
            return '{:.1f}'.format(x*100)
        
        # def percent_formatter(x, pos):
        #     return '{:.1}'.format(x)
        formatter = FuncFormatter(percent_formatter)

        bar_width = 0.35
        x = np.arange(len(ablation_type))

        if loss_type == "MAE":
            # plt.bar(x,ER_mae,width=bar_width,color='forestgreen',label='ER Data')
            # plt.bar(x+bar_width+0.05,SF_mae,width=bar_width,color='c',label='SF Data')
            plt.bar(x,ER_mae,width=bar_width,color="#608595",label='ER Data')
            plt.bar(x+bar_width+0.05,SF_mae,width=bar_width,color="#DFC286",label='SF Data')

            
            for i in range(len(ave_mae)):
                ave_x = [i,i+0.4]
                ave_y = [ave_mae[i],ave_mae[i]]
                if i == 0:
                    plt.plot(ave_x,ave_y,linestyle='--',color='r',linewidth=2,label='Average')
                else:
                    plt.plot(ave_x,ave_y,linestyle='--',color='r',linewidth=2)
                # plt.text(i + 0.2,ave_mae[i],'{:.1%}'.format(ave_mae[i]),ha='center', va='bottom',fontsize=20)
                plt.text(i + 0.2,ave_mae[i],'{:.1f}'.format(ave_mae[i]*100),ha='center', va='bottom',fontsize=25)

            plt.gca().yaxis.set_major_formatter(formatter)
            
            plt.xticks(x + bar_width / 2, ablation_type)
            plt.ylim(0,0.7)
            plt.xlabel("Type of Ablation",fontsize=28,fontweight='bold')
            plt.ylabel("Increase in MAE loss (%)",fontsize=28,fontweight='bold')
            plt.legend(fontsize=20)
            # plt.title("% Increase in MAE loss by Dataset in different network type",fontsize=20,fontweight='bold')
            plt.show()
        elif loss_type == "RMSE":
            plt.bar(x,ER_rmse,width=bar_width,color="#608595",label='ER network')
            plt.bar(x+bar_width+0.05,SF_rmse,width=bar_width,color="#DFC286",label='SF network')

            
            for i in range(len(ave_rmse)):
                ave_x = [i,i+0.4]
                ave_y = [ave_rmse[i],ave_rmse[i]]
                if i == 0:
                    plt.plot(ave_x,ave_y,linestyle='--',color='r',linewidth=2,label='Average')
                else:
                    plt.plot(ave_x,ave_y,linestyle='--',color='r',linewidth=2)
                plt.text(i + 0.2,ave_rmse[i],'{:.1f}'.format(ave_rmse[i]*100),ha='center', va='bottom',fontsize=20)

            plt.gca().yaxis.set_major_formatter(formatter)
            
            plt.xticks(x + bar_width / 2, ablation_type)
            plt.ylim(0,0.6)
            plt.xlabel("Type of Ablation",fontsize=28,fontweight='bold')
            plt.ylabel("Increase in RMSE loss (%)",fontsize=28,fontweight='bold')
            # plt.legend(fontsize=16)
            # plt.title("% Increase in RMSE loss by Dataset in different network type",fontsize=20,fontweight='bold')
            plt.show()

        elif loss_type == "r2":
            plt.bar(x,ER_r2,width=bar_width,color="#608595",label='ER network')
            plt.bar(x+bar_width+0.05,SF_r2,width=bar_width,color="#DFC286",label='SF network')

            
            for i in range(len(ave_r2)):
                ave_x = [i,i+0.4]
                ave_y = [ave_r2[i],ave_r2[i]]
                if i == 0:
                    plt.plot(ave_x,ave_y,linestyle='--',color='r',linewidth=2,label='Average')
                else:
                    plt.plot(ave_x,ave_y,linestyle='--',color='r',linewidth=2)
                plt.text(i + 0.2,ave_r2[i],'{:.1f}'.format(ave_r2[i]*100),ha='center', va='bottom',fontsize=20)

            plt.gca().yaxis.set_major_formatter(formatter)
            
            plt.xticks(x + bar_width / 2, ablation_type)
            plt.ylim(-0.035,0.005)
            plt.xlabel("Type of Ablation",fontsize=28,fontweight='bold')
            plt.ylabel("Increase in R2 loss (%)",fontsize=28,fontweight='bold')
            # plt.legend(fontsize=16)
            plt.axhline(y=0,linewidth=2,color='black')
            # plt.title("% Increase in R2 loss by Dataset in different network type",fontsize=20,fontweight='bold')
            plt.show()

    elif figure == "train R0":
        net_type = 'MAE Errors of Testing Set in Synthetic SF Dataset'

        MAE_1 = []
        MAE_3 = []
        MAE_4 = []

        MAE_ER_1 = [0.183,0.130,0.103,0.071,0.069,0.093,0.132,0.156,0.261,0.353]
        MAE_ER_3 = [0.100,0.079,0.081,0.070,0.063,0.057,0.061,0.063,0.108,0.147]
        MAE_ER_4 = [0.083,0.067,0.068,0.060,0.058,0.052,0.049,0.044,0.055,0.063]

        MAE_SF_1 = [0.214,0.223,0.166,0.134,0.086,0.138,0.173,0.212,0.326,0.381]
        MAE_SF_3 = [0.103,0.131,0.114,0.105,0.078,0.090,0.085,0.109,0.131,0.140]
        MAE_SF_4 = [0.096,0.122,0.106,0.095,0.078,0.082,0.081,0.085,0.077,0.086]

        R0 = [1.2,1.4,1.6,1.8,2.2,2.4,2.6,2.8,3.2,3.4]
        s = [100,100,100,100,100,100,100,100,100,100]

        if net_type == 'MAE Errors of Testing Set in Synthetic ER Dataset':
            MAE_1 = MAE_ER_1
            MAE_3 = MAE_ER_3
            MAE_4 = MAE_ER_4
        elif net_type == 'MAE Errors of Testing Set in Synthetic SF Dataset':
            MAE_1 = MAE_SF_1
            MAE_3 = MAE_SF_3
            MAE_4 = MAE_SF_4

        fig = plt.figure(figsize=(14, 7))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 22  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 22  
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.plot(R0,MAE_1,color='steelblue',linewidth=3,label='R0={2.0}')
        plt.plot(R0,MAE_3,color='skyblue',linewidth=3,label='R0={1.5,2.0,2.5}')
        plt.plot(R0,MAE_4,color='c',linewidth=3,label='R0={1.5,2.0,2.5,3.0}')

        plt.scatter(R0,MAE_1,color='steelblue',s=s)
        plt.scatter(R0,MAE_3,color='skyblue',s=s)
        plt.scatter(R0,MAE_4,color='c',s=s)
        plt.ylim(0,0.40)
        plt.xlim(1.0,3.5)
        for i in range(len(R0)):
            plt.annotate(R0[i], (R0[i], MAE_1[i]),fontsize=24) 

        plt.xlabel("R0",fontsize=28,fontweight='bold')
        plt.ylabel("MAE",fontsize=28,fontweight='bold')
        plt.title(net_type,fontsize=28,fontweight='bold')
        
        plt.legend(fontsize=18)
        plt.show()

    elif figure == "testRate range":
        net_type = 'SF Network'

        MAE_ER_C = [0.052,0.079,0.114,0.074,0.080]
        MAE_ER_E = [0.091,0.147,0.173,0.140,0.158]
        MAE_ER_C_range = ["+51.9%","+119.2%","+42.3%","+53.8%"]
        MAE_ER_E_range = ["+61.5%","+90.1%","+53.8%","+73.6%"]

        MAE_SF_C = [0.082,0.091,0.114,0.098,0.102]
        MAE_SF_E = [0.139,0.211,0.247,0.200,0.226]
        MAE_SF_C_range = ["+11.0%","+39.0%","+19.5%","+24.4%"]
        MAE_SF_E_range = ["+51.8%","+77.7%","+43.9%","+62.6%"]

        test_type = ["Full Detection","Scenario 1","Scenario 2","Scenario 3","Scenario 4"]

        fig = plt.figure(figsize=(10, 7))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 2.0
        plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
        plt.rcParams['ytick.labelsize'] = 20  
        plt.rcParams['font.family'] = 'Times New Roman'

        bar_width = 0.35
        x = np.arange(len(test_type))

        if net_type == 'ER Network':
            MAE_C = MAE_ER_C
            MAE_E = MAE_ER_E
            MAE_C_range = MAE_ER_C_range
            MAE_E_range = MAE_ER_E_range
        elif net_type == 'SF Network':
            MAE_C = MAE_SF_C
            MAE_E = MAE_SF_E
            MAE_C_range = MAE_SF_C_range
            MAE_E_range = MAE_SF_E_range


        plt.bar(x,MAE_C,width=bar_width,color='#608595',label='Ct-Former')
        plt.bar(x+bar_width+0.05,MAE_E,width=bar_width,color='#DFC286',label='EpiEstim')

        for i in range(len(test_type)):
            if i != 0:
                plt.text(i,MAE_C[i],MAE_C_range[i - 1],ha='center', va='bottom',fontsize=18)
                plt.text(i+bar_width+0.05,MAE_E[i],MAE_E_range[i - 1],ha='center', va='bottom',fontsize=18)


        plt.xticks(x + bar_width / 2, test_type)
        plt.xlabel("Detection Scenario",fontsize=22,fontweight='bold')
        plt.ylabel("MAE",fontsize=22,fontweight='bold')
        plt.legend(fontsize=18)
        plt.title("MAE Error Across Detection Scenarios on the " + net_type,fontsize=22,fontweight='bold')
        plt.show()

    elif figure == "quantile":
        count = 1
        net = "ER"
        R0 = 3.4
        fold = "..\\results\\quantile\\"+ net + str(R0)
        
        truF = open(fold + '\\truth.txt')
        dF = open(fold + '\\duration.txt')
        MF = open(fold + '\\mid.txt')
        UF = open(fold + '\\high.txt')
        LF = open(fold + '\\lower.txt')
        

        linet = truF.readline()
        lined = dF.readline()
        linem = MF.readline()
        lineu = UF.readline()
        linel = LF.readline()

        while linet:
            linet = linet.split(' \n')[0]
            t_str = linet.split(' ')
            linem = linem.split(' \n')[0]
            m_str = linem.split(' ')
            lineu = lineu.split(' \n')[0]
            u_str = lineu.split(' ')
            linel = linel.split(' \n')[0]
            l_str = linel.split(' ')
            lined = lined.split(' \n')[0]
            d_str = lined.split(' ')

            if len(t_str) > 50:
                true = [float(t_str[i]) for i in range(len(t_str))]
                mean = [float(m_str[i]) for i in range(len(m_str))]
                lower = [float(l_str[i]) for i in range(len(l_str))]
                upper = [float(u_str[i]) for i in range(len(u_str))]
                duration = [float(d_str[i]) for i in range(len(d_str))]

                fig = plt.figure(figsize=(14, 7))
                plt.rcParams['font.weight'] = 'bold'
                plt.rcParams['axes.linewidth'] = 2.0
                plt.rcParams['grid.linewidth'] = 2.0
                plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
                plt.rcParams['ytick.labelsize'] = 16  
                # plt.rcParams['font.family'] = 'Times New Roman'

                start = int(duration[0])
                end = int(duration[1])

                x = [i for i in range(start,end)]
                if net == "SF" and R0 == 1.2:
                    x = [i for i in range(start-7,end-7)]
                if net == "ER" and R0 == 3.4:
                    x = [i for i in range(start-3,end-3)]
                if net == "SF" and R0 == 3.4:
                    x = [i for i in range(start-7,end-7)]
                plt.plot(x,true,'black',linewidth=3,label='Ground Truth')
                plt.plot(x,mean,'m',label="Estimated Mean",linewidth=3)
                plt.fill_between(x,lower,upper,color='m', alpha=0.2,label='Estimation Interval')
                
                if net == "SF" and R0 == 3.4:
                    plt.ylim(0,10.0)
                    plt.xlim(10,80)
                if net == "ER" and R0 == 3.4:
                    plt.ylim(0,4.0)
                    plt.xlim(20,100)
                if net == "ER" and R0 == 1.8:
                    plt.ylim(0,2.5)
                if net == "SF" and R0 == 1.8:
                    plt.ylim(0,7.0)
                if net == "SF" and R0 == 1.2:
                    plt.ylim(0,6.0)
                    plt.xlim(40,180)
                if net == "ER" and R0 == 1.2:
                    plt.ylim(0,2.25)
                    plt.xlim(50,400)
                # plt.title("Net=" + net + ",R0=" + str(R0),fontsize=28,fontweight='bold')
                plt.title("Simulation on the " + net + " network with R0=" + str(R0),fontsize=28,fontweight='bold')
                plt.xlabel("Days since start of outbreak",fontsize=28,fontweight='bold')
                plt.ylabel("Rt Value",fontsize=28,fontweight='bold')
                
                plt.axhline(y=1,linewidth=3)
                plt.legend(fontsize=18)

                # plt.savefig(fold_quantile + "/draw/" + str(count) + ".png")
                # plt.close()
                plt.show()

            count += 1
            linet = truF.readline()
            lined = dF.readline()
            linem = MF.readline()
            lineu = UF.readline()
            linel = LF.readline()
        truF.close()
        dF.close()
        MF.close()
        UF.close()
        LF.close()



























                





