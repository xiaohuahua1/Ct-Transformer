import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from tqdm import tqdm
from itertools import chain
import torch
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from process import *

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

def getResult():
    pred = []
    lower = []
    upper = []

    MF = open("results\\pred.txt")
    LF = open("results\\lower.txt")
    UF = open("results\\upper.txt")

    linem = MF.readline()
    linel = LF.readline()
    lineu = UF.readline()

    while linem:
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')
        linel = linel.split(' \n')[0]
        l_str = linel.split(' ')
        lineu = lineu.split(' \n')[0]
        u_str = lineu.split(' ')

        # pred = [float(m_str[i]) for i in range(len(m_str))]
        for i in range(len(m_str)):
            pred.append(float(m_str[i]))
            lower.append(float(l_str[i]))
            upper.append(float(u_str[i]))
        linem = MF.readline()
        linel = LF.readline()
        lineu = UF.readline()
    MF.close()
    LF.close()
    UF.close()
    return pred,lower,upper

show = "rt"
df = pd.read_csv('data_daily_all.csv')

df.fillna(df.mean(),inplace=True)

phase1 = df[36:109]
phase2 = df[170:296] 
phase3 = df[109:170]

pred,lower,upper = getResult()

date1 = np.array(phase1["date"])
date1_list = date1.tolist()
date2 = np.array(phase2["date"])
date2_list = date2.tolist()
date3 = np.array(phase3["date"])
date3_list = date3.tolist()
date_list = date1_list + date3_list + date2_list
# date_list = date1_list

ct_mean1 = np.array(phase1["mean"])
ct_mean1_list = ct_mean1.tolist()
ct_mean2 = np.array(phase2["mean"])
ct_mean2_list = ct_mean2.tolist()
ct_mean3 = np.array(phase3["mean"])
ct_mean3_list = ct_mean3.tolist()
ct_mean_list = ct_mean1_list + ct_mean3_list + ct_mean2_list
ct_mean_list = getSmooth(ct_mean_list,14)

ct_skew1 = np.array(phase1["skewness"])
ct_skew1_list = ct_skew1.tolist()
ct_skew2 = np.array(phase2["skewness"])
ct_skew2_list = ct_skew2.tolist()
ct_skew3 = np.array(phase3["skewness"])
ct_skew3_list = ct_skew3.tolist()
ct_skew_list = ct_skew1_list + ct_skew3_list + ct_skew2_list
ct_skew_list = getSmooth(ct_skew_list,14)

Rt_mean1 = np.array(phase1['local.rt.mean'])
Rt_mean1_list = Rt_mean1.tolist()
Rt_mean2 = np.array(phase2['local.rt.mean'])
Rt_mean2_list = Rt_mean2.tolist()
Rt_mean3 = np.array(phase3['local.rt.mean'])
Rt_mean3_list = Rt_mean3.tolist()
Rt_mean_list = Rt_mean1_list + Rt_mean3_list + Rt_mean2_list
# pred_mean_list = Rt_mean1_list + pred

Rt_lower1 = np.array(phase1['local.rt.lower'])
Rt_lower1_list = Rt_lower1.tolist()
Rt_lower2 = np.array(phase2['local.rt.lower'])
Rt_lower2_list = Rt_lower2.tolist()
Rt_lower3 = np.array(phase3['local.rt.lower'])
Rt_lower3_list = Rt_lower3.tolist()
Rt_lower_list = Rt_lower1_list + Rt_lower3_list + Rt_lower2_list
# pred_lower_list = Rt_lower1_list + lower

Rt_upper1 = np.array(phase1['local.rt.upper'])
Rt_upper1_list = Rt_upper1.tolist()
Rt_upper2 = np.array(phase2['local.rt.upper'])
Rt_upper2_list = Rt_upper2.tolist()
Rt_upper3 = np.array(phase3['local.rt.upper'])
Rt_upper3_list = Rt_upper3.tolist()
Rt_upper_list = Rt_upper1_list + Rt_upper3_list + Rt_upper2_list
# pred_upper_list = Rt_upper1_list + upper

records1 = np.array(phase1['records'])
records1_list = records1.tolist()
records2 = np.array(phase2['records'])
records2_list = records2.tolist()
records3 = np.array(phase3['records'])
records3_list = records3.tolist()
records = records1_list + records3_list + records2_list


dates = [datetime.strptime(i, '%Y/%m/%d').date() for i in date_list]
date_pred = [datetime.strptime(i, '%Y/%m/%d').date() for i in date2_list]


if show == "mean":
    fig = plt.figure(figsize=(28, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['grid.linewidth'] = 2.0
    plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
    plt.rcParams['ytick.labelsize'] = 20  
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(dates[::21])
    plt.plot(dates,ct_mean_list,'steelblue',linewidth=3)
    plt.legend(fontsize=18)

    plt.gcf().autofmt_xdate() 
    plt.ylim(20,28)
    plt.xlabel("Date",fontsize=28,fontweight='bold')
    plt.ylabel("Ct Mean Value",fontsize=28,fontweight='bold')
    plt.show()

elif show == "skew":
    fig = plt.figure(figsize=(28, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['grid.linewidth'] = 2.0
    plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
    plt.rcParams['ytick.labelsize'] = 20  
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(dates[::21])
    plt.plot(dates,ct_skew_list,'steelblue',linewidth=3)
    plt.legend(fontsize=18)

    plt.gcf().autofmt_xdate() 
    plt.ylim(-0.4,0.8)
    plt.xlabel("Date",fontsize=28,fontweight='bold')
    plt.ylabel("Ct Skew Value",fontsize=28,fontweight='bold')
    plt.axhline(y=0,linewidth=3,linestyle="dashed",color='darkgrey')
    plt.show()
elif show == "rt":

    fig,ax1 = plt.subplots(figsize=(28, 8))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['grid.linewidth'] = 2.0
    plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
    plt.rcParams['ytick.labelsize'] = 20  
    plt.rcParams['font.family'] = 'Times New Roman'
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(dates[::21])

    ax1.bar(dates,records,color='darkgrey')
    ax1.set_ylim(0,140)
    ax1.set_xlabel("Date",fontsize=25,fontweight='bold')
    ax1.set_ylabel('Records by sampling date',fontsize=25,fontweight='bold')
    ax1.tick_params(axis='both', labelsize=18)

    ax2 = ax1.twinx()
    ax2.plot(dates,Rt_mean_list,'r',linewidth=3)
    ax2.plot(date_pred,pred,'darkviolet',linewidth=3)
    ax2.set_ylim(0,5)
    ax2.legend(fontsize=16)
    ax2.fill_between(dates,Rt_lower_list,Rt_upper_list,color='r', alpha=0.2)
    ax2.fill_between(date_pred,lower,upper,color='darkviolet', alpha=0.2)

    # ax2.gcf().autofmt_xdate()

    ax2.set_xlabel("Date",fontsize=28,fontweight='bold')
    ax2.set_ylabel("Rt Value",fontsize=28,fontweight='bold')
    ax2.axhline(y=1,linewidth=2,linestyle="dashed") 
    ax1.tick_params(axis='both', labelsize=18)
    plt.gcf().autofmt_xdate()
    plt.show()

