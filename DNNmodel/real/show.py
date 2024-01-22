import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
from process import *
from args import *

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

overall = False
isDistrb = True
show_data = "skew"
smooth = True
num = 50

date_list,records_list,Rt_mean_list,Rt_lower_list,Rt_upper_list,ct_mean_list,ct_skew_list = readData("train")
args = CtTransformer_pretrain_args_parser()
# distrb = generate_distrb(ct_mean_list[2],ct_skew_list[2],6,3)
# print(distrb)
# print(' ')
# distrb = generate_distrb(ct_mean_list[2],ct_skew_list[2],6,4)
# print(distrb)


if overall:
    if smooth:
        ct_mean_list = getSmooth(ct_mean_list,7)
        ct_skew_list = getSmooth(ct_skew_list,7)
    dates = [datetime.strptime(i, '%Y-%m-%d').date() for i in date_list]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(dates[::7])

    if show_data == "rt":
        plt.plot(dates,Rt_mean_list,'r')
        plt.fill_between(dates,Rt_lower_list,Rt_upper_list,color='red', alpha=0.2)
        plt.axhline(y=1) 
    elif show_data == "mean":
        plt.plot(dates,ct_mean_list,'r')
    elif show_data == "skew":
        plt.plot(dates,ct_skew_list,'r')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(axis='y')
    plt.xlabel("date")
    plt.ylabel(show_data)
    plt.show()

# data = generate_distrb(ct_mean_list[10],ct_skew_list[10],100,4)
# print("mean:" + str(ct_mean_list[10]))
# print("skew:" + str(ct_skew_list[10]))
# print(data)

if isDistrb:
    length = len(Rt_mean_list)
    data,label,ct_all,ct_min,ct_max = generate_train_data(args,Rt_mean_list,ct_mean_list,ct_skew_list,num,smooth)
    # print(len(data))
    # print(len(label))
    # print(' ')
    # print(len(data[0]))
    # print(len(label[0]))
    # print(' ')
    # print(len(data[0][0]))


    group = args.group
    median_all = []
    low_all = []
    high_all = []
    
    for i in range(group):
        median_all.append([])
        low_all.append([])
        high_all.append([])
        for j in range(length):
            median,lower,upper = calculate(ct_all[i][j])
            median_all[i].append(median)
            low_all[i].append(lower)
            high_all[i].append(upper)

    x = [i for i in range(length)]
    if group == 6:
        fig, ax = plt.subplots(2,3,figsize=(25, 10))
        ax[0,0].plot(x,median_all[0],color='r')
        ax[0,0].fill_between(x,low_all[0],high_all[0],color='red', alpha=0.2)
        ax[0,0].set_title('[16-20]')

        ax[0,1].plot(x,median_all[1],color='r')
        ax[0,1].fill_between(x,low_all[1],high_all[1],color='red', alpha=0.2)
        ax[0,1].set_title('[21-24]')

        ax[0,2].plot(x,median_all[2],color='r')
        ax[0,2].fill_between(x,low_all[2],high_all[2],color='red', alpha=0.2)
        ax[0,2].set_title('[25-28]')

        ax[1,0].plot(x,median_all[3],color='r')
        ax[1,0].fill_between(x,low_all[3],high_all[3],color='red', alpha=0.2)
        ax[1,0].set_title('[29-32]')

        ax[1,1].plot(x,median_all[4],color='r')
        ax[1,1].fill_between(x,low_all[4],high_all[4],color='red', alpha=0.2)
        ax[1,1].set_title('[33-36]')

        ax[1,2].plot(x,median_all[5],color='r')
        ax[1,2].fill_between(x,low_all[5],high_all[5],color='red', alpha=0.2)
        ax[1,2].set_title('[37-40]')
    elif group == 8:
        fig, ax = plt.subplots(2,4,figsize=(25, 10))
        ax[0,0].plot(x,median_all[0],color='r')
        ax[0,0].fill_between(x,low_all[0],high_all[0],color='red', alpha=0.2)
        ax[0,0].set_title('[0-20]')

        ax[0,1].plot(x,median_all[1],color='r')
        ax[0,1].fill_between(x,low_all[1],high_all[1],color='red', alpha=0.2)
        ax[0,1].set_title('[20-21]')

        ax[0,2].plot(x,median_all[2],color='r')
        ax[0,2].fill_between(x,low_all[2],high_all[2],color='red', alpha=0.2)
        ax[0,2].set_title('[21-22]')

        ax[0,3].plot(x,median_all[3],color='r')
        ax[0,3].fill_between(x,low_all[3],high_all[3],color='red', alpha=0.2)
        ax[0,3].set_title('[22-23]')

        ax[1,0].plot(x,median_all[4],color='r')
        ax[1,0].fill_between(x,low_all[4],high_all[4],color='red', alpha=0.2)
        ax[1,0].set_title('[23-24]')

        ax[1,1].plot(x,median_all[5],color='r')
        ax[1,1].fill_between(x,low_all[5],high_all[5],color='red', alpha=0.2)
        ax[1,1].set_title('[24-25]')

        ax[1,2].plot(x,median_all[6],color='r')
        ax[1,2].fill_between(x,low_all[6],high_all[6],color='red', alpha=0.2)
        ax[1,2].set_title('[25-26]')

        ax[1,3].plot(x,median_all[7],color='r')
        ax[1,3].fill_between(x,low_all[7],high_all[7],color='red', alpha=0.2)
        ax[1,3].set_title('[26-40]')

    plt.xlabel("t")
    plt.ylabel("ct distrb")
    plt.legend()
    plt.show()


    