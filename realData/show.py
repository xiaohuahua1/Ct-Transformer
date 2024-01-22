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
from scipy import interpolate

period = "test"
show = "mean"

df = pd.read_csv('data_daily_all.csv')
df.fillna(df.mean(),inplace=True)

phase = df[109:170]
ct_mean1 = np.array(phase["skewness"])
ct_mean1_list = ct_mean1.tolist()
x = [i for i in range(len(ct_mean1_list))]
plt.plot(x,ct_mean1_list)
plt.show()
# for i in range(len(ct_mean1_list)):
#     if math.isnan(ct_mean1_list[i]):
#         print(1)
#         ct_mean1_list[i] = np.nan
    
# indexes = range(len(ct_mean1_list))
# print(np.where(~np.isnan(ct_mean1_list)))
# known_indexes = np.where(~np.isnan(ct_mean1_list))

# interpolated_func = interpolate.interp1d(known_indexes, ct_mean1_list, kind='linear', fill_value="extrapolate")
# completed_data = interpolated_func(indexes)
# print(completed_data)

