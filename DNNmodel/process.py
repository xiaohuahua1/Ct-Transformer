import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import csv
import os

if __name__ == '__main__':
    task = "create test csv"

    if task == "test ct to one":
        fold = "..\\results\\SF\\d=10R=3.4\\test"

        old_mean = fold + "\\ctMean.txt"
        new_mean = fold + "\\ctMean_all.txt"

        old_skew = fold + "\\ctSkew.txt"
        new_skew = fold + "\\ctSkew_all.txt"

        old_distrb = fold + "\\distrb.txt"
        new_distrb = fold + "\\distrb_all.txt"

        os.rename(old_mean,new_mean)
        os.rename(old_skew,new_skew)
        os.rename(old_distrb,new_distrb)

        nmf = open(fold + "\\ctMean.txt","w")
        nsf = open(fold + "\\ctSkew.txt","w")
        ndf = open(fold + "\\distrb.txt","w")

        count = 0

        mf = open(fold + "\\ctMean_all.txt")
        sf = open(fold + "\\ctSkew_all.txt")
        df = open(fold + "\\distrb_all.txt")

        linem = mf.readline()
        lines = sf.readline()
        lined = df.readline()

        while linem:
            if count % 5 == 0:
                nmf.write(linem)
                nsf.write(lines)
                ndf.write(lined)
            linem = mf.readline()
            lines = sf.readline()
            lined = df.readline()
            count += 1
        mf.close()
        sf.close()
        df.close()
        nmf.close()
        nsf.close()
        ndf.close()

    elif task == "average":
        data = [0.754,0.907,0.937,0.967,0.0979,0.987,0.989,0.991,0.989,0.990]
        n = len(data)
        result = 0.0
        for i in range(n):
            result += data[i]
        result /= n
        print(result)

    elif task == "create test csv":
        output_fold = "..\\results\\SF\\d=10R=2.4\\test\\ctData"
        fold = "..\\results\\SF\\d=10R=2.4\\test\\draw"

        nF = open(fold + "\\num4.txt")
        oF = open(output_fold + "\\dailyNumID29-4.csv",'w',newline="")
        linen = nF.readline()
        writer = csv.writer(oF)

        data = []
        head = ["date","imported","local"]
        data.append(head)

        while linen:
            linen = linen.split(' \n')[0]
            n_str = linen.split(' ')

            testNum = [int(n_str[i]) for i in range(len(n_str))]
            for i in range(len(testNum)):
                if i == 0:
                    data.append(["0","1","0"])
                else:
                    data.append([str(i),"0",str(testNum[i])])
            
            writer.writerows(data)
            linen = nF.readline()
        nF.close()
        oF.close()



