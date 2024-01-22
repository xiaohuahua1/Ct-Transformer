import matplotlib.pyplot as plt
import math

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

def mse(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    for i in range(num):
        result += math.pow((pred[i] - y[i]),2)
    result /= num
    return result
    
def mae(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    for i in range(num):
        result += abs(pred[i] - y[i])
    result /= num
    return result

def rmse(mse):
    return math.sqrt(mse)
    
def mape(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    for i in range(num):
        result += abs((y[i] - pred[i]) / y[i])
    result /= num
    result *= 100
    return result

def r2(y,pred):
    result = 0.0
    num  = len(y)
    if num == 0:
        return result
    mean_actual = sum(y) / len(y)
    ss_total = sum((actual_val - mean_actual) ** 2 for actual_val in y)
    ss_residual = sum((actual_val - predicted_val) ** 2 for actual_val, predicted_val in zip(y, pred))
    
    result = 1 - (ss_residual / ss_total)
    return result

if __name__ == '__main__':

    figure = "test Rate"

    if figure == "ER result":
        isV = False
        isE = False
        R0 = 3.4
        path = "..\\results\\ER\\d=10R=" + str(R0) + "\\test\\draw"
        bayesF = open(path + "\\Bayesian.txt")
        duraF = open(path + "\\duration.txt")
        regreF = open(path + "\\regression.txt")
        truF = open(path + "\\truth.txt")
        ctF = open(path + "\\Ct-Former.txt")
        tftF = open(path + "\\TFT.txt")
        tranF = open(path + "\\transformer.txt")
        # viroF = open("..\\results\\ER\\d=10R=3.2\\test\\draw\\viroRt.txt")
        viroF = open(path + "\\viroRt.txt")
        esF = open(path + "\\EpiRt.txt")

        linebayes = bayesF.readline()
        linedura = duraF.readline()
        lineregre = regreF.readline()
        linetru = truF.readline()
        linect = ctF.readline()
        linetft = tftF.readline()
        linetran = tranF.readline()
        lineviro = viroF.readline()
        linees = esF.readline()

        while linebayes:
            linebayes = linebayes.split(' \n')[0]
            bayes_str = linebayes.split(' ')
            linedura = linedura.split(' \n')[0]
            dura_str = linedura.split(' ')
            lineregre = lineregre.split(' \n')[0]
            regre_str = lineregre.split(' ')
            linetru = linetru.split(' \n')[0]
            tru_str = linetru.split(' ')
            linect = linect.split(' \n')[0]
            ct_str = linect.split(' ')
            linetft = linetft.split(' \n')[0]
            tft_str = linetft.split(' ')
            linetran = linetran.split(' \n')[0]
            tran_str = linetran.split(' ')
            lineviro = lineviro.split(' \n')[0]
            viro_str = lineviro.split(' ')
            linees = linees.split(' \n')[0]
            es_str = linees.split(' ')

            bayes = [float(bayes_str[i]) for i in range(len(bayes_str))]
            dura = [float(dura_str[i]) for i in range(len(dura_str))]
            regre = [float(regre_str[i]) for i in range(len(regre_str))]
            tru = [float(tru_str[i]) for i in range(len(tru_str))]
            ct = [float(ct_str[i]) for i in range(len(ct_str))]
            tft = [float(tft_str[i]) for i in range(len(tft_str))]
            tran = [float(tran_str[i]) for i in range(len(tran_str))]
            viro = [float(viro_str[i]) for i in range(len(viro_str))]
            epi = [float(es_str[i]) for i in range(len(es_str))]
            start = int(dura[0])
            end = int(dura[1])
            if R0 == 3.4:
                x = [i for i in range(start-5,end-5)]
            else:
                x = [i for i in range(start,end)]

            viro = viro[start:end]
            epi = epi[start:end]
            if isV:
                mae_value = mae(tru,viro)
                mse_value = mse(tru,viro)
                rmse_value = rmse(mse_value)
                r2_value = r2(tru,viro)
                print(mae_value)
                print(rmse_value)
                print(r2_value)
            if isE:
                mae_value = mae(tru,epi)
                mse_value = mse(tru,epi)
                rmse_value = rmse(mse_value)
                r2_value = r2(tru,epi)
                print(mae_value)
                print(rmse_value)
                print(r2_value)

            fig = plt.figure(figsize=(14, 7))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rcParams['grid.linewidth'] = 2.0
            plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
            plt.rcParams['ytick.labelsize'] = 20  
            plt.rcParams['font.family'] = 'Times New Roman'

            # plt.plot(x,tru,'black',linewidth=3,label="Ground Truth")
            # plt.plot(x,ct,'blueviolet',label="Ct-Former",linewidth=3)
            # plt.plot(x,tft,'crimson',label="TFT",linewidth=2)
            # # plt.plot(x,tran,'royalblue',label="Transformer",linewidth=2)
            # plt.plot(x,viro,'lightcoral',label="ViroSolver",marker="s",markevery=15,linestyle="-.",linewidth=2)
            # # plt.plot(x,regre,'g',label="Regression",marker="s",markevery=15,linestyle="-.",linewidth=2)
            # # plt.plot(x,bayes,'goldenrod',label="EpiEstim",marker="x",markevery=15,linestyle="dashed",linewidth=2)

            plt.plot(x,tru,'black',linewidth=3,label="Ground Truth")
            plt.plot(x,ct,'royalblue',label="Ct-Former",linewidth=3)
            plt.plot(x,tft,'coral',label="TFT",marker="o",markevery=15,linestyle="dashed",linewidth=3)
            # plt.plot(x,tran,'royalblue',label="Transformer",linewidth=2)
            plt.plot(x,viro,'g',label="ViroSolver",marker="s",markevery=15,linestyle="-.",linewidth=3)
            # plt.plot(x,regre,'g',label="Regression",marker="s",markevery=15,linestyle="-.",linewidth=2)
            plt.plot(x,epi,'goldenrod',label="EpiEstim",marker="x",markevery=15,linestyle="dashed",linewidth=3)
            if R0 == 3.4:
                plt.ylim(0,3.6)
                plt.xlim(20,100)
            if R0 == 1.8:
                plt.ylim(0,3.0)
                plt.xlim(20,180)

            # plt.title("R0="+str(R0),fontsize=28,fontweight='bold')
            plt.title("Rt estimation results for simulation with R0="+str(R0),fontsize=28,fontweight='bold')
            plt.xlabel("Days since start of outbreak",fontsize=28,fontweight='bold')
            plt.ylabel("Rt Value",fontsize=28,fontweight='bold')
            # plt.grid()
            plt.axhline(y=1,linewidth=3)
            plt.legend(fontsize=18)
            plt.show()

            linebayes = bayesF.readline()
            linedura = duraF.readline()
            lineregre = regreF.readline()
            linetru = truF.readline()
            linect = ctF.readline()
            linetft = tftF.readline()
            linetran = tranF.readline()
            lineviro = viroF.readline()
            linees = esF.readline()
        bayesF.close()
        duraF.close()
        regreF.close()
        truF.close()
        ctF.close()
        tftF.close()
        tranF.close()
        viroF.close()
        esF.close()

    elif figure == "SF result":
        isE = False
        R0 = 1.8
        path = "..\\results\\SF\\d=10R=" + str(R0) + "\\test\\draw"
        bayesF = open(path + "\\Bayesian.txt")
        duraF = open(path + "\\duration.txt")
        regreF = open(path + "\\regression.txt")
        truF = open(path + "\\truth.txt")
        ctF = open(path + "\\Ct-Former.txt")
        tftF = open(path + "\\TFT.txt")
        tranF = open(path + "\\transformer.txt")
        endF = open(path + "\\end.txt")
        linearF = open(path + "\\linear.txt")
        esF = open(path + "\\EpiRt.txt")

        linebayes = bayesF.readline()
        linedura = duraF.readline()
        lineregre = regreF.readline()
        linetru = truF.readline()
        linect = ctF.readline()
        linetft = tftF.readline()
        linetran = tranF.readline()
        lineend = endF.readline()
        linelin = linearF.readline()
        linees = esF.readline()

        while linebayes:
            linebayes = linebayes.split(' \n')[0]
            bayes_str = linebayes.split(' ')
            linedura = linedura.split(' \n')[0]
            dura_str = linedura.split(' ')
            lineregre = lineregre.split(' \n')[0]
            regre_str = lineregre.split(' ')
            linetru = linetru.split(' \n')[0]
            tru_str = linetru.split(' ')
            linect = linect.split(' \n')[0]
            ct_str = linect.split(' ')
            linetft = linetft.split(' \n')[0]
            tft_str = linetft.split(' ')
            linetran = linetran.split(' \n')[0]
            tran_str = linetran.split(' ') 
            lineend = lineend.split(' \n')[0]
            end_str = lineend.split(' ') 
            linelin = linelin.split(' \n')[0]
            lin_str = linelin.split(' ') 
            linees = linees.split(' \n')[0]
            es_str = linees.split(' ')

            bayes = [float(bayes_str[i]) for i in range(len(bayes_str))]
            dura = [float(dura_str[i]) for i in range(len(dura_str))]
            regre = [float(regre_str[i]) for i in range(len(regre_str))]
            tru = [float(tru_str[i]) for i in range(len(tru_str))]
            ct = [float(ct_str[i]) for i in range(len(ct_str))]
            tft = [float(tft_str[i]) for i in range(len(tft_str))]
            tran = [float(tran_str[i]) for i in range(len(tran_str))]
            toend = [float(end_str[i]) for i in range(len(end_str))]
            lin = [float(lin_str[i]) for i in range(len(lin_str))]
            epi = [float(es_str[i]) for i in range(len(es_str))]

            start = int(dura[0])
            end = int(dura[1])
            x = [i for i in range(start,end)]

            epi = epi[start:end]
            if isE:
                mae_value = mae(tru,epi)
                mse_value = mse(tru,epi)
                rmse_value = rmse(mse_value)
                r2_value = r2(tru,epi)
                print(mae_value)
                print(rmse_value)
                print(r2_value)

            fig = plt.figure(figsize=(14, 7))
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rcParams['grid.linewidth'] = 2.0
            plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
            plt.rcParams['ytick.labelsize'] = 20  
            plt.rcParams['font.family'] = 'Times New Roman'

            plt.plot(x,tru,'black',linewidth=3,label="Ground Truth")
            plt.plot(x,ct,'royalblue',label="Ct-Former Sup.",linewidth=3)
            plt.plot(x,toend,'g',label="Ct-Former Fine-Tuning",marker="s",markevery=15,linestyle="-.",linewidth=3)
            plt.plot(x,lin,'coral',label="Ct-Former Lin.Prob.",marker="o",markevery=15,linestyle="dashed",linewidth=3)

            if R0 == 3.4:
                plt.ylim(0,7)
            if R0 == 1.8:
                plt.ylim(0,5)

            plt.title("R0=" + str(R0),fontsize=28,fontweight='bold')
            plt.xlabel("Days since start of outbreak",fontsize=28,fontweight='bold')
            plt.ylabel("Rt Value",fontsize=28,fontweight='bold')
            # plt.grid()
            plt.axhline(y=1,linewidth=3)
            plt.legend(fontsize=18)
            plt.show()

            linebayes = bayesF.readline()
            linedura = duraF.readline()
            lineregre = regreF.readline()
            linetru = truF.readline()
            linect = ctF.readline()
            linetft = tftF.readline()
            linetran = tranF.readline()
            lineend = endF.readline()
            linelin = linearF.readline()
            linees = esF.readline()

        bayesF.close()
        duraF.close()
        regreF.close()
        truF.close()
        ctF.close()
        tftF.close()
        tranF.close()
        endF.close()
        linearF.close()
        esF.close()

    elif figure == "test Rate":
        Scenario = 1
        net = "SF"
        path = "..\\results\\ER\\testRate\\" + net + "2.4\\draw"
        if Scenario == 1:
            title = "Scenario 1: a fixed detection probability of 25%"
        elif Scenario == 2:
            title = "Scenario 2: a fixed detection probability of 10%"
        elif Scenario == 3:
            title = "Scenario 3: detection probability increases from 15% to 60%"
        elif Scenario == 4:
            title = "Scenario 4: probability of 25% except for the under-detection"

        truF = open(path + "\\truth.txt")
        IF = open(path + "\\num.txt")
        ctF = open(path + "\\all.txt")
        duraF = open(path + "\\duration.txt")
        epiF = open(path + "\\Epi_all.txt")

        testF = open(path + "\\test" + str(Scenario) +".txt")
        numF = open(path + "\\num" + str(Scenario) +".txt")
        eF = open(path + "\\Epi_test" + str(Scenario) +".txt")

        linetru = truF.readline()
        linect = ctF.readline()
        lineI = IF.readline()
        linedura = duraF.readline()
        lineepi = epiF.readline()

        linet = testF.readline()
        linen = numF.readline()
        linee = eF.readline()

        while linetru:
            linetru = linetru.split(' \n')[0]
            tru_str = linetru.split(' ')
            linect = linect.split(' \n')[0]
            ct_str = linect.split(' ')
            linedura = linedura.split(' \n')[0]
            dura_str = linedura.split(' ')
            lineI = lineI.split(' \n')[0]
            I_str = lineI.split(' ')
            lineepi = lineepi.split(' \n')[0]
            epi_str = lineepi.split(' ')

            linet = linet.split(' \n')[0]
            t_str = linet.split(' ')
            linen = linen.split(' \n')[0]
            n_str = linen.split(' ')
            linee = linee.split(' \n')[0]
            e_str = linee.split(' ')

            tru = [float(tru_str[i]) for i in range(len(tru_str))]
            ct = [float(ct_str[i]) for i in range(len(ct_str))]
            dura = [float(dura_str[i]) for i in range(len(dura_str))]
            I = [float(I_str[i]) for i in range(len(I_str))]
            epi = [float(epi_str[i]) for i in range(len(epi_str))]

            test = [float(t_str[i]) for i in range(len(t_str))]
            num = [float(n_str[i]) for i in range(len(n_str))]
            teste = [float(e_str[i]) for i in range(len(e_str))]

            start = int(dura[0])
            end = int(dura[1])

            I = I[start:end]
            num = num[start:end]
            epi = epi[start:end]
            teste = teste[start:end]

            mae_all = mae(tru,epi)
            mse_all = mse(tru,epi)
            rmse_all = rmse(mse_all)
            r2_all = r2(tru,epi)

            print(mae_all)
            print(rmse_all)
            print(r2_all)
            print(' ')

            mae_test = mae(tru,teste)
            mse_test = mse(tru,teste)
            rmse_test = rmse(mse_test)
            r2_test = r2(tru,teste)

            print(mae_test)
            print(rmse_test)
            print(r2_test)
            
            if net == "ER":
                x = [i for i in range(start-5,end-5)]
            if net == "SF":
                x = [i for i in range(start+3,end+3)]
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['axes.linewidth'] = 2.0
            plt.rcParams['grid.linewidth'] = 2.0
            plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
            plt.rcParams['ytick.labelsize'] = 20  
            plt.rcParams['font.family'] = 'Times New Roman'
            fig,ax1 = plt.subplots(figsize=(14, 7))
            # ax1.bar(x, I,color='darkgrey',label='Total Infection')
            # ax1.bar(x,num,color='dimgrey',label='Detected Infection')
            ax1.bar(x, I,color='darkgrey')
            ax1.bar(x,num,color='dimgrey')
            ax1.set_xlabel('Days since start of outbreak',fontsize=28,fontweight='bold')
            ax1.set_ylabel('Infection',fontsize=28,fontweight='bold')
            # ax1.tick_params(axis='both', labelsize=18,width=1)
            if net == "ER":
                ax1.set_ylim(0,30000)
                ax1.set_xlim(20,140)
            if net == "SF":
                ax1.set_ylim(0,35000)
                ax1.set_xlim(20,120)
                
            ax1.tick_params('y')

            ax2 = ax1.twinx()
            ax2.plot(x,tru,'black',linewidth=3,label='Ground Truth')
            ax2.plot(x,ct,'coral',label="Ct-Former in Full Detection",linewidth=3)
            ax2.plot(x,test,'firebrick',label="Ct-Former in Detect Scenario",marker="s",markevery=15,linestyle="-.",linewidth=3)

            # ax2.plot(x,epi,'dodgerblue',label="EpiEstim Total Infection",linewidth=3)
            # ax2.plot(x,teste,'mediumblue',label="EpiEstim Detected Infection",marker="s",markevery=15,linestyle="-.",linewidth=3)
            ax2.plot(x,epi,'dodgerblue',label="EpiEstim in Full Detection",linewidth=3)
            ax2.plot(x,teste,'mediumblue',label="EpiEstim in Detect Scenario",marker="s",markevery=15,linestyle="-.",linewidth=3)
            ax2.set_ylabel('Rt Value',fontsize=28,fontweight='bold')
            if net == "ER":
                ax2.set_ylim(0,3.0)
            if net == "SF":
                ax2.set_ylim(0,8.0)
            # ax1.tick_params(axis='both', labelsize=18,width=3)
            ax2.tick_params('y')

            ax2.axhline(y=1,linewidth=3,color='darkgreen')
            ax2.legend(fontsize=16)
            plt.title(title,fontsize=28,fontweight='bold')
            plt.show()

            linetru = truF.readline()
            linect = ctF.readline()
            linedura = duraF.readline()
            lineI = IF.readline()
            linet = testF.readline()
            linen = numF.readline()
        truF.close()
        ctF.close()
        duraF.close()
        IF.close()
        testF.close()
        numF.close()







