import random
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

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

def read_info(path,Inum,day_list):
    I = []
    mean = []
    distrb = []
    distrb_list = []
    pro_list = []

    dF = open(path + "\\distrbOnset.txt")
    IF = open(path + "\\I.txt")
    mF = open(path + "\\ctMeanOnset.txt")

    lineI = IF.readline()
    lined = dF.readline()
    linem = mF.readline()

    while lined:
        lineI = lineI.split(' \n')[0]
        I_str = lineI.split(' ')
        lined = lined.split(' \n')[0]
        d_str = lined.split(' ')
        linem = linem.split(' \n')[0]
        m_str = linem.split(' ')

        if len(I_str) > 50:
            #127
            I = [float(I_str[i]) for i in range(len(I_str))]
            #126
            distrb = [float(d_str[i]) for i in range(len(d_str))]
            mean = [float(m_str[i]) for i in range(len(m_str))]
            distrb_element = []
            for i in range(len(distrb)):
                if i % 25 == 0:
                    if i != 0:
                        distrb_list.append(distrb_element)
                    distrb_element = []
                distrb_element.append(distrb[i])
            distrb_list.append(distrb_element)

            start,end = stable(I,Inum)
            I = I[start:end]
            distrb_list = distrb_list[start - 1:end - 1]
            mean = mean[start - 1:end - 1]

            break
    lineI = IF.readline()
    lined = dF.readline()

    IF.close()
    dF.close()

    for i in range(len(day_list)):
        day = day_list[i]
        prob = []
        p_all = 0.0
        for j in range(len(distrb_list[day])):
            p = distrb_list[day][j]
            p_all += p
            prob.append(p_all)
        max_value = max(prob)
        min_value = min(prob)
        normal_prob = [(x - min_value) / (max_value - min_value) for x in prob]
        pro_list.append(normal_prob)
    
    return pro_list
 
def generate(pro_list,num_list):
    dot_list = []
    for i in range(len(num_list)):
        dot = []
        num = num_list[i]
        for j in range(num):
            ct = 0.0
            p = random.random()
            for z in range(len(pro_list[i])):
                if p <= pro_list[i][z]:
                    ct = z + 16.0
                    break
            p1 = random.random()
            ct = ct + p1
            dot.append(ct)
        dot_list.append(dot)
    
    return dot_list

def readcsv(path,dot_list,day_list):
    data = [['value','class']]
    for i in range(len(day_list)):
        day = day_list[i]
        dot = dot_list[i]
        for j in range(len(dot)):
            ele = [dot[j],day]
            data.append(ele)

    f = open(path + "//dot.csv",'w',newline='')
    writer = csv.writer(f)
    writer.writerows(data)

def caltile(pro_list,num):
    q1_list = []
    q2_list = []
    q3_list = []
    for i in range(len(pro_list)):
        dot = []
        pro = pro_list[i]
        for j in range(num):
            ct = 0.0
            p = random.random()
            for z in range(len(pro)):
                if p <= pro[z]:
                    ct = z + 16.0
                    break
            p1 = random.random()
            ct = ct + p1
            dot.append(ct)
        q1 = np.percentile(dot, 25)
        q2 = np.percentile(dot, 50)  # 中位数
        q3 = np.percentile(dot, 75)

        q1_list.append(q1)
        q2_list.append(q2)
        q3_list.append(q3)

    q1_list = getSmooth(q1_list,7)
    q2_list = getSmooth(q2_list,7)
    q3_list = getSmooth(q3_list,7)
    return q1_list,q2_list,q3_list

def vio(path,pro_list):

    # q1_list,q2_list,q3_list = caltile(pro_list,200)
    # print(q1_list)
    # print(q2_list)
    # print(q3_list)
    q1_list = [19.471710001743113, 19.46656610069279,19.48944327833544, \
               19.443153410007293, 19.48432620559805, 19.49407076572677, 19.489693294071213, 19.522808396275995, 19.576604947935927, 19.624358922864282, \
                19.69838205919544, 19.719438834564624, 19.73683630575902, 19.763296184961398, 19.764886865254077, 19.78906503610015, 19.781227871989845, \
                19.825275061885627, 19.85970214406613, 19.890240621656808, 19.836890678885435, 19.906545375361617, 19.97377154500986, 19.997467259706948,\
                19.962297444401706, 19.945440414633328, 19.919195237074774, 19.931275602884064, 19.939307893923196, 19.877453851773964, 19.880538494058747, \
                19.87621489217498, 19.890697593696153, 19.91056831579677, 19.91501862066936, 19.90716462085612, 19.906709883653946, 19.96976357596699, \
                20.024738221407148, 20.07083840602467, 20.10759726065994, 20.158566599674515, 20.17257200021393, 20.210013058213644, 20.269656689890763, \
                20.338782651981514, 20.394920181156117, 20.474434899212604, 20.551132821759065, 20.626440210631127, 20.695078529350585, 20.692123723677355, \
                20.67908368043955, 20.679547088327315, 20.759405831957885, 20.785692440385418, 20.804013557287835, 20.800569896889936, 20.8439230433652, \
                20.872357590579927, 20.883444148685804, 20.83165815130989, 20.747977060153055, 20.69731424691731, 20.746620405599625, 20.7126881242224, \
                20.700701432948375, 20.7053462008565, 20.657553341706553, 20.657553341706553, 20.657553341706553, 20.657553341706553]

    q2_list = [21.12792620705945, 21.162783869957614,21.145475343941994, 21.10911562313838, 21.153689819956664, 21.116333829384562, 21.027515052299464, 21.06293200050484, 21.112527572080438, \
                21.14978625768108, 21.15555726648651, 21.184144544548797, 21.297092033291275, 21.36690326510893, 21.355622839890703, 21.423949136601884, \
                21.413210278280655, 21.455026017166777, 21.473460243589777, 21.46233222439654, 21.446582684296366, 21.480412358300395, 21.504343381746274, \
                21.57949361999717, 21.608967077814885, 21.606367145557133, 21.56445485664592, 21.5818441616121, 21.608117610151407, 21.518701315873923, \
                21.552530335532673, 21.527567373394824, 21.553751431854177, 21.5407551426722, 21.54261676661134, 21.54122287478818, 21.589237431650126, \
                21.646716780247043, 21.646829014766016, 21.743538630785054, 21.801890940497902, 21.880678737047283, 21.891415329315503, 21.929088851915413, \
                21.9759977958341, 22.112452101407776, 22.131895102687594, 22.226576438465194, 22.32887706258757, 22.432925355586665, 22.517684071480726, \
                22.522227316378228, 22.46461161209964, 22.447056643756348, 22.499066446361642, 22.46193951587791, 22.53989029575775, 22.51435964824461, \
                22.500526840347984, 22.587076547966642, 22.72069509521203, 22.69925386580639, 22.726252813843594, 22.70623071816993, 22.713204445158148, \
                22.69532946949672, 22.72053128529016, 22.662228540793368, 22.720380791198604, 22.720380791198604, 22.720380791198604, 22.720380791198604]
    
    q3_list = [22.87904830889786, 22.886417754081784,22.947200923306793, 22.839815017710122, 22.972968155195076, 22.90167850375494, 22.857449630840936, 22.78127202367099, 22.734151373754205, \
                22.8357174658521, 22.811992805975333, 22.83854653140009, 22.96251962769103, 23.031191976834492, 23.058715469956358, 23.132863879399455, \
                23.020236712968586, 23.078876161847056, 23.126747638043167, 23.0511343191015, 23.089881535125553, 23.158441966580096, 23.149564906421556,\
                23.22542407198336, 23.297807231299224, 23.316886359243913, 23.324457828155168, 23.249398121701372, 23.271451974134187, 23.19912041171261, \
                23.208207775354854, 23.19555697237598, 23.204743275318386, 23.185240024134917, 23.22055231433958, 23.184017456290114, 23.30851112040221, \
                23.417916487849336, 23.425648205024434, 23.52136359504694, 23.605193402331384, 23.688478268907822, 23.78583477496082, 23.815913133771005, \
                23.928111610974423, 24.08609953233691, 24.204650728613867, 24.284583237121705, 24.419296950768757, 24.47681783474535, 24.502160407212013, \
                24.44448373194493, 24.42715159736594, 24.404908080261862, 24.514430227620213, 24.467209623376196, 24.57734763499672, 24.625510870673338, \
                24.714715826733247, 24.719760868416415, 24.763811096966254, 24.7139120522079, 24.76200711789162, 24.681429247375036, 24.63405933135737, \
                24.589089486785834, 24.588786372473482, 24.551421589191587, 24.63557114372527, 24.63557114372527, 24.63557114372527, 24.63557114372527]
    
    
    data = pd.read_csv(path + "//dot.csv")
    day_list = sorted(data["class"].unique())
    
    y_data = [data[data["class"] == c]["value"].values for c in day_list]

    jitter = 1
    x_data = [np.array([20*i+10] * len(d)) for i, d in enumerate(y_data)]
    x_jittered = [x + st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]
    RED_DARK = "#f14454"
    BLACK = "#000000"
    # BODY = "#caadd8"
    BODY = "#eff4fb"
    EDGE = "#8ccbea"
    COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3", "#7570B3"]

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['grid.linewidth'] = 2.0
    plt.rcParams['xtick.labelsize'] = 20  # 设置x轴刻度标签的字体大小为12
    plt.rcParams['ytick.labelsize'] = 20  
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(12, 7))

    plt.axvline(x=10, color='black', linewidth=2,linestyle='-.')
    plt.axvline(x=30, color='black', linewidth=2,linestyle='-.')
    plt.axvline(x=50, color='black', linewidth=2,linestyle='-.')
    plt.axvline(x=70, color='black', linewidth=2,linestyle='-.')

    violins = ax.violinplot(
        y_data,
        positions=day_list,
        widths=8,
        bw_method="silverman",
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    
    for pc in violins["bodies"]:
        pc.set_facecolor(BODY)
        pc.set_edgecolor(EDGE)
        pc.set_linewidth(2)
        pc.set_alpha(1)

    
    for x, y, color in zip(x_jittered, y_data, COLOR_SCALE):
        ax.scatter(x, y, s=80, color="#2983b1", alpha=0.8)


    # mean
    ax.scatter(10,q2_list[6],s=100,color=RED_DARK,zorder=3,marker='s')
    ax.scatter(30,q2_list[26],s=100,color=RED_DARK,zorder=3,marker='s')
    ax.scatter(50,q2_list[46],s=100,color=RED_DARK,zorder=3,marker='s')
    ax.scatter(70,q2_list[66],s=100,color=RED_DARK,zorder=3,marker='s')

    ax.text(13,21.7,r"$\bar{Ct}=$" + str(round(q2_list[6],2)),fontsize=15)
    ax.plot([10,13],[q2_list[6],21.7],color='black',linewidth=2)

    ax.text(34,22,r"$\bar{Ct}=$" + str(round(q2_list[26],2)),fontsize=15)
    ax.plot([30,34],[q2_list[26],22],color='black',linewidth=2)

    ax.text(54,22.8,r"$\bar{Ct}=$" + str(round(q2_list[46],2)),fontsize=15)
    ax.plot([50,54],[q2_list[46],22.8],color='black',linewidth=2)

    ax.text(58,23.7,r"$\bar{Ct}=$" + str(round(q2_list[66],2)),fontsize=15)
    ax.plot([70,62],[q2_list[66],23.7],color='black',linewidth=2)


    # means = [y.mean() for y in y_data]
    # for i, mean in enumerate(means):
    #     # 添加表示均值的点
    #     ax.scatter(i, mean, s=250, color=RED_DARK, zorder=3)
    
    #     # 添加连接均值和标签的线
    #     ax.plot([i, i + 0.25], [mean, mean], ls="dashdot", color="black", zorder=3)
    
    #     # 添加均值标签
    #     ax.text(
    #         i + 0.25,
    #         mean,
    #         r"$\hat{\mu}_{\rm{mean}} = $" + str(round(mean, 2)),
    #         fontsize=13,
    #         va="center",
    #         bbox=dict(
    #             facecolor="white",
    #             edgecolor="black",
    #             boxstyle="round",
    #             pad=0.15
    #         ),
    #         zorder=10  # 确保线在顶部
    #     )
    x = [i for i in range(4,4+len(q1_list))]

    plt.text(1,19.3,r"Q1",color="#af8fd0",fontsize=15)
    plt.text(1,21,r"Q2",color="#af8fd0",fontsize=15)
    plt.text(1,22.8,r"Q3",color="#af8fd0",fontsize=15)

    plt.plot(x,q1_list,'#af8fd0',linewidth=3)
    plt.plot(x,q2_list,'#af8fd0',linewidth=3)
    plt.plot(x,q3_list,'#af8fd0',linewidth=3)
    plt.xlim(0,75)
    plt.ylim(16,32)
    plt.xlabel('Days since start of outbreak',fontsize=28,fontweight='bold')
    plt.ylabel('Ct value',fontsize=28,fontweight='bold')
    plt.show()




if __name__ == '__main__':
    Inum = 100
    path = "..\\results\\ER\\d=10R=3\\val"
    # day_list = [10,30,50,70]
    # num_list = [20,40,50,20]
    x = [i for i in range(77)]
    pro_list = read_info(path,Inum,x)
    # dot_list = generate(pro_list,num_list)
    # readcsv(path,dot_list,day_list)
    vio(path,pro_list)
    
