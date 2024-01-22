#include "simulation.h"
#include "agent.h"
#include "CtModel.h"
#include "network.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
using namespace std;



Simulation::Simulation(Network &net1){
    
    net = net1;
	R_GLOBAL=gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(R_GLOBAL,time(NULL));
    
}

Simulation::Simulation(){}

// void Simulation::initialNodes()
// {
//     for(int i = 0; i < net.nodesNum; i++)
//     {
//         int reco = gsl_ran_poisson(R_GLOBAL,Mu);
//         Agent a(reco);
//         a.initial(i);
//         nodes.push_back(a);
//     }
// }

void Simulation::initialSeeds()
{
    while(oldInfection.size()<kSeed){
        // int seed=Uniform(nodesNum);
        int seedID = gsl_rng_uniform(R_GLOBAL)*net.seedList.size();
        int seed = net.seedList[seedID];
        if(find(oldInfection.begin(),oldInfection.end(),seed)==oldInfection.end()){     //no repeat seed
            net.nodes[seed].infected=true;          //infected
            net.nodes[seed].timeInfected=0;
            oldInfection.push_back(seed);
            totalI ++;
        }
    }
    infectNodes.push_back(oldInfection);
    dailyInfection.push_back(oldInfection.size());
    Snum.push_back(net.nodesNum - oldInfection.size() - recover);
    Inum.push_back(oldInfection.size());
    Rnum.push_back(recover);
    totalInfection.push_back(totalI);
}

void Simulation::infect_HM(int t)
{
    for(int i=0;i<oldInfection.size();i++)
        {
            int infector=oldInfection[i];

            //if node is not isolated without individual contacts and is infectious
            if(net.nodes[infector].is_infectious(t) && net.nodes[infector].recoverd == false)
            {
                int m = gsl_ran_binomial(R_GLOBAL,Beta/(double)(net.nodesNum - 1),net.nodesNum - oldInfection.size() - recover);
                for(int j = 0; j < m; j ++)
                {
                    int infectee = gsl_rng_uniform(R_GLOBAL)*net.nodesNum;
                    if (net.nodes[infectee].infected == false && infector != infectee)
                    {
                        net.nodes[infectee].infected=true;
                        net.nodes[infectee].timeInfected=t;
                        net.nodes[infectee].infector=infector;
                        net.nodes[infector].infectNum ++;
                        newInfection.push_back(infectee);
                        totalI ++;
                        generateList.push_back(net.nodes[infectee].timeInfected - net.nodes[infector].timeInfected);
                    }
                }
            }
        }

        //outcome
        dailyInfection.push_back(newInfection.size());
        // dailyInfection[t]=newInfection.size();                  //num of new infection at day t
        infectNodes.push_back(newInfection);
}

void Simulation::recovery_HM(int t)
{
    for(int i=0;i<oldInfection.size();i++)
    {
        int infector=oldInfection[i];
        recover += net.nodes[infector].update(t);       
        if(!net.nodes[infector].recoverd)
        {
            newInfection.push_back(infector);
        }
    }
}

void Simulation::calculate_Rt()
{
    vector<int> infectorList;
    int infect_all = 0;
    double Rt = 0.0;

    for(int i = 0; i <= endTime; i ++)
    {
        infectorList = infectNodes[i];
        if(infectorList.size() == 0)
        {
            RtList.push_back(0.0);
        }
        else
        {
            for(int j = 0; j < infectorList.size(); j++)
            {
                int index = infectorList[j];
                infect_all += net.nodes[index].infectNum;
            }
            Rt = (double)infect_all / infectorList.size();
            RtList.push_back(Rt);
            infect_all = 0;
        }
    }
}

void Simulation::transmission_HM()
{
    /*-----------------initial---------------------------------*/
    
    int t = 0;

    //initial seeds
    initialSeeds();

    /*----------------transmission-------------------------*/
    while(oldInfection.size()>0)
    {
        t ++;
        
        infect_HM(t);

        //update oldInfection and newInfection
        recovery_HM(t);

        
        Inum.push_back(newInfection.size());
        Rnum.push_back(recover);
        Snum.push_back(net.nodesNum - newInfection.size() - recover);
        totalInfection.push_back(totalI);

        oldInfection.swap(newInfection);
        newInfection.clear(); 

    }
    endTime = t;

    calculate_Rt();

}


void Simulation::simuInfo(ofstream &f,int type)
{
    if(!f.is_open())
        cout<<"cannot open file_simuInfo"<<endl;

    if (type == 0)
    {
        for (int i = 0; i < Snum.size(); i ++)
        {
            f << Snum[i] <<' ';
        }
    }
    else if (type == 1)
    {
        for (int i = 0; i < Inum.size(); i ++)
        {
            f << Inum[i] <<' ';
        }
    }
    else if(type == 2)
    {
        for(int i = 0; i < Rnum.size(); i ++)
        {
            f << Rnum[i] << ' ';
        }
    }
    else if(type == 3)
    {
        for(int i = 0; i < totalInfection.size(); i ++)
        {
            f << totalInfection[i] << ' ';
        }
    }
    else if (type == 4)
    {
        for(int i = 0; i < dailyInfection.size(); i ++)
        {
            f << dailyInfection[i] << ' ';
        }
    }
    else if(type == 5)
    {
        for(int i = 0; i < RtList.size(); i ++)
        {
            f << RtList[i] << ' ';
        }
    }
    else if(type == 6)
    {
        for(int i = 0; i < generateList.size(); i ++)
        {
            f << generateList[i] << ' ';
        }
    }
    else if(type == 7)
    {
        for(int i = 0; i < loglikelihoodList.size(); i ++)
        {
            f << loglikelihoodList[i] << ' ';
        }
    }
    else if(type == 8)
    {
        f << Tstart;
    }
    f << endl;
}

long double Simulation::loglikelihood(int tt,double Rt,vector<double> Tg)
{
    long double loglike;
	double lambda;
	int s;
	double *omega2;
    // vector<double> omega2;
	double sumomega;
    int n = dailyInfection.size();
    bool flag;

	
	omega2 = (double *)calloc(n,sizeof(double));
	sumomega = 0.;
	for(s=1; s<=tt; s++)
	{
		sumomega += Tg[s]; 
   	}
  
  	for(s=1; s<=tt; s++)
  	{
    	omega2[s] = Tg[s]/sumomega;
  	}
  
  	lambda = 0.;
  	for(s=1; s<=tt; s++)
  	{
    	lambda += dailyInfection[tt-s]*omega2[s];
  	}
    if (tt == 21)
    {
        flag = true;
    }
  
  	// loglike = log(gsl_ran_poisson_pdf(data[tt],Rt*lambda)); 
    // long double result = Poisson2(dailyInfection[tt],Rt*lambda);
    long double result = gsl_ran_poisson_pdf(dailyInfection[tt],Rt*lambda);
    loglike = log(result); 
  	free(omega2);

  	return loglike;	
}

void Simulation::Baseline_Rt()
{
    // build generation time distribution
    int n_seq=1000;
    vector<double> Tg;
    for(int i = 0; i < n_seq; i ++)
    {
        Tg.push_back(0.0);
    }
    int gSize = generateList.size();
    for(int i = 0; i < gSize; i ++)
    {
        Tg[generateList[i]] += 1;
    }
    for(int i = 0; i < n_seq; i ++)
    {
        if(Tg[i] != 0)
        {
            Tg[i] /= gSize;
        }
    }

    int n = dailyInfection.size();

    //mark the time point of estiamting the Rt
    int t0;         //the first day of estimating
    int cum_case;
    cum_case=0;
    t0=0;
    int size = tau/2;

    while(cum_case<15)
    {
        cum_case+=dailyInfection[t0];
        t0+=1;
    }
    if(t0<10)
        t0=10;

    Tstart = t0;
    
    // double **R;
    // R=(double **) calloc(n,sizeof(double *));
    for(int i = 0; i < n; i++)
    {
        vector<double> base_Rt;
        for(int j = 0; j < nsim; j ++)
        {
            base_Rt.push_back(0.0);
        }
        // R[i] = (double *)calloc(nsim,sizeof(double));
        base_RtList.push_back(base_Rt);
    }

    long double loglike;
    long double tmploglike;
    double flag = false;

    for(int i = t0;i < n;i++)
    {
        int start,end;


        start = i - size;
        end = i - size + tau;
        if(start < 0)
        {
            start = 0;
            end = tau;
        }
        else if (i - size + tau > n)
        {
            end = n;
            start = n-tau;
        }

        if (i == t0)
        {
            flag = true;
        }
        else
        {
            flag = false;
        }

        if(dailyInfection[i]>0)
        {
            // R[i][0]=1.0;
            base_RtList[i][0] = 1.0;

            //loglike=loglikelihood(i, R[i][0], clean_caseSeq, tg, n);
            loglike=0;
            int tt;
            for(tt=start;tt<end;tt++)
            {
                loglike+=loglikelihood(tt,base_RtList[i][0],Tg);
            }
            if (flag)
            {
                loglikelihoodList.push_back(loglike);
            }
        }

        else
        {
            // R[i][0]=0.0;
            // base_Rt.push_back(0.0);
            base_RtList[i][0] = 0.0;
        }

        // loop over the mcmc iterations
        for(int e = 1; e < nsim; e++)
        {
            // cout << i <<' ' << e << endl;
            // R[i][e] = R[i][e-1];
            base_RtList[i][e] = base_RtList[i][e-1];
            if( dailyInfection[i]>0 )
            {
                double Rtmp = -1.;
                while(Rtmp < 0.0)
                {
                    // cout << normal(0.0,sigma) << endl;
                    // Rtmp = base_RtList[i][e-1] + normal_Rt(0.0,sigma);
                    Rtmp = base_RtList[i][e-1] + gsl_ran_gaussian(R_GLOBAL,sigma);

                }

                //Rtmp = R[i][e-1] + gsl_ran_gamma(R_GLOBAL,gamma_a,gamma_b);

                tmploglike = 0;

                //calculate the total likelyhood in the window [i-tau+1, i]
                int tt;
                for(tt=start;tt<end;tt++)
                {
                    tmploglike+=loglikelihood(tt,Rtmp,Tg);
                }

                // if(Prand() < min(tmploglike/loglike,1.0))
                // {
                //     loglike = tmploglike;
                //     base_RtList[i][e] = Rtmp;
                // }

                // if(Prand() < exp(tmploglike-loglike))
                // {
                //     loglike = tmploglike;
                //     base_RtList[i][e] = Rtmp;
                // }
                if(gsl_rng_uniform(R_GLOBAL) < exp(tmploglike-loglike))
                {
                    loglike = tmploglike;
                    base_RtList[i][e] = Rtmp;
                }
                if(flag)
                {
                    loglikelihoodList.push_back(loglike);
                }
            }
        }
    }

}

void Simulation::outRt(ofstream &f,int type)
{
    if(!f.is_open())
        cout<<"cannot open file_simuInfo"<<endl;
    
    // if(type == 0)
    // {
    //     vector<double> Rt_sim;
    //     cout << Tstart << endl;

    //     for(int i = Tstart; i < base_RtList.size(); i ++)
    //     {
    //         Rt_sim = base_RtList[i];
    //         int n = Rt_sim.size();
    //         for(int j = n - resim; j < Rt_sim.size(); j ++)
    //         {
    //             f << Rt_sim[j] << ' ';
    //         }
    //         f << endl;
    //     }
    // }
    // if(type == 1)
    // {
    //     int size = base_RtList.size();
    //     vector<double> Rt_sim = base_RtList[0];
    //     int n = Rt_sim.size();
    //     for(int i = n - resim; i < Rt_sim.size(); i ++)
    //     {
    //         for(int j = Tstart; j < size; j ++)
    //         {
    //             f << base_RtList[j][i] <<' ';
    //         }
    //         f << endl;
    //     }
    // }

    if(type == 0)
    {
        for(int i = 0; i < meanList.size(); i ++)
        {
            f << meanList[i] << ' ';
        }
    }
    else if (type == 1)
    {
        for(int i = 0; i < lowerList.size(); i ++)
        {
            f << lowerList[i] << ' ';
        }
    }
    else if (type == 2)
    {
        for(int i = 0; i < upperList.size(); i ++)
        {
            f << upperList[i] << ' ';
        }
    }
    f << endl;
    


}

vector<double> Simulation::calculate(vector<double> dataList)
{
    double mean = 0.0;
    double upper = 0.0;
    double lower = 0.0;
    vector<double> bound;
    vector<double> data;
    int n = 0;

    for(int i = 0; i < dataList.size(); i ++)
    {
        if(dataList[i] != 0.0)
        {
            data.push_back(dataList[i]);
        }
    }

    n = data.size();

    if(n > 0)
    {
        int size = int(n*0.025);
        for(int i = 0; i < n; i ++)
        {
            mean += data[i];
        }
        mean /= n;
        sort(data.begin(),data.end());
        lower = data[size];
        upper = data[n - size];
    }
    bound.push_back(mean);
    bound.push_back(lower);
    bound.push_back(upper);

    return bound;
    
}

vector<double> Simulation::smooth(vector<double> dataList)
{
    vector<double> result;
    double ave = 0.0;
    int n = dataList.size();
    int size = int(tau/2);

    for(int i = 0; i < n; i++)
    {
        ave = 0.0;
        if ((i - size) < 0)
        {
            for(int j = 0; j < tau; j ++)
            {
                ave += dataList[j];
            }
        }
        else if((i - size + tau) > n)
        {
            for(int j = n - tau; j < n; j++)
            {
                ave += dataList[j];
            }
        }
        else
        {
            for(int j = i - size; j < i - size + tau; j ++)
            {
                ave += dataList[j];
            }
        }
        ave /= tau;
        result.push_back(ave);
    }
    return result;

}

void Simulation::calculate_Baseline_Rt()
{
    vector<double> Rt_sim;
    vector<double> bound;
    vector<double> res;
    int day = base_RtList.size() - Tstart;

    // for(int i = Tstart; i < base_RtList.size(); i ++)
    // {
    //     Rt_sim = base_RtList[i];
    //     bound = calculate(Rt_sim);
    //     meanList.push_back(bound[0]);
    //     lowerList.push_back(bound[1]);
    //     upperList.push_back(bound[2]);
    //     day ++;
    // }

    vector<vector<double>> all;
    for(int i = 0; i < day; i ++)
    {
        vector<double> all_sim;
        for(int j = 0; j < resim; j ++)
        {
            all_sim.push_back(0.0);
        }
        all.push_back(all_sim);
    }

    int size = base_RtList.size();
    Rt_sim = base_RtList[0];
    int n = Rt_sim.size();
    for(int i = n - resim; i < n; i ++)
    {
        vector<double> Rt_sim1;
        for(int j = Tstart; j < size; j ++)
        {
            Rt_sim1.push_back(base_RtList[j][i]);
        }
        res = smooth(Rt_sim1);
        for(int z = 0; z < res.size(); z ++)
        {
            all[z][i- n + resim] = res[z];
        }
   
    }

    for(int i = 0; i < all.size(); i ++)
    {
        vector<double> all_sim = all[i];
        bound = calculate(all_sim);
        meanList.push_back(bound[0]);
        lowerList.push_back(bound[1]);
        upperList.push_back(bound[2]);

    }


}


void Simulation::Reset()
{
    oldInfection.clear();
    newInfection.clear();
    dailyInfection.clear();
    Snum.clear();
    Inum.clear();
    Rnum.clear();
    totalInfection.clear();
    infectNodes.clear();
    RtList.clear();
    generateList.clear();
    // ctMean.clear();
    // ctSkew.clear();
    loglikelihoodList.clear();
    base_RtList.clear();
    meanList.clear();
    lowerList.clear();
    upperList.clear();
    Tstart = 0;
    totalI = 0;
    recover = 0;
    // net.contact.clear();
    for(int i = 0; i < net.nodesNum; i ++)
    {
        net.nodes[i].reset();
        // net.contact.push_back(vector<int> ());
    }
}

Simulation_Ct::Simulation_Ct(Network &net1){
    
    // nodesNum = num;
    net = net1;
	R_GLOBAL=gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(R_GLOBAL,time(NULL));
    
}

Simulation_Ct::Simulation_Ct(){}

// void Simulation_Ct::initialNodes()
// {
//     for(int i = 0; i < nodesNum; i++)
//     {
//         int incu = gsl_ran_poisson(R_GLOBAL,sig);
//         int reco = gsl_ran_poisson(R_GLOBAL,Mu);
//         Agent_SEIR a(incu,reco);
//         a.initial(i);
//         nodes.push_back(a);
//     }
// }

// void Simulation_Ct::initialNodes_age()
// {
//     string file_age = "data\\vec_ShangHai.txt";
    
//     for(int i = 0; i < nodesNum; i++)
//     {
//         int incu = gsl_ran_poisson(R_GLOBAL,sig);
//         int reco = gsl_ran_poisson(R_GLOBAL,Mu);
//         Agent_SEIR a(incu,reco);
//         a.initial(i);
//         nodes.push_back(a);
//     }
// }

// void Simulation_Ct::initialSeeds()
// {
//     while(oldInfection.size()<kSeed){
//         // int seed=Uniform(nodesNum);
//         int seed = gsl_rng_uniform(R_GLOBAL)*net.nodesNum;
//         if(find(oldInfection.begin(),oldInfection.end(),seed)==oldInfection.end()){     //no repeat seed
//             net.nodes[seed].infected=true;          //infected
//             net.nodes[seed].timeInfected=0;
//             oldInfection.push_back(seed);
//             totalI ++;
//         }
//     }
//     infectNodes.push_back(oldInfection);
//     dailyInfection.push_back(oldInfection.size());
//     Snum.push_back(net.nodesNum - oldInfection.size() - recover);
//     Inum.push_back(oldInfection.size());
//     Rnum.push_back(recover);
//     totalInfection.push_back(totalI);
// }

void Simulation_Ct::infect_HM(int t)
{
    for(int i=0;i<oldInfection.size();i++)
    {
        int infector=oldInfection[i];

        //if node is not isolated without individual contacts and is infectious
        if(net.nodes[infector].is_infectious(t) && net.nodes[infector].recoverd == false)
        {
            int m = gsl_ran_binomial(R_GLOBAL,Beta/(double)(net.nodesNum - 1),net.nodesNum - oldInfection.size() - recover);
            for(int j = 0; j < m; j ++)
            {
                // int infectee = Uniform(nodesNum);
                int infectee = gsl_rng_uniform(R_GLOBAL)*net.nodesNum;
                if (net.nodes[infectee].infected == false && infector != infectee)
                {
                    net.nodes[infectee].infected=true;
                    net.nodes[infectee].timeInfected=t;
                    net.nodes[infectee].infector=infector;
                    net.nodes[infectee].vlList = getCtValue(net.nodes[infectee].incubation);
                    net.nodes[infector].infectNum ++;
                    newInfection.push_back(infectee);
                    totalI ++;
                    generateList.push_back(net.nodes[infectee].timeInfected - net.nodes[infector].timeInfected);
                }
            }
        }
    }

    //outcome
    dailyInfection.push_back(newInfection.size());
    infectNodes.push_back(newInfection);
}


// void Simulation_Ct::recovery_HM(int t)
// {
//     for(int i=0;i<oldInfection.size();i++)
//     {
//         int infector=oldInfection[i];
//         recover += net.nodes[infector].update(t);       
//         if(!net.nodes[infector].recoverd)
//         {
//             newInfection.push_back(infector);
//         }
//     }
// }

// void Simulation_Ct::calculate_Rt()
// {
//     vector<int> infectorList;
//     int infect_all = 0;
//     double Rt = 0.0;

//     for(int i = 0; i <= endTime; i ++)
//     {
//         infectorList = infectNodes[i];
//         if(infectorList.size() == 0)
//         {
//             RtList.push_back(0.0);
//         }
//         else
//         {
//             for(int j = 0; j < infectorList.size(); j++)
//             {
//                 int index = infectorList[j];
//                 infect_all += net.nodes[index].infectNum;
//             }
//             Rt = (double)infect_all / infectorList.size();
//             RtList.push_back(Rt);
//             infect_all = 0;
//         }
//     }
// }

void Simulation_Ct::transmission_HM()
{
    /*-----------------initial---------------------------------*/
    
    // vector<int> oldInfection,newInfection;          //record infection
    int t = 0;

    //initial seeds
    initialSeeds();
    ctMean.push_back(40.0);
    ctSkew.push_back(0);

    for(int i = 16; i < 40; i ++)
    {
        distrb.push_back(0.0);
    }

    distrb.push_back(1.0);



    /*----------------transmission-------------------------*/
    while(oldInfection.size()>0)
    {
        t ++;
        // if(t < 70)
        // {
        //     Beta = 2.0/4;
        // }
        // else if(t >= 70 && t < 100)
        // {
        //     Beta = 0.3/4;
        // }
        // else if(t >= 100 && t < 150)
        // {
        //     Beta = 1.9/4;
        // }
        // else if(t >= 150)
        // {
        //     Beta = 0.3/4;
        // }
        
        infect_HM(t);

        //update oldInfection and newInfection
        recovery_HM(t);

        // vector<double> ctList;
        // vector<vector<double>> total;
        double mean = 0.0;
        double skew = 0.0;

        vector<vector<double>> total = CtTest(t);
        vector<double> dur_time = total[0];
        vector<double> ctList = total[1];
        mean = calculate_mean(ctList);
        skew = calculate_skewness(ctList,mean);
        ctMean.push_back(mean);
        ctSkew.push_back(skew);
        // calculate_distrb(ctList,dur_time,);


        
        Inum.push_back(newInfection.size());
        Rnum.push_back(recover);
        Snum.push_back(net.nodesNum - newInfection.size() - recover);
        totalInfection.push_back(totalI);

        oldInfection.swap(newInfection);
        newInfection.clear(); 

    }
    endTime = t;

    calculate_Rt();

}

void Simulation_Ct::transmission_net(string path,int printct)
{
    /*-----------------initial---------------------------------*/
    
    // vector<int> oldInfection,newInfection;          //record infection
    vector<vector<int>> contact = net.contact;
    int t = 0;

    //initial seeds
    initialSeeds();
    ctMean.push_back(40.0);
    ctSkew.push_back(0);

    for(int i = 16; i < 40; i ++)
    {
        distrb.push_back(0.0);
    }

    distrb.push_back(1.0);

    generation.push_back(1.0);

    for(int i = 1; i <= 24; i ++)
    {
        generation.push_back(0.0);
    }

    ofstream csvCt;
    ofstream csvdailyNum;
    ofstream csvGeneration;

    if (printct != 0)
    {
        csvCt.open(path + "\\ctData\\ctDataID" + to_string(printct) + ".csv");
        csvCt << "t,ct" << endl;

        csvdailyNum.open(path + "\\ctData\\dailyNumID" + to_string(printct) + ".csv");
        csvdailyNum << "date,imported,local" << endl;
        csvdailyNum << "0,1,0" << endl;

        csvGeneration.open(path + "\\ctData\\GenerationID" + to_string(printct) + ".csv");
        csvGeneration << "EL,ER,SL,SR,type" << endl;

    }
    /*----------------transmission-------------------------*/
    while(oldInfection.size()>0)
    {
        t ++;
        
        for(int i=0;i<oldInfection.size();i++)
        {
            int infector=oldInfection[i];

            //if node is not isolated without individual contacts and is infectious
            if(net.nodes[infector].is_infectious(t) && net.nodes[infector].recoverd == false)
            {
                vector<int> member = contact[infector];
                int size = member.size();
                int m = 0;
                if(size != 0)
                {
                    if(size < 20)
                    {
                        m=gsl_ran_binomial(R_GLOBAL,Beta,size);
                    }
                    else
                    {
                        m=gsl_ran_poisson(R_GLOBAL,Beta*size);
                    }
                }
                for(int j = 0; j < m; j ++)
                {
                    int index = gsl_rng_uniform(R_GLOBAL)*size;
                    int infectee = member[index];
                    if (net.nodes[infectee].infected == false && infector != infectee)
                    {
                        net.nodes[infectee].infected=true;
                        net.nodes[infectee].timeInfected=t;
                        net.nodes[infectee].infector=infector;
                        net.nodes[infectee].vlList = getCtValue(net.nodes[infectee].incubation);
                        net.nodes[infectee].general = net.nodes[infectee].timeInfected - net.nodes[infector].timeInfected;
                        net.nodes[infector].infectNum ++;
                        newInfection.push_back(infectee);
                        totalI ++;
                        generateList.push_back(net.nodes[infectee].timeInfected - net.nodes[infector].timeInfected);
                    }
                }
            }
        }

        //outcome
        dailyInfection.push_back(newInfection.size());
        infectNodes.push_back(newInfection);

        //update oldInfection and newInfection
        recovery_HM(t);

        // vector<double> ctList;
        vector<vector<double>> total;
        double mean = 0.0;
        double skew = 0.0;
        double mean_onset = 0.0;
        double skew_onset = 0.0;

        total = CtTest(t);
        vector<double> dur_time = total[0];
        vector<double> ctList = total[1];
        vector<double> ctList_onset = total[2];
        mean = calculate_mean(ctList);
        skew = calculate_skewness(ctList,mean);
        ctMean.push_back(mean);
        ctSkew.push_back(skew);

        mean_onset = calculate_mean(ctList_onset);
        skew_onset = calculate_skewness(ctList_onset,mean_onset);
        ctMean_onset.push_back(mean_onset);
        ctSkew_onset.push_back(skew_onset);

        calculate_distrb(ctList,dur_time,ctList_onset);

        if (printct != 0 && t > 100 && t % 50 == 0)
        {
            for (int id = 0; id < net.nodesNum; id ++)
            {
                if (net.nodes[id].infected == false || net.nodes[id].recoverd == true)
                {
                    csvCt << to_string(t) << ",40" << endl;
                }
                else
                {
                    int deltaT = t - net.nodes[id].timeInfected;
                    csvCt << to_string(t) << "," << to_string(net.nodes[id].vlList[deltaT]) << endl;
                }
            }
        }

        if (printct != 0)
        {
            csvdailyNum << to_string(t) << ",0," << dailyInfection[t] << endl;
            vector<int> nodeList = infectNodes[t];
            for(int i = 0; i < nodeList.size(); i ++)
            {
                int id = nodeList[i];
                int infector = net.nodes[id].infector;
                int Elr = net.nodes[infector].timeInfected;
                int Slr = net.nodes[id].timeInfected;
                csvGeneration << to_string(Elr) << "," << to_string(Elr) << "," << to_string(Slr) << "," << to_string(Slr) << ",2" << endl;

            }
        }
        
        Inum.push_back(newInfection.size());
        Rnum.push_back(recover);
        Snum.push_back(net.nodesNum - newInfection.size() - recover);
        totalInfection.push_back(totalI);

        oldInfection.swap(newInfection);
        newInfection.clear(); 

    }
    endTime = t;

    calculate_Rt();
    csvCt.close();
    csvdailyNum.close();
    csvGeneration.close();

}

vector<vector<double>> Simulation_Ct::CtTest(int t)
{
    vector<vector<double>> total;
    vector<double> dur_time;
    vector<double> ctList;
    vector<double> ctList_onset;

    for(int i = 0; i < newInfection.size(); i ++)
    {
        int index = newInfection[i];
        int timed = t - net.nodes[index].timeInfected;
        dur_time.push_back(timed);

        int time1 = net.nodes[index].vlList.size();
        vector<double> vlList = net.nodes[index].vlList;
        if(timed >= time1)
        {
                // cout << time <<" " << time1 << endl;
            ctList.push_back(40.0);

        }
        else
        {
            ctList.push_back(net.nodes[index].vlList[timed]);
        }


        if(net.nodes[index].symptomatic == true)
        {
            int time = t - net.nodes[index].timeInfected;
            // dur_time.push_back(time);
            int time1 = net.nodes[index].vlList.size();
            vector<double> vlList = net.nodes[index].vlList;
            if(time >= time1)
            {
                    // cout << time <<" " << time1 << endl;
                ctList_onset.push_back(40.0);

            }
            else
            {
                ctList_onset.push_back(net.nodes[index].vlList[time]);
            }

        }
        
    }
    total.push_back(dur_time);
    total.push_back(ctList);
    total.push_back(ctList_onset);
    return total;
}

void Simulation_Ct::csvCt()
{
    
}

void Simulation_Ct::calculate_distrb(vector<double> ctList,vector<double> dur_time,vector<double> ctList_onset)
{
    vector<double> pcount;
    for(int i = 16; i <= 40; i ++)
    {
        pcount.push_back(0.0);
    }
    int n = ctList.size();
    if(n > 0)
    {
        for(int i = 0; i < n; i ++)
        {
            int ct = ctList[i];
            if(ct <= 16)
            {
                pcount[0] ++;
            }
            else
            {
                pcount[ct - 16] ++;
            }

        }
        for(int i = 16; i <= 40; i ++)
        {
            pcount[i - 16] /= n;
        }
        
    }
    for(int i = 16; i <= 40; i ++)
    {
        distrb.push_back(pcount[i - 16]);
    }
    

    vector<double> tcount;
    for(int i = 0; i <= 24; i ++)
    {
        tcount.push_back(0.0);
    }
    int n1 = dur_time.size();
    if(n1 > 0)
    {
        for(int i = 0; i < n; i ++)
        {
            int time_d = dur_time[i];
            if(time_d >= 24)
            {
                tcount[24] ++;
            }
            else
            {
                tcount[time_d] ++;
            }

        }
        for(int i = 0; i <= 24; i ++)
        {
            tcount[i] /= n1;
        }
        
    }
    for(int i = 0; i <= 24; i ++)
    {
        generation.push_back(tcount[i]);
    }

    vector<double> ocount;
    for(int i = 16; i <= 40; i ++)
    {
        ocount.push_back(0.0);
    }
    int n2 = ctList_onset.size();
    if(n2 > 0)
    {
        for(int i = 0; i < n2; i ++)
        {
            int ct = ctList_onset[i];
            if(ct <= 16)
            {
                ocount[0] ++;
            }
            else
            {
                ocount[ct - 16] ++;
            }

        }
        for(int i = 16; i <= 40; i ++)
        {
            ocount[i - 16] /= n;
        }
        
    }
    for(int i = 16; i <= 40; i ++)
    {
        distrb_onset.push_back(ocount[i - 16]);
    }
}

void Simulation_Ct::simuInfo(ofstream &f,int type)
{
    if(!f.is_open())
        cout <<"cannot open file_simuInfo"<<endl;

    if (type == 0)
    {
        for (int i = 0; i < Snum.size(); i ++)
        {
            f << Snum[i] <<' ';
        }
    }
    else if (type == 1)
    {
        for (int i = 0; i < Inum.size(); i ++)
        {
            f << Inum[i] <<' ';
        }
    }
    else if(type == 2)
    {
        for(int i = 0; i < Rnum.size(); i ++)
        {
            f << Rnum[i] << ' ';
        }
    }
    else if(type == 3)
    {
        for(int i = 0; i < totalInfection.size(); i ++)
        {
            f << totalInfection[i] << ' ';
        }
    }
    else if (type == 4)
    {
        for(int i = 0; i < dailyInfection.size(); i ++)
        {
            f << dailyInfection[i] << ' ';
        }
    }
    else if(type == 5)
    {
        for(int i = 0; i < RtList.size(); i ++)
        {
            f << RtList[i] << ' ';
        }
    }
    else if(type == 6)
    {
        for(int i = 0; i < ctMean.size(); i ++)
        {
            f << ctMean[i] << ' ';
        }

    }
    else if(type == 7)
    {
        for(int i = 0; i < ctSkew.size(); i ++)
        {
            f << ctSkew[i] << ' ';
        }
    }
    else if(type == 8)
    {
        for(int i = 0; i < generateList.size(); i ++)
        {
            f << generateList[i] << ' ';
        }
    }
    else if(type == 9)
    {
        for(int i = 0; i < loglikelihoodList.size(); i ++)
        {
            f << loglikelihoodList[i] << ' ';
        }
    }
    else if(type == 10)
    {
        for(int i = 0; i < distrb.size(); i ++)
        {
            f << distrb[i] << ' ';
        }
    }
    else if(type == 12)
    {
        for(int i = 0; i < generation.size(); i ++)
        {
            f << generation[i] << ' ';
        }
    }
    else if(type == 13)
    {
        for(int i = 0; i < distrb_onset.size(); i ++)
        {
            f << distrb_onset[i] << ' ';
        }
    }
    else if(type == 14)
    {
        for(int i = 0; i < ctMean_onset.size(); i ++)
        {
            f << ctMean_onset[i] << ' ';
        }
    }
    else if(type == 15)
    {
        for(int i = 0; i < ctSkew_onset.size(); i ++)
        {
            f << ctSkew_onset[i] << ' ';
        }
    }
    else if(type == 11)
    {
        f << Tstart;
    }
    
    f << endl;
}


void Simulation_Ct::Reset()
{
    oldInfection.clear();
    newInfection.clear();
    dailyInfection.clear();
    Snum.clear();
    Inum.clear();
    Rnum.clear();
    totalInfection.clear();
    infectNodes.clear();
    RtList.clear();
    generateList.clear();
    // ctMean.clear();
    // ctSkew.clear();
    loglikelihoodList.clear();
    base_RtList.clear();

    ctMean.clear();
    ctSkew.clear();
    ctMean_onset.clear();
    ctSkew_onset.clear();
    distrb.clear();
    distrb_onset.clear();
    generation.clear();

    meanList.clear();
    lowerList.clear();
    upperList.clear();
    Tstart = 0;
    
    totalI = 0;
    recover = 0;
    // net.contact.clear();
    for(int i = 0; i < net.nodesNum; i ++)
    {
        net.nodes[i].reset();
        // net.contact.push_back(vector<int> ());
    }
}



Simulation_Ct_test::Simulation_Ct_test(Network &net1){
    
    net = net1;
	R_GLOBAL=gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(R_GLOBAL,time(NULL));
    
}


Simulation_Ct_test::Simulation_Ct_test(){}



// void Simulation_Ct_test::transmission_HM()
// {
//     /*-----------------initial---------------------------------*/
    
//     // vector<int> oldInfection,newInfection;          //record infection
//     int t = 0;

//     //initial seeds
//     initialSeeds();

//     ctMean.push_back(40.0);
//     ctSkew.push_back(0);

//     testNum1.push_back(kSeed);
//     ctMean1.push_back(40.0);
//     ctSkew1.push_back(0);

//     testNum2.push_back(kSeed);
//     ctMean2.push_back(40.0);
//     ctSkew2.push_back(0);

//     testNum3.push_back(kSeed);
//     ctMean3.push_back(40.0);
//     ctSkew3.push_back(0);

//     testNum4.push_back(kSeed);
//     ctMean4.push_back(40.0);
//     ctSkew4.push_back(0);

//     for(int i = 0; i < 5; i ++)
//     {
//         distrb.push_back(0.0);
//         distrb1.push_back(0.0);
//         distrb2.push_back(0.0);
//         distrb3.push_back(0.0);
//         distrb4.push_back(0.0);
//     }

//     distrb.push_back(1.0);
//     distrb1.push_back(1.0);
//     distrb2.push_back(1.0);
//     distrb3.push_back(1.0);
//     distrb4.push_back(1.0);



//     /*----------------transmission-------------------------*/
//     while(oldInfection.size()>0)
//     {
//         t ++;

//         if(t < 70)
//         {
//             Beta = 2.0/4;
//         }
//         else if(t >= 70 && t < 100)
//         {
//             Beta = 0.3/4;
//         }
//         else if(t >= 100 && t < 150)
//         {
//             Beta = 1.9/4;
//         }
//         else if(t >= 150)
//         {
//             Beta = 0.3/4;
//         }
        
//         infect_HM(t);

//         //update oldInfection and newInfection
//         recovery_HM(t);

//         vector<vector<double>> total;
//         double mean = 0.0;
//         double skew = 0.0;

//         total = CtTest(t);
//         vector<double> dur_time = total[0];
//         vector<double> ctList = total[1];
//         mean = calculate_mean(ctList);
//         skew = calculate_skewness(ctList,mean);
//         ctMean.push_back(mean);
//         ctSkew.push_back(skew);
//         // calculate_distrb(ctList,dur_time);
        
//         ctList = CtTest_type(1,t,csvD);

//         mean = calculate_mean(ctList);
//         ctMean1.push_back(mean);
//         skew = calculate_skewness(ctList,mean);
//         ctSkew1.push_back(skew);
//         calculate_distrb_type(ctList,1);

//         ctList = CtTest_type(2,t);

//         mean = calculate_mean(ctList);
//         ctMean2.push_back(mean);
//         skew = calculate_skewness(ctList,mean);
//         ctSkew2.push_back(skew);
//         calculate_distrb_type(ctList,2);

//         ctList = CtTest_type(3,t);

//          mean = calculate_mean(ctList);
//         ctMean3.push_back(mean);
//         skew = calculate_skewness(ctList,mean);
//         ctSkew3.push_back(skew);
//         calculate_distrb_type(ctList,3);

//         ctList = CtTest_type(4,t);

//         mean = calculate_mean(ctList);
//         ctMean4.push_back(mean);
//         skew = calculate_skewness(ctList,mean);
//         ctSkew4.push_back(skew);
//         calculate_distrb_type(ctList,4);


//         Inum.push_back(newInfection.size());
//         Rnum.push_back(recover);
//         Snum.push_back(net.nodesNum - newInfection.size() - recover);
//         totalInfection.push_back(totalI);

//         oldInfection.swap(newInfection);
//         newInfection.clear(); 

//     }
//     endTime = t;

//     calculate_Rt();

// }

void Simulation_Ct_test::transmission_net(string path,int printct)
{
    /*-----------------initial---------------------------------*/
    
    // vector<int> oldInfection,newInfection;          //record infection
    vector<vector<int>> contact = net.contact;
    int t = 0;

    //initial seeds
    initialSeeds();
    ctMean.push_back(40.0);
    ctSkew.push_back(0);

    testNum1.push_back(kSeed);
    ctMean1.push_back(40.0);
    ctSkew1.push_back(0);

    testNum2.push_back(kSeed);
    ctMean2.push_back(40.0);
    ctSkew2.push_back(0);

    testNum3.push_back(kSeed);
    ctMean3.push_back(40.0);
    ctSkew3.push_back(0);

    testNum4.push_back(kSeed);
    ctMean4.push_back(40.0);
    ctSkew4.push_back(0);

    for(int i = 16; i < 40; i ++)
    {
        distrb.push_back(0.0);
        distrb1.push_back(0.0);
        distrb2.push_back(0.0);
        distrb3.push_back(0.0);
        distrb4.push_back(0.0);
    }

    distrb.push_back(1.0);
    distrb1.push_back(1.0);
    distrb2.push_back(1.0);
    distrb3.push_back(1.0);
    distrb4.push_back(1.0);

    generation.push_back(1.0);

    for(int i = 1; i <= 24; i ++)
    {
        generation.push_back(0.0);
    }

    ofstream csvdailyNum;
    ofstream csvGeneration;

    ofstream csvdailyNum1;
    ofstream csvGeneration1;

    ofstream csvdailyNum2;
    ofstream csvGeneration2;

    ofstream csvdailyNum3;
    ofstream csvGeneration3;

    ofstream csvdailyNum4;
    ofstream csvGeneration4;

    if (printct != 0)
    {

        csvdailyNum.open(path + "\\ctData\\dailyNumID" + to_string(printct) + ".csv");
        csvdailyNum << "date,imported,local" << endl;
        csvdailyNum << "0,1,0" << endl;

        csvGeneration.open(path + "\\ctData\\GenerationID" + to_string(printct) + ".csv");
        csvGeneration << "EL,ER,SL,SR,type" << endl;

        csvdailyNum1.open(path + "\\ctData\\dailyNumID1-" + to_string(printct) + ".csv");
        csvdailyNum1 << "date,imported,local" << endl;
        csvdailyNum1 << "0,1,0" << endl;

        csvGeneration1.open(path + "\\ctData\\GenerationID1-" + to_string(printct) + ".csv");
        csvGeneration1 << "EL,ER,SL,SR,type" << endl;

        csvdailyNum2.open(path + "\\ctData\\dailyNumID2-" + to_string(printct) + ".csv");
        csvdailyNum2 << "date,imported,local" << endl;
        csvdailyNum2 << "0,1,0" << endl;

        csvGeneration2.open(path + "\\ctData\\GenerationID2-" + to_string(printct) + ".csv");
        csvGeneration2 << "EL,ER,SL,SR,type" << endl;

        csvdailyNum3.open(path + "\\ctData\\dailyNumID3-" + to_string(printct) + ".csv");
        csvdailyNum3 << "date,imported,local" << endl;
        csvdailyNum3 << "0,1,0" << endl;

        csvGeneration3.open(path + "\\ctData\\GenerationID3-" + to_string(printct) + ".csv");
        csvGeneration3 << "EL,ER,SL,SR,type" << endl;

        csvdailyNum4.open(path + "\\ctData\\dailyNumID4-" + to_string(printct) + ".csv");
        csvdailyNum4 << "date,imported,local" << endl;
        csvdailyNum4 << "0,1,0" << endl;

        csvGeneration4.open(path + "\\ctData\\GenerationID4-" + to_string(printct) + ".csv");
        csvGeneration4 << "EL,ER,SL,SR,type" << endl;

    }

    while(oldInfection.size()>0)
    {
        t ++;
        
        for(int i=0;i<oldInfection.size();i++)
        {
            int infector=oldInfection[i];

            //if node is not isolated without individual contacts and is infectious
            if(net.nodes[infector].is_infectious(t) && net.nodes[infector].recoverd == false)
            {
                vector<int> member = contact[infector];
                int size = member.size();
                int m = 0;
                if(size != 0)
                {
                    if(size < 20)
                    {
                        m=gsl_ran_binomial(R_GLOBAL,Beta,size);
                    }
                    else
                    {
                        m=gsl_ran_poisson(R_GLOBAL,Beta*size);
                    }
                }
                for(int j = 0; j < m; j ++)
                {
                    int index = gsl_rng_uniform(R_GLOBAL)*size;
                    int infectee = member[index];
                    if (net.nodes[infectee].infected == false && infector != infectee)
                    {
                        net.nodes[infectee].infected=true;
                        net.nodes[infectee].timeInfected=t;
                        net.nodes[infectee].infector=infector;
                        net.nodes[infectee].vlList = getCtValue(net.nodes[infectee].incubation);
                        net.nodes[infectee].general = net.nodes[infectee].timeInfected - net.nodes[infector].timeInfected;
                        net.nodes[infector].infectNum ++;
                        newInfection.push_back(infectee);
                        totalI ++;
                        generateList.push_back(net.nodes[infectee].timeInfected - net.nodes[infector].timeInfected);
                    }
                }
            }
        }

        //outcome
        dailyInfection.push_back(newInfection.size());
        infectNodes.push_back(newInfection);

        //update oldInfection and newInfection
        recovery_HM(t);

        // vector<double> ctList;
        vector<vector<double>> total;
        double mean = 0.0;
        double skew = 0.0;
        double mean_onset = 0.0;
        double skew_onset = 0.0;

        total = CtTest(t);
        vector<double> dur_time = total[0];
        vector<double> ctList = total[1];
        vector<double> ctList_onset = total[2];
        mean = calculate_mean(ctList);
        skew = calculate_skewness(ctList,mean);
        ctMean.push_back(mean);
        ctSkew.push_back(skew);

        calculate_distrb(ctList,dur_time,ctList_onset);

        ctList = CtTest_type(1,t,csvdailyNum1,csvGeneration1);
        mean = calculate_mean(ctList);
        ctMean1.push_back(mean);
        skew = calculate_skewness(ctList,mean);
        ctSkew1.push_back(skew);
        calculate_distrb_type(ctList,1);

        ctList = CtTest_type(2,t,csvdailyNum2,csvGeneration2);
        mean = calculate_mean(ctList);
        ctMean2.push_back(mean);
        skew = calculate_skewness(ctList,mean);
        ctSkew2.push_back(skew);
        calculate_distrb_type(ctList,2);

        ctList = CtTest_type(3,t,csvdailyNum3,csvGeneration3);
        mean = calculate_mean(ctList);
        ctMean3.push_back(mean);
        skew = calculate_skewness(ctList,mean);
        ctSkew3.push_back(skew);
        calculate_distrb_type(ctList,3);

        ctList = CtTest_type(4,t,csvdailyNum4,csvGeneration4);
        mean = calculate_mean(ctList);
        ctMean4.push_back(mean);
        skew = calculate_skewness(ctList,mean);
        ctSkew4.push_back(skew);
        calculate_distrb_type(ctList,4);

        if (printct != 0)
        {
            csvdailyNum << to_string(t) << ",0," << dailyInfection[t] << endl;
            vector<int> nodeList = infectNodes[t];

            for(int i = 0; i < nodeList.size(); i ++)
            {
                int id = nodeList[i];
                int infector = net.nodes[id].infector;
                int Elr = net.nodes[infector].timeInfected;
                int Slr = net.nodes[id].timeInfected;
                csvGeneration << to_string(Elr) << "," << to_string(Elr) << "," << to_string(Slr) << "," << to_string(Slr) << ",2" << endl;

            }
        }


        
        Inum.push_back(newInfection.size());
        Rnum.push_back(recover);
        Snum.push_back(net.nodesNum - newInfection.size() - recover);
        totalInfection.push_back(totalI);

        oldInfection.swap(newInfection);
        newInfection.clear(); 

    }
    endTime = t;

    calculate_Rt();
    csvdailyNum.close();
    csvGeneration.close();

    csvdailyNum1.close();
    csvGeneration1.close();

    csvdailyNum2.close();
    csvGeneration2.close();

    csvdailyNum3.close();
    csvGeneration3.close();

    csvdailyNum4.close();
    csvGeneration4.close();
}

vector<double> Simulation_Ct_test::CtTest_type(int type,int t,ofstream& daily,ofstream& generation)
{
    vector<double> ctList;
    int count = 0;
    int count1 = 0;
    vector<int> nodeList = infectNodes[t];
    
    if(type == 1)
    {
        for(int i = 0; i < newInfection.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < 0.25)
            {
                int index = newInfection[i];
                int time = t - net.nodes[index].timeInfected;
                int time1 = net.nodes[index].vlList.size();

                vector<double> vlList = net.nodes[index].vlList;
                if(time >= time1)
                {
                    // cout << time <<" " << time1 << endl;
                    ctList.push_back(40.0);

                }
                else
                {
                    ctList.push_back(net.nodes[index].vlList[time]);
                }
                count ++;
            }

        }
        for(int i = 0; i < nodeList.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < 0.25)
            {
                int index = newInfection[i];
                int infector = net.nodes[index].infector;
                int Elr = net.nodes[infector].timeInfected;
                int Slr = net.nodes[index].timeInfected;

                generation << to_string(Elr) << "," << to_string(Elr) << "," << to_string(Slr) << "," << to_string(Slr) << ",2" << endl;
                count1 ++;

            }

        }
        testNum1.push_back(count);
        daily << to_string(t) << ",0," << to_string(count1) << endl;
    }
    else if(type == 2)
    {
        for(int i = 0; i < newInfection.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < 0.1)
            {
                int index = newInfection[i];
                int time = t - net.nodes[index].timeInfected;
                int time1 = net.nodes[index].vlList.size();

                vector<double> vlList = net.nodes[index].vlList;
                if(time >= time1)
                {
                    // cout << time <<" " << time1 << endl;
                    ctList.push_back(40.0);

                }
                else
                {
                    ctList.push_back(net.nodes[index].vlList[time]);
                }
                count ++;
            }

        }
        for(int i = 0; i < nodeList.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < 0.1)
            {
                int index = newInfection[i];
                int infector = net.nodes[index].infector;
                int Elr = net.nodes[infector].timeInfected;
                int Slr = net.nodes[index].timeInfected;

                generation << to_string(Elr) << "," << to_string(Elr) << "," << to_string(Slr) << "," << to_string(Slr) << ",2" << endl;
                count1 ++;

            }

        }
        testNum2.push_back(count);
        daily << to_string(t) << ",0," << to_string(count1) << endl;
    }
    else if(type == 3)
    {
        double p = 0.0;
        double k = 0.45/90;
        p = k*t + 0.15;
        if(p >= 0.6)
        {
            p = 0.6;
        }
        
        for(int i = 0; i < newInfection.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < p)
            {
                int index = newInfection[i];
                int time = t - net.nodes[index].timeInfected;
                int time1 = net.nodes[index].vlList.size();

                vector<double> vlList = net.nodes[index].vlList;
                if(time >= time1)
                {
                    // cout << time <<" " << time1 << endl;
                    ctList.push_back(40.0);

                }
                else
                {
                    ctList.push_back(net.nodes[index].vlList[time]);
                }
                count ++;
            }

        }
        for(int i = 0; i < nodeList.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < p)
            {
                int index = newInfection[i];
                int infector = net.nodes[index].infector;
                int Elr = net.nodes[infector].timeInfected;
                int Slr = net.nodes[index].timeInfected;

                generation << to_string(Elr) << "," << to_string(Elr) << "," << to_string(Slr) << "," << to_string(Slr) << ",2" << endl;
                count1 ++;

            }

        }
        testNum3.push_back(count);
        daily << to_string(t) << ",0," << to_string(count1) << endl;
    }
    else if(type == 4)
    {
        double p = 0.0;
        if(t <= 20)
        {
            p = 0.05;
        }
        else
        {
            p = 0.25;
        }
        
        for(int i = 0; i < newInfection.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < p)
            {
                int index = newInfection[i];
                int time = t - net.nodes[index].timeInfected;
                int time1 = net.nodes[index].vlList.size();

                vector<double> vlList = net.nodes[index].vlList;
                if(time >= time1)
                {
                    // cout << time <<" " << time1 << endl;
                    ctList.push_back(40.0);

                }
                else
                {
                    ctList.push_back(net.nodes[index].vlList[time]);
                }
                count ++;
            }

        }
        for(int i = 0; i < nodeList.size(); i ++)
        {
            double p1 = gsl_rng_uniform(R_GLOBAL);
            if(p1 < p)
            {
                int index = newInfection[i];
                int infector = net.nodes[index].infector;
                int Elr = net.nodes[infector].timeInfected;
                int Slr = net.nodes[index].timeInfected;

                generation << to_string(Elr) << "," << to_string(Elr) << "," << to_string(Slr) << "," << to_string(Slr) << ",2" << endl;
                count1 ++;

            }

        }
        testNum4.push_back(count);
        daily << to_string(t) << ",0," << to_string(count1) << endl;
    }
    return ctList;
    
}

void Simulation_Ct_test::calculate_distrb_type(vector<double> ctList,int type)
{
    vector<double> pcount;
    for(int i = 16; i <= 40; i ++)
    {
        pcount.push_back(0.0);
    }
    int n = ctList.size();
    if(n > 0)
    {
        for(int i = 0; i < n; i ++)
        {
            int ct = ctList[i];
            if(ct <= 16)
            {
                pcount[0] ++;
            }
            else
            {
                pcount[ct - 16] ++;
            }

        }
        for(int i = 16; i <= 40; i ++)
        {
            pcount[i - 16] /= n;
        }
        
    }
    if (type == 1)
    {
        for(int i = 16; i <= 40; i ++)
        {
            distrb1.push_back(pcount[i - 16]);
        }
    }
    else if (type == 2)
    {
        for(int i = 16; i <= 40; i ++)
        {
            distrb2.push_back(pcount[i - 16]);
        }
    }
    else if (type == 3)
    {
        for(int i = 16; i <= 40; i ++)
        {
            distrb3.push_back(pcount[i - 16]);
        }
    }
    else if (type == 4)
    {
        for(int i = 16; i <= 40; i ++)
        {
            distrb4.push_back(pcount[i - 16]);
        }
    }
    

}


void Simulation_Ct_test::simuInfo(ofstream &f,int type)
{
    if(!f.is_open())
        cout <<"cannot open file_simuInfo"<<endl;

    if (type == 0)
    {
        for (int i = 0; i < Snum.size(); i ++)
        {
            f << Snum[i] <<' ';
        }
    }
    else if (type == 1)
    {
        for (int i = 0; i < Inum.size(); i ++)
        {
            f << Inum[i] <<' ';
        }
    }
    else if(type == 2)
    {
        for(int i = 0; i < Rnum.size(); i ++)
        {
            f << Rnum[i] << ' ';
        }
    }
    else if(type == 3)
    {
        for(int i = 0; i < totalInfection.size(); i ++)
        {
            f << totalInfection[i] << ' ';
        }
    }
    else if (type == 4)
    {
        for(int i = 0; i < dailyInfection.size(); i ++)
        {
            f << dailyInfection[i] << ' ';
        }
    }
    else if(type == 5)
    {
        for(int i = 0; i < RtList.size(); i ++)
        {
            f << RtList[i] << ' ';
        }
    }
    else if(type == 6)
    {
        for(int i = 0; i < ctMean.size(); i ++)
        {
            f << ctMean[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctMean1.size(); i ++)
        {
            f << ctMean1[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctMean2.size(); i ++)
        {
            f << ctMean2[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctMean3.size(); i ++)
        {
            f << ctMean3[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctMean4.size(); i ++)
        {
            f << ctMean4[i] << ' ';
        }

    }
    else if(type == 7)
    {
        for(int i = 0; i < ctSkew.size(); i ++)
        {
            f << ctSkew[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctSkew1.size(); i ++)
        {
            f << ctSkew1[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctSkew2.size(); i ++)
        {
            f << ctSkew2[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctSkew3.size(); i ++)
        {
            f << ctSkew3[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < ctSkew4.size(); i ++)
        {
            f << ctSkew4[i] << ' ';
        }
    }
    else if(type == 8)
    {
        for(int i = 0; i < generateList.size(); i ++)
        {
            f << generateList[i] << ' ';
        }
    }
    else if(type == 9)
    {
        for(int i = 0; i < loglikelihoodList.size(); i ++)
        {
            f << loglikelihoodList[i] << ' ';
        }
    }
    else if(type == 10)
    {
        for(int i = 0; i < distrb.size(); i ++)
        {
            f << distrb[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < distrb1.size(); i ++)
        {
            f << distrb1[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < distrb2.size(); i ++)
        {
            f << distrb2[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < distrb3.size(); i ++)
        {
            f << distrb3[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < distrb4.size(); i ++)
        {
            f << distrb4[i] << ' ';
        }
    }
    else if(type == 11)
    {
        f << Tstart;
    }
    else if(type == 12)
    {
        for(int i = 0; i < generation.size(); i ++)
        {
            f << generation[i] << ' ';
        }
    }
    else if (type == 13)
    {
        for (int i = 0; i < Inum.size(); i ++)
        {
            f << Inum[i] <<' ';
        }
        f << endl;
        for(int i = 0; i < testNum1.size(); i ++)
        {
            f << testNum1[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < testNum2.size(); i ++)
        {
            f << testNum2[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < testNum3.size(); i ++)
        {
            f << testNum3[i] << ' ';
        }
        f << endl;
        for(int i = 0; i < testNum4.size(); i ++)
        {
            f << testNum4[i] << ' ';
        }
    }
    
    f << endl;
}

void Simulation_Ct_test::Reset()
{

    oldInfection.clear();
    newInfection.clear();
    dailyInfection.clear();
    Snum.clear();
    Inum.clear();
    Rnum.clear();
    totalInfection.clear();
    infectNodes.clear();
    RtList.clear();
    generateList.clear();
    generation.clear();
    // ctMean.clear();
    // ctSkew.clear();
    loglikelihoodList.clear();
    base_RtList.clear();

    ctMean_onset.clear();
    ctSkew_onset.clear();
    distrb_onset.clear();


    ctMean.clear();
    ctSkew.clear();
    distrb.clear();

    ctMean1.clear();
    ctSkew1.clear();
    testNum1.clear();
    distrb1.clear();

    ctMean2.clear();
    ctSkew2.clear();
    testNum2.clear();
    distrb2.clear();

    ctMean3.clear();
    ctSkew3.clear();
    testNum3.clear();
    distrb3.clear();

    ctMean4.clear();
    ctSkew4.clear();
    testNum4.clear();
    distrb4.clear();

    meanList.clear();
    lowerList.clear();
    upperList.clear();
    Tstart = 0;
    
    totalI = 0;
    recover = 0;
    // net.contact.clear();
    for(int i = 0; i < net.nodesNum; i ++)
    {
        net.nodes[i].reset();
        // net.contact.push_back(vector<int> ());
    }
}


