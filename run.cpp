#include "run.h"
#include <iostream>
#include "simulation.h"
#include "CtModel.h"
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;



void run_Ct(string path,string netType,vector<int> d,vector<double> R0,int Nnum,int num,int base,int over,int ct)
{

    ofstream S;
    ofstream I;
    ofstream R;
    ofstream total;
    ofstream Rt;
    ofstream daily;
    ofstream ctMean;
    ofstream ctSkew;
    ofstream testNum;
    ofstream distrb;
    ofstream baseMean;
    ofstream baseUpper;
    ofstream baseLower;
    ofstream Tstart;
    ofstream generate;
    ofstream distrb_onset;
    ofstream ctMean_onset;
    ofstream ctSkew_onset;
    int printct = 1;

    if (over == 1)
    {
        S.open(path + "\\S.txt",ios::app);
        I.open(path + "\\I.txt",ios::app);
        R.open(path + "\\R.txt",ios::app);
        total.open(path + "\\totalInfection.txt",ios::app);
        Rt.open(path + "\\Rt.txt",ios::app);
        daily.open(path + "\\dailyInfection.txt",ios::app);
        // ofstream generate(path + "\\generate.txt");
        ctMean.open(path + "\\ctMean.txt",ios::app);
        ctSkew.open(path + "\\ctSkew.txt",ios::app);
        // ofstream likelihood(path + "\\likelihood.txt");
        // ofstream Rtbase(path + "\\Rtbase.txt");
        // ofstream Rtbase1(path + "\\Rtbase1.txt");
        testNum.open(path + "\\testNum.txt",ios::app);
        distrb.open(path + "\\distrb.txt",ios::app);
        baseMean.open(path + "\\baseMean.txt",ios::app);
        baseUpper.open(path + "\\baseUpper.txt",ios::app);
        baseLower.open(path + "\\baseLower.txt",ios::app);
        Tstart.open(path + "\\Tstart.txt",ios::app);
        generate.open(path + "\\generate.txt",ios::app);
        distrb_onset.open(path + "\\distrbOnset.txt",ios::app);
        ctMean_onset.open(path + "\\ctMeanOnset.txt",ios::app);
        ctSkew_onset.open(path + "\\ctSkewOnset.txt",ios::app);
    }
    else
    {
        S.open(path + "\\S.txt");
        I.open(path + "\\I.txt");
        R.open(path + "\\R.txt");
        total.open(path + "\\totalInfection.txt");
        Rt.open(path + "\\Rt.txt");
        daily.open(path + "\\dailyInfection.txt");
        // ofstream generate(path + "\\generate.txt");
        ctMean.open(path + "\\ctMean.txt");
        ctSkew.open(path + "\\ctSkew.txt");
        // ofstream likelihood(path + "\\likelihood.txt");
        // ofstream Rtbase(path + "\\Rtbase.txt");
        // ofstream Rtbase1(path + "\\Rtbase1.txt");
        testNum.open(path + "\\testNum.txt");
        distrb.open(path + "\\distrb.txt");
        baseMean.open(path + "\\baseMean.txt");
        baseUpper.open(path + "\\baseUpper.txt");
        baseLower.open(path + "\\baseLower.txt");
        Tstart.open(path + "\\Tstart.txt");
        generate.open(path + "\\generate.txt");
        distrb_onset.open(path + "\\distrbOnset.txt");
        ctMean_onset.open(path + "\\ctMeanOnset.txt");
        ctSkew_onset.open(path + "\\ctSkewOnset.txt");
    }

    int agentNum = 100000;

    double Mu = 4.0;
    double sig = 5.0;
    Network net(agentNum);
    net.initNode(Mu,sig);
    if (netType == "ER")
    {
        for(int i = 0; i < d.size(); i ++)
        {
            for(int j = 0; j < R0.size(); j++)
            {
                // cout << d[i] << ' ' << R0[j] << endl;
                for(int x = 0; x < Nnum; x ++)
                {
                    cout << d[i] << ' ' << R0[j] << ' ' << x << endl;
                    net.ER_network(d[i]);
                    net.getSeed(d[i]);
                    Simulation_Ct s(net);
                    s.nsim = 50000;
                    s.tau = 4;
                    s.sigma = 0.1;
                    s.resim = 1000;
                    s.Beta = (R0[j])/(Mu*d[i]);

                    for(int z = 0; z < num; z ++)
                    {
                        if(ct == 0)
                        {
                            s.transmission_net(path,0);
                        }
                        else
                        {
                            s.transmission_net(path,printct);
                        }

                        s.simuInfo(S,0);
                        s.simuInfo(I,1);
                        s.simuInfo(R,2);
                        s.simuInfo(total,3);
                        s.simuInfo(daily,4);
                        s.simuInfo(Rt,5);
                        s.simuInfo(ctMean,6);
                        s.simuInfo(ctSkew,7);
                        s.simuInfo(distrb,10);
                        s.simuInfo(generate,12);
                        s.simuInfo(distrb_onset,13);
                        s.simuInfo(ctMean_onset,14);
                        s.simuInfo(ctSkew_onset,15);

                        if (base == 1)
                        {
                            s.Baseline_Rt();
                            s.calculate_Baseline_Rt();
                            // s.simuInfo(generate,8);
                            // s.simuInfo(likelihood,9);

                            s.outRt(baseMean,0);
                            s.outRt(baseLower,1);
                            s.outRt(baseUpper,2);
                            s.simuInfo(Tstart,11);
                        }
                        s.Reset();
                        printct += 1;
                    }
                    net.clearContact();
                }
            }
           
        }
    }
    else if(netType == "SF")
    {
        double alpha = 3.0;
        vector<double> CDD = net.getCDD(alpha);
        for(int i = 0; i < d.size(); i ++)
        {
            for(int j = 0; j < R0.size(); j++)
            {
                // cout << d[i] << ' ' << R0[j] << endl;
                for(int x = 0; x < Nnum; x ++)
                {
                    cout << d[i] << ' ' << R0[j] << ' ' << x << endl;
                    net.SF_Model(CDD,alpha,d[i]);
                    net.getSeed(d[i]);
                    Simulation_Ct s(net);
                    s.Beta = (R0[j])/(Mu*d[i]);
                    s.nsim = 50000;
                    s.tau = 4;
                    s.sigma = 0.1;
                    s.resim = 1000;

                    for(int z = 0; z < num; z ++)
                    {
                        if(ct == 0)
                        {
                            s.transmission_net(path,0);
                        }
                        else
                        {
                            s.transmission_net(path,printct);
                        }

                        s.simuInfo(S,0);
                        s.simuInfo(I,1);
                        s.simuInfo(R,2);
                        s.simuInfo(total,3);
                        s.simuInfo(daily,4);
                        s.simuInfo(Rt,5);
                        s.simuInfo(ctMean,6);
                        s.simuInfo(ctSkew,7);
                        s.simuInfo(distrb,10);
                        s.simuInfo(generate,12);
                        s.simuInfo(distrb_onset,13);
                        s.simuInfo(ctMean_onset,14);
                        s.simuInfo(ctSkew_onset,15);

                        if (base == 1)
                        {
                            s.Baseline_Rt();
                            s.calculate_Baseline_Rt();
                            // s.simuInfo(generate,8);
                            // s.simuInfo(likelihood,9);

                            s.outRt(baseMean,0);
                            s.outRt(baseLower,1);
                            s.outRt(baseUpper,2);
                            s.simuInfo(Tstart,11);
                        }
        
                        s.Reset();
                        printct += 1;
                    }
                    net.clearContact();

                }
            }
           
        }
    }

    S.close();
    I.close();
    R.close();
    total.close();
    daily.close();
    Rt.close();
    ctMean.close();
    ctSkew.close();
    // generate.close();
    // likelihood.close();
    testNum.close();
    distrb.close();
    baseMean.close();
    baseLower.close();
    baseUpper.close();
    Tstart.close();
    generate.close();
    distrb_onset.close();

}

/**
 * @brief 哈哈哈哈
 * @param path 啦啦啦
 * 
 **/
void run_Ct_test(string path,string netType,vector<int> d,vector<double> R0,int Nnum,int num,int base,int over,int ct)
{

    ofstream S;
    ofstream I;
    ofstream R;
    ofstream total;
    ofstream Rt;
    ofstream daily;
    ofstream ctMean;
    ofstream ctSkew;
    ofstream testNum;
    ofstream distrb;
    ofstream baseMean;
    ofstream baseUpper;
    ofstream baseLower;
    ofstream Tstart;
    ofstream generate;
    int printct = 1;

    if (over == 1)
    {
        S.open(path + "\\S.txt",ios::app);
        I.open(path + "\\I.txt",ios::app);
        R.open(path + "\\R.txt",ios::app);
        total.open(path + "\\totalInfection.txt",ios::app);
        Rt.open(path + "\\Rt.txt",ios::app);
        daily.open(path + "\\dailyInfection.txt",ios::app);
        // ofstream generate(path + "\\generate.txt");
        ctMean.open(path + "\\ctMean.txt",ios::app);
        ctSkew.open(path + "\\ctSkew.txt",ios::app);
        // ofstream likelihood(path + "\\likelihood.txt");
        // ofstream Rtbase(path + "\\Rtbase.txt");
        // ofstream Rtbase1(path + "\\Rtbase1.txt");
        testNum.open(path + "\\testNum.txt",ios::app);
        distrb.open(path + "\\distrb.txt",ios::app);
        baseMean.open(path + "\\baseMean.txt",ios::app);
        baseUpper.open(path + "\\baseUpper.txt",ios::app);
        baseLower.open(path + "\\baseLower.txt",ios::app);
        Tstart.open(path + "\\Tstart.txt",ios::app);
        generate.open(path + "\\generate.txt",ios::app);
        
    }
    else
    {
        S.open(path + "\\S.txt");
        I.open(path + "\\I.txt");
        R.open(path + "\\R.txt");
        total.open(path + "\\totalInfection.txt");
        Rt.open(path + "\\Rt.txt");
        daily.open(path + "\\dailyInfection.txt");
        // ofstream generate(path + "\\generate.txt");
        ctMean.open(path + "\\ctMean.txt");
        ctSkew.open(path + "\\ctSkew.txt");
        // ofstream likelihood(path + "\\likelihood.txt");
        // ofstream Rtbase(path + "\\Rtbase.txt");
        // ofstream Rtbase1(path + "\\Rtbase1.txt");
        distrb.open(path + "\\distrb.txt");
        baseMean.open(path + "\\baseMean.txt");
        baseUpper.open(path + "\\baseUpper.txt");
        baseLower.open(path + "\\baseLower.txt");
        Tstart.open(path + "\\Tstart.txt");
        generate.open(path + "\\generate.txt");
        testNum.open(path + "\\testNum.txt");
        
    }

    int agentNum = 100000;

    double Mu = 4.0;
    double sig = 5.0;
    Network net(agentNum);
    net.initNode(Mu,sig);
    if (netType == "ER")
    {
        for(int i = 0; i < d.size(); i ++)
        {
            for(int j = 0; j < R0.size(); j++)
            {
                // cout << d[i] << ' ' << R0[j] << endl;
                for(int x = 0; x < Nnum; x ++)
                {
                    cout << d[i] << ' ' << R0[j] << ' ' << x << endl;
                    net.ER_network(d[i]);
                    net.getSeed(d[i]);
                    Simulation_Ct_test s(net);
                    s.nsim = 50000;
                    s.tau = 4;
                    s.sigma = 0.1;
                    s.resim = 1000;
                    s.Beta = (R0[j])/(Mu*d[i]);

                    for(int z = 0; z < num; z ++)
                    {
                        if(ct == 0)
                        {
                            s.transmission_net(path,0);
                        }
                        else
                        {
                            s.transmission_net(path,printct);
                        }
                        
                        s.simuInfo(S,0);
                        s.simuInfo(I,1);
                        s.simuInfo(R,2);
                        s.simuInfo(total,3);
                        s.simuInfo(daily,4);
                        s.simuInfo(Rt,5);
                        s.simuInfo(ctMean,6);
                        s.simuInfo(ctSkew,7);
                        s.simuInfo(distrb,10);
                        s.simuInfo(generate,12);
                        s.simuInfo(testNum,13);

                        if (base == 1)
                        {
                            s.Baseline_Rt();
                            s.calculate_Baseline_Rt();
                            // s.simuInfo(generate,8);
                            // s.simuInfo(likelihood,9);

                            s.outRt(baseMean,0);
                            s.outRt(baseLower,1);
                            s.outRt(baseUpper,2);
                            s.simuInfo(Tstart,11);
                        }
                        s.Reset();
                        printct += 1;
                    }
                    net.clearContact();

                }
            }
           
        }
    }
    else if(netType == "SF")
    {
        double alpha = 3.0;
        vector<double> CDD = net.getCDD(alpha);
        for(int i = 0; i < d.size(); i ++)
        {
            for(int j = 0; j < R0.size(); j++)
            {
                // cout << d[i] << ' ' << R0[j] << endl;
                for(int x = 0; x < Nnum; x ++)
                {
                    cout << d[i] << ' ' << R0[j] << ' ' << x << endl;
                    net.SF_Model(CDD,alpha,d[i]);
                    net.getSeed(d[i]);
                    Simulation_Ct_test s(net);
                    s.Beta = (R0[j])/(Mu*d[i]);
                    s.nsim = 50000;
                    s.tau = 4;
                    s.sigma = 0.1;
                    s.resim = 1000;

                    for(int z = 0; z < num; z ++)
                    {
                        if(ct == 0)
                        {
                            s.transmission_net(path,0);
                        }
                        else
                        {
                            s.transmission_net(path,printct);
                        }

                        s.simuInfo(S,0);
                        s.simuInfo(I,1);
                        s.simuInfo(R,2);
                        s.simuInfo(total,3);
                        s.simuInfo(daily,4);
                        s.simuInfo(Rt,5);
                        s.simuInfo(ctMean,6);
                        s.simuInfo(ctSkew,7);
                        s.simuInfo(distrb,10);
                        s.simuInfo(generate,12);
                        s.simuInfo(testNum,13);

                        if (base == 1)
                        {
                            s.Baseline_Rt();
                            s.calculate_Baseline_Rt();
                            // s.simuInfo(generate,8);
                            // s.simuInfo(likelihood,9);

                            s.outRt(baseMean,0);
                            s.outRt(baseLower,1);
                            s.outRt(baseUpper,2);
                            s.simuInfo(Tstart,11);
                        }
        
                        s.Reset();
                        printct += 1;
                    }
                    net.clearContact();

                }
            }
           
        }
    }

    S.close();
    I.close();
    R.close();
    total.close();
    daily.close();
    Rt.close();
    ctMean.close();
    ctSkew.close();
    // generate.close();
    // likelihood.close();
    testNum.close();
    distrb.close();
    baseMean.close();
    baseLower.close();
    baseUpper.close();
    Tstart.close();
    generate.close();
    testNum.close();

}
void Ct_trajectory(int num)
{
    gsl_rng *R_GLOBAL=gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(R_GLOBAL,time(NULL));

    ofstream Ct;
    Ct.open("results\\trajectory.txt");
    double sig = 5.0;
    for(int i = 0; i < num; i ++)
    {
        int incu = gsl_ran_poisson(R_GLOBAL,sig);
        vector<double> vlList = getCtValue(incu);
        for(int j = 0; j < vlList.size(); j ++)
        {
            Ct << vlList[j] << ' ';
        }
        Ct << endl;
    }
    Ct.close();
}