#ifndef SIMULATIONH
#define SIMULATIONH

#include "agent.h"
#include "population.h"
#include "distribution.h"
#include "network.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;



class Simulation
{
    public:
        gsl_rng *R_GLOBAL;
        // vector<Agent> nodes;
        // int nodesNum = 0;
        Network net;
        int endTime=-1;             //time of simulation ends
        int kSeed=1;                //initial infected seed number

        double Beta;               //transmit probability
        // double Mu;

        vector<int> oldInfection,newInfection;

        //simulation results
        vector<int> dailyInfection;         //new infection at each day
        vector<int> Snum;                   //Susceptible at present
        vector<int> Inum;
        vector<int> Rnum;
        vector<int> totalInfection;         //total infection
        vector<vector<int>> infectNodes;
        vector<double> RtList;
        vector<int> generateList;

        int totalI=0;                       //total infections num
        int recover = 0;

        //baseline
        int nsim;
        int tau;
        double sigma;
        vector<long double> loglikelihoodList;
        vector<vector<double>> base_RtList;
        int Tstart=0;
        int resim;

        vector<double> meanList;
        vector<double> lowerList;
        vector<double> upperList;


        Simulation(Network &net1);
        Simulation();
        ~Simulation()=default;
        void simuInfo(ofstream &f,int type);             //output simulation result

        // void initialNodes();
        void initialSeeds();
        void infect_HM(int t);
        void recovery_HM(int t);
        void calculate_Rt();
        void transmission_HM();
        // vector<int> ctTest(int t,double p);
        long double loglikelihood(int tt,double Rt,vector<double> Tg);
        void Baseline_Rt();
        void outRt(ofstream &f,int type);
        vector<double> calculate(vector<double> dataList); 
        vector<double> smooth(vector<double> dataList);
        void calculate_Baseline_Rt();
        void Reset();

};

class Simulation_Ct:public Simulation
{
    public:
    // vector<Agent_SEIR> nodes;
    // double sig;

    vector<double> ctMean;
    vector<double> ctSkew;
    vector<double> ctMean_onset;
    vector<double> ctSkew_onset;
    vector<double> distrb;
    vector<double> distrb_onset;
    vector<double> generation;
    // vector<vector<int>> generation_all;

    Simulation_Ct(Network &net1);
    Simulation_Ct();
    ~Simulation_Ct()=default;
    // void initialNodes();
    // void initialNodes_age();
    // void initialSeeds();
    void infect_HM(int t);
    // void recovery_HM(int t);
    void transmission_HM();
    void transmission_net(string path,int printct);
    // void calculate_Rt();
    vector<vector<double>> CtTest(int t);
    void csvCt();
    void calculate_distrb(vector<double> ctList,vector<double> dur_time,vector<double> ctList_onset);
    void simuInfo(ofstream &f,int type); 
    void Reset();

};

class Simulation_Ct_test:public Simulation_Ct
{
    public:
    vector<double> ctMean1;
    vector<double> ctSkew1;
    vector<int> testNum1;
    vector<double> distrb1;

    vector<double> ctMean2;
    vector<double> ctSkew2;
    vector<int> testNum2;
    vector<double> distrb2;

    vector<double> ctMean3;
    vector<double> ctSkew3;
    vector<int> testNum3;
    vector<double> distrb3;

    vector<double> ctMean4;
    vector<double> ctSkew4;
    vector<int> testNum4;
    vector<double> distrb4;

    Simulation_Ct_test(Network &net1);
    Simulation_Ct_test();
    ~Simulation_Ct_test()=default;
    void transmission_HM();
    void transmission_net(string path,int printct);
    vector<double> CtTest_type(int type,int t,ofstream& daily,ofstream& generation);
    void calculate_distrb_type(vector<double> ctList,int type);
    void simuInfo(ofstream &f,int type); 
    void Reset();


};



#endif

