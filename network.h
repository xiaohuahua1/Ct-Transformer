#ifndef NETWORKH
#define NETWORKH
#include "Agent.h"
#include "population.h"
#include <utility>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;


class Network{
    public:
        
        gsl_rng *R_GLOBAL;
        int nodesNum;
        vector<int> seedList;
        vector<vector<int>> contact;                             //each day, matrix of groups, list of nodes in each group 
        vector<Agent_SEIR> nodes;                                             

        Network(string path,int n);
        Network(int n);
        Network();
        void initNode(double Mu,double sig);
        void ER_network(int d);
        vector<double> getCDD(double alpha);
        // void degree_Seq(double alpha,int average_degree);
        void SF_Model(vector<double> CDD,double alpha,int average_degree);
        void getSeed(int d);
        void clearContact();
        ~Network()=default;

};

#endif