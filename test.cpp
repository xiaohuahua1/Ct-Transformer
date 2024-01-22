#include <iostream>
#include <cmath>
#include "agent.h"
// #include "population.h"
#include "network.h"
#include "distribution.h"
// #include "CtModel.h"
#include "simulation.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

gsl_rng *R_GLOBAL;
using namespace std;

int main()
{

    // int num = 100000;
    // double Mu = 4.0;
    // double sig = 5.0;
    // double alpha = 3.0;
    // int d = 5;

    // Network net(num);
    // net.initNode(Mu,sig);
    // // net.SF_Model(alpha,d);
    // net.ER_network(d);
    // Simulation_Ct s(net);
    // s.initialSeeds();
    // cout << endl;
    int cnt = 0;
    for(int i = 3; i < 10; i ++)
    {
        cnt ++;
    }
    cout << cnt << endl;







    
    return 0;
}