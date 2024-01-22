#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <time.h>
#include <cmath>
#include "agent.h"
#include "population.h"
#include "network.h"
#include "distribution.h"
#include "simulation.h"
#include "run.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

int main()
{
    string path = "results\\ER\\testRate\\SF2.4";
    string netType = "ER";
    vector<int> d = {10};
    vector<double> R0 = {2.4};
    // vector<double> R0 = {2.0};
    int Nnum = 1;
    int num = 30;
    int base = 0;
    int over = 0;
    int ct = 1;

    run_Ct_test(path,netType,d,R0,Nnum,num,base,over,ct);


    



    // int num = 100;
    // Ct_trajectory(num);



    
    
    return 0;
}