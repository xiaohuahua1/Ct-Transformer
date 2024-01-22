#include "distribution.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>

using namespace std;

int Uniform(int size){                        
    static default_random_engine e(seed1);
    static uniform_int_distribution<int> u(1000000);
    int rand=u(e);
    return rand%size;
}

int Gamma(double a,double b){
    static default_random_engine e(seed2);
    // static gamma_distribution<double> u(kShape,1/kRate);
    static gamma_distribution<double> u(a,b);
    return round(u(e));
}

int Exp(double kMu){
    static default_random_engine e(seed3);
    static exponential_distribution<double> u(kMu);
    return round(u(e));
}

double Prand(){
    static default_random_engine e(seed4);
    static uniform_real_distribution<double> u(0.0,1);
    return u(e);
}

int Poisson(int m){
    static default_random_engine e(seed5);
    static poisson_distribution<int> u(m);
    return u(e);
}

long double Poisson2(int k,long double m){
    long double p = exp(-m);

    for(int i = 1; i <= k ; i ++)
    {
        double a = m/i;
        p = a*p;
    }

    return p;
}

double Heter(){
    static default_random_engine e(seed6);
    static gamma_distribution<double> u(kHeter,kHeter);
    return u(e);
}

int binomial(double p,int n)
{
    // cout << p << ' ' << n << endl;
    static default_random_engine e(seed8);
    static binomial_distribution<int> u(n,p);
    // cout << u(e) << endl;
    return u(e);
}

int log_normal(double mean,double sd)
{
    static default_random_engine e(seed9);
    static lognormal_distribution<double> u(mean,sd);
    return round(u(e));
}

double normal(double mean,double sd)
{
    static default_random_engine e(seed11);
    static normal_distribution<double> u(mean,sd);
    return u(e);
}

double normal_Rt(double mean,double sd)
{
    static default_random_engine e(seed12);
    static normal_distribution<double> u(mean,sd);
    return u(e);
}