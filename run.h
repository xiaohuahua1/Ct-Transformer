#ifndef RUNH
#define RUNH

#include "distribution.h"
#include <iostream>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

void run_Ct(string path,string netType,vector<int> d,vector<double> R0,int Nnum,int num,int base,int over,int ct);
void run_Ct_test(string path,string netType,vector<int> d,vector<double> R0,int Nnum,int num,int base,int over,int ct);
void Ct_trajectory(int num);

#endif
