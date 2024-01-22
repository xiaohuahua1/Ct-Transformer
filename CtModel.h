#ifndef CTMODELH
#define CTMODELH
#include <iostream>
#include <vector>
#include <cmath>
#include "distribution.h"

using namespace std;

// double CtValue(int t);
// double CtValue_check(int t,double Mu);
vector<double> getCtValue(int incubation);
double calculate_mean(vector<double> ctList);
double calculate_skewness(vector<double> ctList,double mean);





#endif