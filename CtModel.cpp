#include "CtModel.h"
#include <iostream>
#include <math.h>

using namespace std;

const int Teclipse = 0;
const double Czero = 40.0;
const int Tpeak = 5;

// double CtValue(int t)
// {
//     double ct = 0.0;
//     double Cpeak = normal(19.7,2.0);
//     int Tend = round(normal(13.3,3.0));

//     double kup = (Cpeak - Czero) / (Tpeak - Teclipse);
//     double kdown = (Czero - Cpeak) / Tend;

//     if(t <= Teclipse)
//     {
//         ct = Czero;
//     }
//     else  if(t > Teclipse && t <= Teclipse + Tpeak)
//     {
//         ct = Czero + kup*(t - Teclipse);
//     }
//     else if(t > Teclipse + Tpeak && t <= Teclipse + Tpeak + Tend)
//     {
//         ct = Cpeak + kdown*(t - Teclipse - Tpeak);
//     }
//     else if(t > Teclipse + Tpeak + Tend)
//     {
//         ct = Czero;
//     }
//     return ct;
// }

// double CtValue_check(int t,double Mu)
// {
//     double ct = 0.0;
//     double Cpeak = normal(19.7,2.0);
//     int Tend = round(Poisson(Mu));
//     int Tpeak = Tend / 3;

//     cout << Tend <<' ' << Tpeak << endl;
    

//     if(Tpeak == 0)
//     {
//         Tpeak = 1;
//     }

//     double kup = (Cpeak - Czero) / (Tpeak - Teclipse);
//     double kdown = (Czero - Cpeak) / (Tend - Tpeak);

//     if(t <= Teclipse)
//     {
//         ct = Czero;
//     }
//     else if (t > Teclipse && t <= Teclipse + Tpeak)
//     {
//         ct = Czero + kup*(t - Teclipse);
//     }
//     else if(t > Teclipse + Tpeak && t <= Teclipse + Tend)
//     {
//         ct = Cpeak + kdown*(t - Teclipse - Tpeak);
//     }
//     else if(t > Teclipse + Tend)
//     {
//         ct = Czero;
//     }
    
//     return ct;

// }

vector<double> getCtValue(int incubation)
{
    vector<double> vlList;
    double Czero = 40.0;
    double Cpeak = normal(19.7,2.0);
    int Tshed = int(normal(11.9,0.94));
    if(incubation == 0)
    {
        incubation = 1;
    }

    double Kup = (Cpeak - Czero) / incubation;
    double Kdown = (Czero - Cpeak) / Tshed;
    double Ct = 0.0;

    for(int i = 0; i <= incubation; i ++)
    {
        Ct = Czero + Kup*i;
        vlList.push_back(Ct);
    }
    for(int i = incubation + 1; i <= Tshed + incubation; i ++)
    {
        Ct = Cpeak-Kdown*incubation + Kdown*i;
        vlList.push_back(Ct);
    }
    // for(int i = 0; i < vlList.size(); i ++)
    // {
    //     cout << vlList[i] <<' ';
    // }
    // cout << endl;
    return vlList;

}

double calculate_mean(vector<double> ctList)
{
    double mean = 0.0;
    if(ctList.size() > 0)
    {
        for(int i = 0; i < ctList.size(); i ++)
        {
            mean += ctList[i];
        }
        mean /= ctList.size();
    }
    else
    {
        mean = 40.0;
    }

    return mean;
}

double calculate_skewness(vector<double> ctList,double mean)
{
    double skew1 = 0.0;
    double skew2 = 0.0;
    double skew = 0.0;
    int n = ctList.size();
    double a = 1.5;
    
    if (n > 1)
    {
        for(int i = 0; i < n; i ++)
        {

        }
        for(int i = 0; i < n; i ++)
        {
            skew1 += pow((ctList[i] - mean),3);
            skew2 += pow((ctList[i] - mean),2);
        }
        skew1 = skew1 / n;
        skew2 = pow(skew2/(n - 1),a);
        if (skew2 > 0)
        {
            skew = skew1 / skew2;
        }
        else
        {
            skew = 0.0;
        }
    }

    // if (skew < -100)
    // {
    //     cout << skew << ' ' << mean << ' ' << n << ' '<< skew1 << ' ' << skew2 << endl;
    //     for(int i = 0; i < n; i ++)
    //     {
    //         cout << CtValue(dayList[i]) << ' ';
    //     }
    //     cout << endl;
    // }
    
    return skew;

}