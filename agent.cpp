#include "agent.h"
#include "distribution.h"
using namespace std;

// Agent::Agent(){
//     // incubation=Gamma();            
//     // Tsc=Poisson(kTsc);
//     // if(Tsc==0) Tsc=1;      //Tsc >=1
//     // Tcr=Poisson2(kTcr); 
// }

// void Agent::initial(int i, int a, int d,double h){
//     id=i;
//     age=a;
//     iDegree=d;
//     heterogeneity=h;
// }
// // bool Agent::is_infectious(int t){
// //     //infectious
// //     if(Lambda>=incubation){
// //         return true;
// //     }
// //     else if(t>=(timeInfected+incubation-Lambda)){
// //         return true;
// //     }
// //     else return false;
// // }
// void Agent::check_incubation(int t){
//     //symptom onset
//     if(t==(timeInfected+incubation)){
//         timeSymptomOn=t;
//         if(willSymp){
//             symptomatic=true;   
//         }          
//     }
// }
// int Agent::update(int t){
//     //recover state
//     if(t==(timeInfected+incubation+recovery)){
//         recoverd=true;
//         timeRecover=t;
//         return 1;
//     }
//     else
//     {
//         return 0;
//     }
// }
// void Agent::symTest(int t){
//     isTesting=true;
//     sampleCollected.push_back(t);
//     testType.push_back(1);
//     symTestFlag=1;
// }
// void Agent::reset(){
//     infector=-1;
//     symTestFlag=-1;  
    
//     infected=false;
//     symptomatic=false;
//     recoverd=false;
 
//     isTesting=false;
//     confirmed=false;


//     timeInfected=-1;
//     timeSymptomOn=-1;
//     timeRecover=-1;
//     timeConfirmd=-1;
//     timeTraced=-1;

//     sampleCollected.clear();
//     testResult.clear();
//     testType.clear();

// }

Agent_base::Agent_base(double incu_mean,double incu_sd,double pre_mean,double pre_sd,double infec_mean,double infec_sd,double test_a,double test_b)
{
    incubation=log_normal(incu_mean,incu_sd);
    pre_infectious = log_normal(pre_mean,pre_sd);  
    recovery = log_normal(infec_mean,infec_sd);
    Tsc = Gamma(test_a,1/test_b);
    if(Tsc==0) Tsc=1;      //Tsc >=1

}

void Agent_base::initial(int i)
{
    id = i;
}

bool Agent_base::is_infectious(int t){
    //infectious
    if(pre_infectious>=incubation){
        return true;
    }
    else if(t>=(timeInfected+incubation-pre_infectious)){
        return true;
    }
    else return false;

    // if (t > timeInfected && recoverd == false)
    // {
    //     return true;
    // }
    // return false;
}

void Agent_base::check_incubation(int t){
    //symptom onset
    if(t==(timeInfected+incubation)){
        timeSymptomOn=t;          
    }
}

int Agent_base::update(int t){
    //recover state
    // cout << "recover day:" << timeInfected <<' ' << incubation <<' ' << recovery << ' ' << timeInfected+incubation+recovery << endl;
    if((t >= (timeInfected+incubation+recovery)) && recoverd == false){
        recoverd=true;
        timeRecover=t;
        return 1;
    }
    else
    {
        return 0;
    }
}

int Agent_base::symTest(int t)
{
    if ((t >= (timeInfected + incubation + Tsc)) && tested == false)
    {
        tested = true;
        timeTested = t;
        return 1;
    }
    return 0;
    
}

void Agent_base::reset()
{
    infector=-1;
    infectNum = 0;

    
    infected=false;
    recoverd=false;
    tested = false;


    timeInfected=-1;
    timeSymptomOn=-1;
    timeRecover=-1;
    timeTested = -1;
}

Agent::Agent(double Mu)
{
    recovery = Mu;
}

Agent::Agent(){}

void Agent::initial(int i)
{
    id = i;
}

void Agent::initial_age(int i,int a)
{
    id = i;
    age = a;
}

bool Agent::is_infectious(int t){
    //infectious
    
    if(t >= timeInfected && recoverd == false){
        return true;
    }
    else return false;

}

int Agent::update(int t){
    
    if((t >= (timeInfected+recovery)) && recoverd == false){
        recoverd=true;
        timeRecover=t;
        return 1;
    }
    else
    {
        return 0;
    }
}

void Agent::reset()
{
    infector=-1;
    infectNum = 0;
    degree = 0;

    
    infected=false;
    recoverd=false;


    timeInfected=-1;
    timeRecover=-1;
}

Agent_SEIR::Agent_SEIR(int sigma,int gamma)
{
    incubation = sigma;
    recovery = gamma;
}
Agent_SEIR::Agent_SEIR(){}

bool Agent_SEIR::is_infectious(int t){
    //infectious
    if(t >= timeInfected + incubation)
    {
        if(symptomatic == false)
        {
            symptomatic = true;
            timeSym =  t;
        }
        if(recoverd == false)
        {
            return true;
        }
    }
    return false;

}

int Agent_SEIR::update(int t){
    
    if((t >= (timeInfected+incubation+recovery)) && recoverd == false){
        recoverd=true;
        timeRecover=t;
        vlList.clear();
        return 1;
    }
    else
    {
        return 0;
    }
}

void Agent_SEIR::reset()
{
    infector=-1;
    infectNum = 0;
    // degree = 0;
   
    infected=false;
    recoverd=false;
    symptomatic = false;
    checked = false;


    timeInfected=-1;
    timeRecover=-1;
    timeSym = -1;

    general = 0;

    vlList.clear();
}











