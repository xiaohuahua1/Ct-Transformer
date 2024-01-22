#ifndef AGENTH
#define AGENTH

#include <iostream>
#include <vector>
#include <utility>

using namespace std;

// const int kTsc=2;                                           //mean delay
// const int kTcr=1;                                           //mean delay
// // const int Lambda=2;   //pre-symptomatic infectious stage

// class Agent
// {
// public:
//     /* NODE INFORMATION */
//     int id;                     //0,1,...,n-1
//     int age;                    //age_group:0,1,..,15
//     int iDegree;                //num of individual contacts
//     double heterogeneity;       //heterogeneity
//     double susceptibility;      //susceptibility to infection by age a  
//     int incubation;             //incubation period
//     int recovery;               //recovery period 
    
//     int Tsc;                    //Delay between symptom set on and collection of the sample
//     int Tcr;                    //Delay between collection of the sample and PCR results
      
//     vector<int> iCnt;           //individual contacts
//     int infector=-1;            //infection from infector

//     /* STATE SYMBOL */
//     bool infected=false;        //be infected
//     bool symptomatic=false;     //have developed symptom
//     bool recoverd=false;        //have recoverd
    
//     bool willSymp=true;        //will develop symptom after incubation
//     bool willTest=false;        //will be tested if willsymp
//     int symTestFlag=-1;         //flag=-1,:have not start sym-test; flag=1: have sampled

//     bool isTesting=false;       //is waiting for test results
//     bool confirmed=false;       //be confirmed

//     /* TIME POINT */
//     int timeInfected=-1;
//     int timeSymptomOn=-1;
//     int timeRecover=-1;
//     int timeConfirmd=-1;
//     int timeTraced=-1;

//     vector<int> sampleCollected;            //time point when sample collected
//     vector<bool> testResult;                //result of test
//     vector<int> testOut;                    //time when test result is out
//     vector<int> testType;                   //1:symptom driven or 0:by contact tracing

//     /* FUNCTION */
//     Agent();
//     ~Agent()=default;

//     void initial(int i, int a, int d, double h);        //initial parameter
//     bool is_infectious(int t);                          //if is infectious
//     void check_incubation(int t);                       //if incubation is over
//     int update(int t);         //update recovery state
//     void symTest(int t);        //add symptom-driven test
//     void reset();               //reset state and timepoint
    
// };

class Agent_base
{
    public:
    /* NODE INFORMATION */
    int id;                     //0,1,...,n-1
    int recovery;               //recovery period
    int infector=-1;            //infection from infector
    int incubation;             //incubation period
    int infectNum = 0;
    int pre_infectious;
    int Tsc;                    //Delay between symptom set on and collection of the sample

    /* STATE SYMBOL */
    bool infected=false;        //be infected
    bool recoverd=false;        //have recoverd
    bool tested = false;

     /* TIME POINT */
    int timeInfected=-1;
    int timeSymptomOn=-1;
    int timeRecover=-1;
    int timeTested = -1;

    /* FUNCTION */
    Agent_base(double incu_mean,double incu_sd,double pre_mean,double pre_sd,double infec_mean,double infec_sd,double test_a,double test_b);
    ~Agent_base()=default;

    void initial(int i);        //initial parameter
    bool is_infectious(int t);  //if is infectious
    void check_incubation(int t); 
    int update(int t);          //update recovery state
    int symTest(int t);
    void reset();               //reset state and timepoint

};

class Agent
{
    public:
    /* NODE INFORMATION */
    int id;                     //0,1,...,n-1
    int recovery;               //recovery period
    int age = -1;
    int degree = 0;
    int infector=-1;            //infection from infector
    
    int infectNum = 0;


    /* STATE SYMBOL */
    bool infected=false;        //be infected
    bool recoverd=false;        //have recoverd

     /* TIME POINT */
    int timeInfected=-1;
    int timeRecover=-1;

    /* FUNCTION */
    Agent(double Mu);
    Agent();
    ~Agent()=default;

    void initial(int i);        //initial parameter
    void initial_age(int i,int a);
    bool is_infectious(int t);  //if is infectious
    int update(int t);          //update recovery state
    void reset();               //reset state and timepoint

};

class Agent_SEIR:public Agent
{
    public:
    int incubation;
    bool symptomatic=false;
    bool checked = false;
    int timeSym = -1;
    int general = 0;
    vector<double> vlList;

    Agent_SEIR(int sigma,int gamma);
    Agent_SEIR();
    ~Agent_SEIR()=default;
    bool is_infectious(int t);  //if is infectious
    int update(int t);          //update recovery state
    void reset();               //reset state and timepoint

};


#endif