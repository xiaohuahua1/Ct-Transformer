#include "population.h"
#include "agent.h"
#include "distribution.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>
#include <utility>
#include <algorithm>

using namespace std;

// Population::Population(int n){
//     agentNum=n;
// }

// void Population::initial(){
//     string name1="data/vec_ShangHai.txt";
//     string name2="data/dis_num_group_contact.txt";
//     string name3="data/dis_num_individual_contact.txt";
//     string name4="data/age_individual.txt";
//     string name5="data/age_group.txt";
//     string name6="data/pro_age_group.txt";
//     // string name7="../network/heterogeneity.txt";

//     ifstream f;
//     string line;
//     int index;

//     /* size of each age group */
//     vector<double> vec;                                    //propotion of each age group of the population
//     f.open(name1,ios::in);
//     while(getline(f,line)){
//         stringstream ss(line);
//         double d;
//         while(ss>>d)
//             vec.push_back(d);
//     }
//     f.close();

//     ageClass=vec.size();
    
//     for(int i=1;i<ageClass;i++)
//         vec[i]+=vec[i-1];
//     for(int i=0;i<agentNum;i++){
//         index=0;
//         double p=Prand();
//         while(index+1<ageClass){
//             if(p>vec[index])
//                 index++;
//             else break;
//         }
//         listAge.push_back(index);     
//     }

//     /* read distribution infomation */
//     f.open(name2,ios::in);                      
//     while(getline(f,line)){
//         stringstream ss(line);
//         double d;
//         while(ss>>d){
//             disNumg.push_back(d);
//         }         
//     }
//     f.close();
    
//     f.open(name3,ios::in);
//     while(getline(f,line)){
//         stringstream ss(line);
//         vector<double> v;
//         double d;
//         while(ss>>d){        
//             v.push_back(d);
//         }
//         disNumi.push_back(v);
//     }
//     f.close();

//     f.open(name4,ios::in);
//     while(getline(f,line)){
//         stringstream ss(line);
//         vector<double> v;
//         double d;
//         while(ss>>d){
//             v.push_back(d);
//         }
//         disAgei.push_back(v);
//     }
//     f.close();
    
//     for(int i=0;i<ageClass;i++){
//         for(int j=1;j<ageClass;j++){
//             disAgei[i][j]+=disAgei[i][j-1];
//         }
//     }

//     f.open(name5,ios::in);
//     while(getline(f,line)){
//         stringstream ss(line);
//         vector<double> v;
//         double d;
//         while(ss>>d){
//             v.push_back(d);
//         }
//         disAgeg.push_back(v);
//         v.clear();
//     }
//     f.close();

//     for(int i=0;i<ageClass;i++){
//         for(int j=1;j<ageClass;j++){
//             disAgeg[i][j]+=disAgeg[i][j-1];
//         }
//     }

//     f.open(name6,ios::in);                       
//     while(getline(f,line)){
//         stringstream ss(line);
//         double d;
//         while(ss>>d){
//             pro_group.push_back(d);
//         }            
//     }
//     f.close();

//     /* initial the iDegree of nodes */
//     vector<vector<double>> test;
//     for(int i=0;i<ageClass;i++){
//         test.push_back({});
//         for(int j=0;j<41;j++){
//             for(int k=0;k<disNumi[i][j]*agentNum;k++)
//                 test[i].push_back(j);
//         }    
//     }
//     for(int i=0;i<agentNum;i++){
//         int a=listAge[i];
//         index=Uniform(agentNum);
//         int d=test[a][index];
//         listIdegree.push_back(d);
//     }

//     /*initial the heterogeneity of nodes */
//     // for(int i=0;i<agentNum;i++){
//     //     double h=Heter();
//     //     if(h<0.00001) h=0;
//     //     listH.push_back(h);
//     // }
    
//     // ofstream file;
//     // file.open(name7,ios::out);
//     // for(int i=0;i<agentNum;i++){
//     //     file<<listH[i]<<endl;
//     // }
//     // file.close();

//     // for(int i=0;i<ageClass;i++){
//     //     int s=0;
//     //     vector<int> v;
//     //     for(int j=0;j<agentNum;j++){
//     //         if(listAge[j]==i){
//     //             s++;
//     //             v.push_back(j);
//     //         }
//     //     }
//     //     ageSize.push_back(s);
//     //     ageGroup.push_back(v);
//     // }
// }