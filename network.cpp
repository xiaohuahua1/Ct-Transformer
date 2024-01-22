#include "agent.h"
#include "population.h"
#include "network.h"
#include "distribution.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

using namespace std;

Network::Network(string path,int n)
{
    nodesNum = n;
    R_GLOBAL=gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(R_GLOBAL,time(NULL));

    ifstream f;
    string line;
    f.open(path);
    vector<int> tmp;
    int i = 0;
    if(f)
    {
        while (getline(f,line))
        {
            i ++; istringstream is(line);
            int index;
            while(!is.eof())
            {
                is >> index;
                // cout << index << endl;
                tmp.push_back(index);
            }
            contact.push_back(tmp);
            tmp.clear();
            line.clear();

        }
    }
    else
    {
        cout << "Can not open file." << endl;
    }
    f.close();

}

Network::Network(int n)
{
    nodesNum = n;
    R_GLOBAL=gsl_rng_alloc(gsl_rng_taus2);
	gsl_rng_set(R_GLOBAL,time(NULL));
}

Network::Network(){}

void Network::initNode(double Mu,double sig)
{
    for(int i = 0; i < nodesNum; i++)
    {
        int incu = gsl_ran_poisson(R_GLOBAL,sig);
        int reco = gsl_ran_poisson(R_GLOBAL,Mu);
        Agent_SEIR a(incu,reco);
        a.initial(i);
        // vector<int> contactn = contact[i];
        // a.degree = contactn.size();
        nodes.push_back(a);
        contact.push_back(vector<int> ());
    }
}

void Network::ER_network(int d)
{
    // for(int i = 0; i < nodesNum - 1; i ++)
    // {
    //     vector<int> nodeA = contact[i];
    //     for(int j = i + 1; j < nodesNum; j ++)
    //     {
    //         vector<int> nodeB = contact[j];
    //         float rp = gsl_rng_uniform(R_GLOBAL);
    //         if(rp < p && (find(nodeA.begin(),nodeA.end(),j)==nodeA.end()) && (find(nodeB.begin(),nodeB.end(),i)==nodeB.end()))
    //         {
    //             nodes[i].degree += 1;
    //             nodes[j].degree += 1;

    //             contact[i].push_back(j);
    //             contact[j].push_back(i);
    //         }
    //     }
    // }

    int edge = 0.5*nodesNum*d;
    while(edge > 0)
    {
        int nodeA = gsl_rng_uniform(R_GLOBAL)*nodesNum;
        int nodeB = gsl_rng_uniform(R_GLOBAL)*nodesNum;
        vector<int> contactA = contact[nodeA];
        vector<int> contactB = contact[nodeB];
        int cnt = 0;
        bool flag = true;

        while(nodeA == nodeB || 
        (find(contactA.begin(),contactA.end(),nodeB)!=contactA.end()) ||
        (find(contactB.begin(),contactB.end(),nodeA)!=contactB.end()))
        {
            if(cnt > 10)
            {
                flag = false;
                break;
            }

            nodeA = gsl_rng_uniform(R_GLOBAL)*nodesNum;
            nodeB = gsl_rng_uniform(R_GLOBAL)*nodesNum;
            contactA = contact[nodeA];
            contactB = contact[nodeB];
            cnt ++;
        }
        if(flag)
        {
            nodes[nodeA].degree += 1;
            nodes[nodeB].degree += 1;
            contact[nodeA].push_back(nodeB);
            contact[nodeB].push_back(nodeA);
        }
        edge --;
    }


}

vector<double> Network::getCDD(double alpha)
{
    int kmin=3;			
	int kmax=(int)pow((double)nodesNum, 1/(double)(alpha-1));		
	int i,j;

	double zeta=0;   //zeta

    vector<double> seqence;
    vector<double> CDD;

    for(i=0;i<nodesNum/2;i++)
    {
        seqence.push_back(0);
        CDD.push_back(0);
    }

    for(i=kmin;i<=kmax;i++)		
	{
        seqence[i] = pow(i*1.0,-alpha);	
	    zeta=zeta+seqence[i];
	}

    for(i=kmin;i<=kmax;i++)
	{
		seqence[i]=(seqence[i])/zeta;
		CDD[i]=CDD[i-1]+seqence[i];
	}
    return CDD;
}

// void Network::degree_Seq(double alpha,int average_degree)
// {
//     int kmin=3;			
// 	int kmax=(int)pow((double)nodesNum, 1/(double)(alpha-1));		
// 	int i,j;

// 	double zeta=0;   //zeta

//     vector<double> seqence;
//     vector<double> CDD;

//     for(i=0;i<nodesNum/2;i++)
//     {
//         seqence.push_back(0);
//         CDD.push_back(0);
//     }

//     for(i=kmin;i<=kmax;i++)		
// 	{
//         seqence[i] = pow(i*1.0,-alpha);	
// 	    zeta=zeta+seqence[i];
// 	}

//     for(i=kmin;i<=kmax;i++)
// 	{
// 		seqence[i]=(seqence[i])/zeta;
// 		CDD[i]=CDD[i-1]+seqence[i];
// 	}

//     int total = 0;
//     for(i=0;i<nodesNum;i++)
// 	{
// 		float rd=gsl_rng_uniform(R_GLOBAL);
// 		for(j=kmin;j<=kmax;j++)
// 		{
// 			if(rd<CDD[j])
// 			{
// 				nodes[i].degree=j;
// 				total+=j;
// 				break;
// 			}
// 		}
// 	}
//     float fact_degree=(total+0.0)/(nodesNum+0.0);
//     for(i=0;i<nodesNum;i++)
// 	{
// 		float d=nodes[i].degree/fact_degree*average_degree;
// 		int low=(int)d;
// 		int high=(int)d+1;
// 		float p_choose=d-(float)low;
// 		float f_rand=gsl_rng_uniform(R_GLOBAL);
//         // float f_rand=rand()/(float)RAND_MAX;
// 		if(f_rand<p_choose)
// 		{
// 			nodes[i].degree=high;
// 		}
// 		else
// 			nodes[i].degree=low;
// 	}


// }

void Network::SF_Model(vector<double> CDD,double alpha,int average_degree)
{
    // degree_Seq(alpha,average_degree);
    int kmin=3;			
	int kmax=(int)pow((double)nodesNum, 1/(double)(alpha-1));	
    vector<int> seq;

    int i,j;

    int total = 0;
    for(i=0;i<nodesNum;i++)
	{
		float rd=gsl_rng_uniform(R_GLOBAL);
		for(j=kmin;j<=kmax;j++)
		{
			if(rd<CDD[j])
			{
				nodes[i].degree=j;
				total+=j;
				break;
			}
		}
	}
    float fact_degree=(total+0.0)/(nodesNum+0.0);
    for(i=0;i<nodesNum;i++)
	{
		float d=nodes[i].degree/fact_degree*average_degree;
		int low=(int)d;
		int high=(int)d+1;
		float p_choose=d-(float)low;
		float f_rand=gsl_rng_uniform(R_GLOBAL);
        // float f_rand=rand()/(float)RAND_MAX;
		if(f_rand<p_choose)
		{
			nodes[i].degree=high;
		}
		else
			nodes[i].degree=low;
	}

    int count=0;	

	int m,n;
	
	int temp;
	for(i=0;i<nodesNum;i++)
	{
		for(j=0;j<nodes[i].degree;j++)
		{
			// seq[count++]=i;
            seq.push_back(i);
            count ++;
		}
	}
	j=count;
	//printf("a\n");
	while(j>1)
	{
        m = gsl_rng_uniform(R_GLOBAL)*j;
        n = gsl_rng_uniform(R_GLOBAL)*j;
        vector<int> nodeM = contact[seq[m]];
        vector<int> nodeN = contact[seq[n]];
        int cnt = 0;
        bool flag = true;

        while(m==n || 
        (find(nodeM.begin(),nodeM.end(),seq[n])!=nodeM.end()) || 
        (find(nodeN.begin(),nodeN.end(),seq[m])!=nodeN.end()) || 
        (seq[m] == seq[n]))
        {
            if(cnt >= 10)
            {
                flag = false;
                break;
            }
            m = gsl_rng_uniform(R_GLOBAL)*j;
            n = gsl_rng_uniform(R_GLOBAL)*j;
            vector<int> nodeM = contact[seq[m]];
            vector<int> nodeN = contact[seq[n]];
            cnt ++;

        }

        if(flag)
        {
            contact[seq[m]].push_back(seq[n]);
            contact[seq[n]].push_back(seq[m]);

            temp=seq[m];		 
            seq[m]=seq[j-1];
            seq[j-1]=temp;

            j--;

            temp=seq[n];
            seq[n]=seq[j-1];
            seq[j-1]=temp;

            j--;
        }
        else
        {
            j -= 2;
        }

        
	}
}

void Network::getSeed(int d)
{
    for(int i = 0; i < nodesNum; i ++)
    {
        if (nodes[i].degree == d)
        {
            seedList.push_back(i);
        }
    }
}

void Network::clearContact()
{
    contact.clear();
    seedList.clear();
    for(int i = 0; i < nodesNum; i ++)
    {
        contact.push_back(vector<int> ());
        nodes[i].degree = 0;
    }
}
// Network::Network(int nClass, int nAgent){
//     ageClass=nClass;
//     agentNum=nAgent;
// }

// void Network::individual_layer(Population &popu){  
//     //initial node id , age , idegree, heterogeneity    
//     for(int i=0; i<agentNum; i++){                          
//         Agent a;
//         nodes.push_back(a);
//         nodes[i].initial(i,popu.listAge[i],popu.listIdegree[i],0.5);
//     }

//     /* ---------CONFIGURATION NETWORK---------------- */
//     vector<vector<int>> stubArray;                                  //store free stubs of each age group
//     vector<int> enableAge;                                          //age group that still have stubs
//     int total=0;                                                    //total number of free stubs
//     int failure=0;                                                  //+1 when connect failed to break while loop

//     //initial stubArray:store agent id repeat degree times
//     for(int i=0; i<ageClass; i++)                                   
//         stubArray.push_back({});
//     for(int i=0; i<agentNum; i++){                                  
//         int id     = nodes[i].id;
//         int age    = nodes[i].age;
//         int degree = nodes[i].iDegree;
//         for (int j=0;j<degree;j++){
//             stubArray[age].push_back(id);
//         }        
//     }

//     //initial enableAge   
//     for(int i=0;i<ageClass;i++){
//         total+=stubArray[i].size();
//         enableAge.push_back(i);
//     }
            
//     //connect free stubs
//     while(total>0 && failure<kFailure){
//         int age1   = Uniform(enableAge.size());                         //random pick age group from enableAge
//             age1   = enableAge[age1];
//         int index1 = Uniform(stubArray[age1].size());                   //random pick one agent from the group
//         int id1    = stubArray[age1][index1];

//         double p = Prand();                                             //pick age gruop to according to pro
//         int age2 = 0;
//         while(p>popu.disAgei[age1][age2] && age2<ageClass-1)
//             age2++;
//         if(stubArray[age2].size()<1)                                    //fail when age gruop2 has not enough stubs 
//             failure++;
//         else{
//             int index2 = Uniform(stubArray[age2].size());
//             int id2    = stubArray[age2][index2];

//             //agent1 has no iCnt with agent2
//             if(find(nodes[id1].iCnt.begin(),nodes[id1].iCnt.end(),id2)==nodes[id1].iCnt.end()){
//                 if(age1==age2 && id1!=id2){
//                     nodes[id1].iCnt.push_back(id2);
//                     nodes[id2].iCnt.push_back(id1);
//                     stubArray[age1].erase(stubArray[age1].begin()+index1);
//                     if(index1<index2)
//                         stubArray[age2].erase(stubArray[age2].begin()+index2-1);
//                     else stubArray[age2].erase(stubArray[age2].begin()+index2);
//                     total-=2;
//                 }
//                 else if(age1!=age2){
//                     nodes[id1].iCnt.push_back(id2);
//                     nodes[id2].iCnt.push_back(id1);
//                     stubArray[age1].erase(stubArray[age1].begin()+index1);
//                     stubArray[age2].erase(stubArray[age2].begin()+index2);
//                     total-=2;
//                 }
//                 else failure++;

//                 //update enableAge: delete age group that has not enough stubs
//                 if(stubArray[age1].size()<1){
//                     for(vector<int>::iterator it=enableAge.begin();it!=enableAge.end();){
//                         if(*it==age1){
//                             it=enableAge.erase(it);
//                         }
//                         else ++it;              
//                     }
//                 }
//                 if(stubArray[age2].size()<1){
//                     for(vector<int>::iterator it=enableAge.begin();it!=enableAge.end();){
//                         if(*it==age2){
//                             it=enableAge.erase(it);
//                         }
//                         else ++it;              
//                     }
//                 }
//             }
//             else failure++;     
//         }
//     }

//     for(int i=0;i<agentNum;i++)    
//         nodes[i].iDegree=nodes[i].iCnt.size();           //update degree=iCnt size     
// }

