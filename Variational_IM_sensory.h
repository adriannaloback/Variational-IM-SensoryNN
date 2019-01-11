//_________________________________________________________________
//  Variational_IM_sensory.h
//  Header file for Variational_IM_sensory, which performs stochastic
//      (I)  Encoding updates, &
//      (II) Decoding updates
//  for variational information maximization using
//  biologically plausible (local & on-line) updates (see Agakov),
//  to simulate afferent neural activity responses.
//  This version does not require any feedback. 
//  Copyright © 2019 adrianna. All rights reserved.
//_________________________________________________________________

#ifndef Variational_IM_sensory_h
#define Variational_IM_sensory_h

#include <iostream>
#include <gsl/gsl_rng.h>
#include <vector>
#include <string>
#include <map>

using namespace std;

// *********************** myMatrix ****************************
template <class T>
class myMatrix
{
public:
    myMatrix();
    myMatrix(vector<T>&, int, int);
    void assign(vector<T>&, int, int);
    int get_m();
    int get_N();
    
    const T& at(const int k, const int i) const;       //Subscripted index
    const T& at(const int k) const;                    //Linear index
    void addto(const int k, const int i, T a);         //Add a to the (k,i)-th entry
    void assign_entry(const int k, const int i, T a);  //Assign value a to the (k,i)-th entry
    
    vector<T>* data();
    
private:
    vector<T> matrix_data;
    int m,N; // m = #rows, N = #columns
};

// -- myMatrix Definition: --
template <class T>
myMatrix<T>::myMatrix() : m(0),N(0) {}; //Constructor def

template <class T>
myMatrix<T>::myMatrix(vector<T>& _data, int _m, int _N) : m(_m), N(_N) {
    if (_data.size() != _m*_N) {
        cerr << "Matrix dimensions must agree." << endl;
        m = 0; N = 0;
    } else {
        matrix_data = _data;
    }
}

template <class T>
void myMatrix<T>::assign(vector<T>& _data, int _m, int _N) {
    if (_data.size() != _m*_N) {
        cerr << "Matrix dimensions must agree." << endl;
        m = 0; N = 0;
    } else {
        m = _m; N = _N;
        matrix_data = _data;
    }
    return;
}

template <class T>
int myMatrix<T>::get_m() { return m; }

template <class T>
int myMatrix<T>::get_N() { return N; }

template <class T>
const T& myMatrix<T>::at(const int k, const int i) const {
    return matrix_data[i*m + k];
}

template <class T>
void myMatrix<T>::addto(const int k, const int i, T a) {
    matrix_data[i*m + k] += a;
}

template <class T>
void myMatrix<T>::assign_entry(const int k, const int i, T a) {
    matrix_data[i*m + k] = a;
}

template <class T>
vector<T>* myMatrix<T>::data() { return &matrix_data; }
// *************************************************************

// ********************* paramsStruct **************************
template<typename T>
struct paramsStruct
{
    myMatrix<T> W_star;       //the learned encoding weights
    vector<T> b_star;         //the learned b_k* values
    myMatrix<T> U_star;       //the learned decoding weights
    vector<T> Converg_avgW;   //to assess convergence
    vector<T> Converg_avgb;   //to assess convergence
    vector<T> Converg_avgU;   //to assess convergence
};
// *************************************************************

// ************************** RNG ******************************
class RNG
{
public:
    RNG();
    ~RNG();
    double uniform(double min, double max);
    int discrete(const vector<double>&);
    bool bernoulli(double);
    vector<int> randperm(int);
    double gaussian(double sigma, double mu);
private:
    gsl_rng* rng_pr;
};
// *************************************************************

// ************************* Spike *****************************
struct Spike
{
    double time; 
    int bin;
    int neuron_ind;
};
// *************************************************************

// ********************* Sensory NN Encoding Model (Maximise variational IM l.b.) ***********************
class sensoryNNmodel
{
public:
    sensoryNNmodel(int N, int p, double _etab, double _etaW, double _etaU, double* mx, size_t nT);
    paramsStruct<double> train_via_stochasticVIM();
    vector<vector<int> > return_simZ();
    double logistic_fn(double x);
    ~sensoryNNmodel();
protected:
    //Constant Hyperparams:
    int N;           //PPC population size (# of PPC neurons = dim(z))
    int p;           //p=dim(x)
    size_t nT;       //# of time bins of kinematic time series data
    double mu_w;     //mean for initializing encoding weights
    double sigma_w;  //std for initializing encoding weights
    double mu_u;     //mean for initializing decoding weights
    double sigma_u;  //std for initializing decoding weights
    double eta_b;    //learning rate for encoding b_k terms (hyperparam)
    double eta_W;    //learning rate for encoding W_ki terms (hyperparam)
    double eta_U;    //learning rate for decoding U_ki terms (hyperparam)
    RNG* rng;
    
    //Dynamic/learned params:
    vector<double> b;             //encoding baseline params
    myMatrix<double> W;           //encoding weight params
    myMatrix<double> U;           //decoding weight params
    
    //Data-handling:
    double current_x;             //x(t) = kinematic vars at current time t
    vector<double> xPos;          //will cache all kinematic time series values
    
    //Simulated PPC population activity:
    vector<int> current_z;        //=z(t) \{-1,1}^N here
    vector<vector<int> > Z;       //cache of all simulated population activity
    
    //Training via Stochastic (Local) Variational IM:
    //double EPSP_kernel(double delta, double binsize);
    int    compute_zi(int i);
    double compute_lambda_i(int i);
    double compute_dbi(int i, double lambda_i, double Uz);
    double compute_dWki(int k, int i, double lambda_i, double Uz);
    double compute_dUki(int k, int i);
    
    //To check convergence of learning (both encoding & decoding):
    vector<double> deltaW;        //will cache for each i & k (will use to take mean)
    vector<double> deltab;        //will cache for each i (will use to take mean)
    vector<double> deltaU;        //will cache for each i & k (will use to take mean)
    double calcAvg_deltaW();      //for convergence tracking
    double calcAvg_deltab();      //for convergence tracking
    double calcAvg_deltaU();      //for convergence tracking
};
// *************************************************************

#endif /* Variational_IM_sensory_h */
