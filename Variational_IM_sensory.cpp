//_________________________________________________________________
//  Variational_IM_sensory.cpp
//  Mex version - can be called locally in Matlab, once mexed.
//  Performs stochastic:
//      (I)  Encoding updates, &
//      (II) Decoding updates
//  for variational information maximization using
//  biologically plausible (local & on-line) updates (see Agakov).
//  This version does not require any feedback, i.e.
//  is a feedforward model of sensory neural coding areas.
//
//  Created & tested by adrianna on 20 November 2018.
//  Updated 11 January 2019.
//  Use Check_VIMperf.m matlab code to call & assess output.
//  Copyright © 2019 adrianna. All rights reserved.
//  Note: Class RNG & WTACircuit methods are defined here.
//_________________________________________________________________

#include "Variational_IM_sensory.h"

#include "matrix.h"
#include "mex.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <exception>
#include <random>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

template <typename T>
void writeOutputMatrix(int pos, vector<T> value, int N, int M, mxArray**& plhs) {
    mxArray* out_matrix = mxCreateDoubleMatrix(N,M,mxREAL);
    double* pr = mxGetPr(out_matrix);
    for (typename vector<T>::iterator it=value.begin(); it!=value.end(); ++it) {
        *pr++ = (double) *it;
    }
    plhs[pos] = out_matrix;
    return;
}

// *****************************************************************
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    // ** Command-Line Argument & Initializations: **
    int p            = 1;                  //p=dim(x), where x is the external variable to encode
    int N            = *mxGetPr(prhs[0]);  //PPC neuron population size
    double eta_b     = *mxGetPr(prhs[1]);  //Learning rate for b_i values (hyperparam)
    double eta_W     = *mxGetPr(prhs[2]);  //Learning rate for W_ki encoding weights (hyperparam)
    double eta_U     = *mxGetPr(prhs[3]);  //Learning rate for U_ki decoding weights (hyperparam)
    double* mexArray = mxGetPr(prhs[4]);   //Pointer to Matlab vector of kinematic data
    
    // ** Acquire # of Time Bins for Kinematic Data: **
    size_t nT = mxGetNumberOfElements(prhs[4]);
    
    // ** Instantiate PPC Model: **
    cout << "Initializing VIM object..." << endl;
    sensoryNNmodel VIM_obj(N, p, eta_b, eta_W, eta_U, mexArray, nT);  //instantiate
    
    // ** Find Params that Max Variational MI Over Observed Kinematic Data: **
    paramsStruct<double> learned_W_b_U = VIM_obj.train_via_stochasticVIM();
    cout << "Finished training via variational IM update protocol." << endl;
    
    // ** Return Params in Matlab Format: **
    vector<double>& W_learned = *learned_W_b_U.W_star.data();
    vector<double>& U_learned = *learned_W_b_U.U_star.data();
    writeOutputMatrix(0, W_learned, p, N, plhs);
    writeOutputMatrix(1, learned_W_b_U.b_star, N, 1, plhs);
    writeOutputMatrix(2, U_learned, p, N, plhs);
    writeOutputMatrix(3, learned_W_b_U.Converg_avgW, int(nT), 1, plhs);
    writeOutputMatrix(4, learned_W_b_U.Converg_avgb, int(nT), 1, plhs);
    writeOutputMatrix(5, learned_W_b_U.Converg_avgU, int(nT), 1, plhs);
    
    // ** Return Simulated PPC Population Activity: **
    vector<vector<int> > simZ = VIM_obj.return_simZ();
    string outfile2 = "VIM_simZ_N200.txt";
    ofstream z_output(outfile2);
    for (int n_t=0; n_t<simZ.size(); n_t++) {
        vector<int> z = simZ[n_t];
        for (int i=0; i<N; i++) {
            z_output << z[i] << " ";
        }
        z_output << endl;
    }
    z_output.close();
    
}
// *****************************************************************

// ************************** RNG Methods **************************
//Constructor:
RNG::RNG() {
    // Initialize mersenne twister RNG
    rng_pr = gsl_rng_alloc(gsl_rng_mt19937);
}
//Destructor:
RNG::~RNG() {
    gsl_rng_free(rng_pr);
}
//RNG::uniform
double RNG::uniform(double min, double max) {
    double u = gsl_rng_uniform(rng_pr);
    return ((max-min)*u)+min;
}
//RNG::discrete
int RNG::discrete(const vector<double>& p) {
    double u = gsl_rng_uniform(rng_pr);
    double c = p[0];
    int ix=0;
    while (c<u) {
        ix++;
        c += p[ix];
    }
    return ix;
}
//RNG::bernoulli
bool RNG::bernoulli(double p) {
    return (gsl_rng_uniform(rng_pr) < p);
}
//RNG::randperm
vector<int> RNG::randperm(int nmax) {
    //Description: This fn is analogous to Matlab's randperm(nmax)
    vector<int> nvals (nmax);
    for (int i=0; i<nmax; i++) {
        nvals[i] = i;
    }
    for (int i=0; i<nmax; i++) {
        // select random integer ix between i and nmax-1
        // swap i with ix
        unsigned long ix = i + gsl_rng_uniform_int(rng_pr, nmax-i);
        int tmp   = nvals[i];
        nvals[i]  = nvals[ix];
        nvals[ix] = tmp;
    }
    return nvals;
}
//RNG:gaussian
double RNG::gaussian(double sigma, double mu) {
    double g = gsl_ran_gaussian(rng_pr, sigma);
    return g+mu;
}
// *****************************************************************

// ******************* sensoryNNmodel Methods *********************
//Constructor:
sensoryNNmodel::sensoryNNmodel(int N, int p, double eta_b, double eta_W, double eta_U, double* mexArray, size_t nT) : N(N), p(p), eta_b(eta_b), eta_W(eta_W), eta_U(eta_U), nT(nT)
{
    rng = new RNG();
    
    //-- Instantiate constant hyperparameters: --
    mu_w    = 0;       //mean for initializing W_ki params
    sigma_w = 1.2;     //std for initializing W_ki params
    mu_u    = 0;       //mean for initializing U_ki params
    sigma_u = 1.2;     //std for initializing U_ki params
    
    //-- Load the behavioural (kinematic) time series data from input filename: --
    cout << "Loading in kinematic time series data..." << endl;
    for (int n_t=0; n_t<nT; n_t++) {
        xPos.push_back(mexArray[n_t]);
    }
    
    //-- Initialize W encoding weight matrix \in R^{p \times N}: --
    vector<double> W_int_vec; //prelim
    //const gsl_rng_type* T;  //can alternatively use
    //gsl_rng* r;
    //gsl_rng_env_setup();
    //T = gsl_rng_default;
    //r = gsl_rng_alloc(T);
    for (int ind=0; ind<(p*N); ind++) {
        W_int_vec.push_back(rng->gaussian(sigma_w, mu_w));
        //W_int_vec.push_back(mu_w + gsl_ran_gaussian(r, sigma_w));
        deltaW.push_back(0); //ensures cache is correct size
    }
    W.assign(W_int_vec,p,N);
    
    //-- Initialize baseline excitabilities, {b_i}, i \in [N]: --
    for (int ind=0; ind<N; ind++) {
        b.push_back(rng->gaussian(0.01, -1.6)); //(0.01,0.1)
        deltab.push_back(0); //ensures cache is correct size
    }
    
    //-- Initialize U decoding weight matrix \in R^{p \times N}:: --
    vector<double> U_int_vec; //prelim
    for (int ind=0; ind<(p*N); ind++) {
        U_int_vec.push_back(rng->gaussian(sigma_u, mu_u));
        deltaU.push_back(0); //ensures cache is correct size
    }
    U.assign(U_int_vec,p,N);
}

//sensoryNNmodel Destructor:
sensoryNNmodel::~sensoryNNmodel() {
    delete rng;
}

//sensoryNNmodel::train_via_stochasticVIM
paramsStruct<double> sensoryNNmodel::train_via_stochasticVIM() {
    paramsStruct<double> params_learned;
    vector<double> Converg_avgW(nT);
    vector<double> Converg_avgb(nT);
    vector<double> Converg_avgU(nT);
    
    for (int n_t=0; n_t<nT; n_t++) {               //n_t denotes the current time bin index
        //--
        //(0a) Compute current kinematic variable, x(t):
        current_x = xPos[n_t];
        
        //(0b) Compute current simulated PPC population activity vector, z(t):
        vector<int> z_pre;
        for (int i=0; i<N; i++) {
            int zi = compute_zi(i);
            z_pre.push_back(zi);
        }
        current_z = z_pre;      //assign
        Z.push_back(current_z); //cache
        
        //(0c) Compute Uz (constant for all i, k):
        double Uz = 0;
        for (int i=0; i<N; i++) {
            Uz += U.at(0, i) * current_z[i];
        }
        
        //** Apply Learning Updates: **//
        //Stochastic I-Step & M-Step Approximation:
        for (int i=0; i<N; i++) {
            double lambda_i = compute_lambda_i(i);
            double delta_bi = compute_dbi(i, lambda_i, Uz);
            deltab[i] = delta_bi;
            b[i] += delta_bi;
            for (int k=0; k<p; k++) {
                double delta_Wki = compute_dWki(k, i, lambda_i, Uz);
                deltaW[i*p + k] = delta_Wki;
                W.addto(k, i, delta_Wki);
                double delta_Uki = compute_dUki(k, i);
                deltaU[i*p + k] = delta_Uki;
                U.addto(k, i, delta_Uki);
            }
        }
        //** Computations to Check Convergence (After Simulation): **
        Converg_avgW[n_t] = calcAvg_deltaW();
        Converg_avgb[n_t] = calcAvg_deltab();
        Converg_avgU[n_t] = calcAvg_deltaU();
        
        cout << "Finished n_t = " << n_t << endl;
        //--
    } //end over time bins n_t
    
    //Assign learned parameters
    params_learned.W_star = W;
    params_learned.b_star = b;
    params_learned.U_star = U;
    params_learned.Converg_avgW = Converg_avgW;
    params_learned.Converg_avgb = Converg_avgb;
    params_learned.Converg_avgU = Converg_avgU;
    
    return params_learned;
}

//sensoryNNmodel:compute_zi
int sensoryNNmodel::compute_zi(int i) {
    double dp=0;
    for (int k=0; k<p; k++) {
        dp += W.at(k, i) * current_x;
    }
    double p_zi1 = logistic_fn(dp + b[i]);  //=p(z_i(t)=1|x(t))
    int zi = (rng->bernoulli(p_zi1))*2 - 1; //in {-1,1}
    return zi;
}

//sensoryNNmodel:compute_lambda_i
double sensoryNNmodel::compute_lambda_i(int i) {
    double dp=0;
    for (int k=0; k<p; k++) {
        dp += W.at(k, i) * current_x;
    }
    double p_zi1 = logistic_fn(dp + b[i]);
    double lambda_i = (2*p_zi1) - 1;
    return lambda_i;
}

//sensoryNNmodel::compute_dbi (current version works only for scalar kinematic var)
double sensoryNNmodel::compute_dbi(int i, double lambda_i, double Uz) {
    double t3 = current_x + (U.at(0, i)*current_z[i]) - Uz;
    double delta_bi = eta_b * (1-lambda_i) * U.at(0,i) * t3;
    return delta_bi;
}

//sensoryNNmodel::compute_dWki (current version works only for scalar kinematic var)
double sensoryNNmodel::compute_dWki(int k, int i, double lambda_i, double Uz) {
    double t3 = current_x + (U.at(0, i)*current_z[i]) - Uz;
    double delta_Wi = eta_W * (1-lambda_i) * current_x * U.at(0,i) * t3;
    return delta_Wi;
}

//sensoryNNmodel::compute_dUki (current version works only for scalar kinematic var)
double sensoryNNmodel::compute_dUki(int k, int i) {
    double t2 = 0;
    for (int j=0;j<N;j++) {
        if (j==i) {t2 += U.at(0, j);}
        else {t2 += U.at(0, j)*current_z[j]*current_z[i];}
    }
    double delta_Uki = eta_U * ( (current_x*current_z[i]) - t2 );
    return delta_Uki;
}

//sensoryNNmodel::calcAvg_deltaW
double sensoryNNmodel::calcAvg_deltaW() {
    double sum = 0.0;
    for(std::size_t i = 0; i < deltaW.size(); i++)
        sum += abs(deltaW.at(i));
    return sum/deltaW.size();
}

//sensoryNNmodel::calcAvg_deltab
double sensoryNNmodel::calcAvg_deltab() {
    double sum = 0.0;
    for(std::size_t i = 0; i < deltab.size(); i++)
        sum += abs(deltab.at(i));
    return sum/deltab.size();
}

//sensoryNNmodel::calcAvg_deltaU
double sensoryNNmodel::calcAvg_deltaU() {
    double sum = 0.0;
    for(std::size_t i = 0; i < deltaU.size(); i++)
        sum += abs(deltaU.at(i));
    return sum/deltaU.size();
}

//sensoryNNmodel::logistic_fn
double sensoryNNmodel::logistic_fn(double x) {
    double positive_cutoff = 10000;
    double negative_cutoff = -10000;
    if (x > positive_cutoff) return 1;
    if (x < negative_cutoff) return 0;
    return 1 / (1 + exp(-x));
}

//sensoryNNmodel::return_simZ
vector<vector<int> > sensoryNNmodel::return_simZ() {
    return Z;
}

// *****************************************************************
