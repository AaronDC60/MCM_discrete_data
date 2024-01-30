#pragma once

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <map>
#include <float.h>
#include <math.h>
#include <bitset>

using namespace std;

// Representation of the Minimally complex model
struct mcm {
    // Observed data
    vector<string> data;
    // Number of states
    int q;
    // Number of variables
    int n;
    // Number of observations
    int N;
    // Number of communities
    int n_comm;

    // Number of members per community
    vector<int> members;
    // Structure of the best MCM
    vector<int> best_mcm;
    // Evidence of the best MCM
    double best_evidence = -DBL_MAX;
};

// Data
mcm data_processing(string file);

// Evidence
map<string, int> count_observations(vector<string> data, int community);
double calc_evidence_icc(int community, mcm& model);
double calc_evidence(vector<int> partition, mcm& model);