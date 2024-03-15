#pragma once

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <float.h>
#include <math.h>
#include <bitset>

using namespace std;

// Representation of the Minimally complex model
struct mcm {
    // Observed data
    vector<vector<int>> data;
    // Number of variables
    int n;
    // Number of observations
    int N;
    // Number of states
    int q;
    // Powers of q
    vector<int> pow_q;

    // Output file to store search steps
    bool log_file;
    ofstream greedy_file;
    ofstream divide_and_conquer_file;

    // Structure of the best MCM
    vector<vector<int>> best_mcm;
    // Evidence of the best MCM
    double best_evidence = -DBL_MAX;
};

// Data
vector<vector<int>> data_processing(string file, int n);

// Model
mcm create_model(vector<vector<int>> data, int q, int n, bool log_file);

// Evidence
unordered_map<int, int> count_observations(mcm& model, int community);
double calc_evidence_icc(int community, mcm& model, int r);
double calc_evidence(vector<int> partition, mcm& model);

// Partition
string community_as_string(int community, int n);
int community_size(int community);
vector<int> convert_partition(int* a, int n);
void print_partition_to_terminal(vector<int>& partition);
void print_partition_to_file(ofstream& file, vector<int>& partition);