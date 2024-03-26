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

    // Storage of log-evidence for communities
    unordered_map<int, double> evidence_storage;

    // Output file to store search steps
    bool log_file;
    ofstream greedy_file;
    ofstream divide_and_conquer_file;

    // N most biased operators (best basis/ IM)
    vector<int> best_basis;

    // Structure of the best MCM(s)
    vector<vector<int>> best_mcm;
    // Evidence of the best MCM
    double best_evidence = -DBL_MAX;
};

// Data
vector<vector<int>> data_processing(string file, int n);

// Model
mcm create_model(vector<vector<int>>& data, int q, int n, bool log_file);

// Evidence
unordered_map<int, int> count_observations(mcm& model, int community);
double get_evidence_icc(int community, mcm& model);
double calc_evidence_icc(int community, mcm& model, int r);
double calc_evidence(vector<int> partition, mcm& model);

// Partition
string community_as_string(int community, int n);
int community_size(int community);
vector<int> convert_partition(int* a, int n);
void print_partition_to_terminal(vector<int>& partition);
void print_partition_to_file(ofstream& file, vector<int>& partition);

// Spin operators
int spin_value(vector<int>& state, int op, int q);
double entropy(vector<double>& prob_distr);
double entropy_of_op(vector<vector<int>>& data, int op, int q);
pair<int, int> comb_ops(int op_nu, int op_mu, int q);
pair<int, int> get_spin_ops_index(std::vector<int>& vars, std::vector<int>& interaction, int q);

// Gauge transformation
bool comp_entropy(pair<int, double> op1, pair<int, double> op2);
void gt_state(vector<int>& state, vector<int>& gt, int q, int n);
void transform_data(vector<vector<int>>& data, vector<int>& gt, int q, int n);
void find_best_basis(mcm& model);

