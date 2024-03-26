#include "model.h"

mcm create_model(vector<vector<int>>& data, int q, int n, bool log_file){
    // Create object
    mcm model;
    model.data = data;
    // Determine the number of observations
    model.N = data.size();
    // Number of variables
    model.n = n;
    // Number of states
    model.q = q;
    // Precompute the powers of q
    model.pow_q.assign(n, 0);
    int element = 1;
    for(int i = 0; i < n; i++){
        model.pow_q[i] = element;
        element *= q;
    }
    // Indicate if search steps must be written to log file
    model.log_file = log_file;
    
    return model;
}