#include "model.h"

bool comp_entropy(pair<int, double> op1, pair<int, double> op2){
    return op1.second < op2.second;
}

void gt_state(vector<int>& state, vector<int>& gt, int q, int n){
    // Variable for transformed state
    vector<int> new_state;
    // Index for the bit
    int j;
    // Spin value
    int alpha_j;
    // Spin operator
    int op;
    for (int i = 0; i < n; ++i){
        // Read from right to left
        j = n-1;
        alpha_j = 0;
        op = gt[i];

        while (op){
            // Last bit of the operator is the remainder
            alpha_j += state[j] * (op % q);
            // Bitshift to the left
            op /= q;
            j -= 1;
        }
        //new_state[i] = alpha_j % q;
        new_state.push_back(alpha_j % q);
    }
    state = new_state;
}

void transform_data(vector<vector<int>>& data, vector<int>& gt, int q, int n){
    // Loop over all observations and perform a gauge transformation on each observations
    for (vector<int>& obs : data){
        gt_state(obs, gt, q, n);
    }
}

void find_best_basis(mcm& model){
    // Keep track of all combinations that can be performed with the operators in the IM to check the independence
    vector<int> all_comb = {0};
    // Number of operators (q^n)
    int n_op = model.pow_q.back() * model.q;

    // Calculate and store the entropy of all operators
    vector<pair<int, double>> entropy_of_ops;
    pair<int, double> entropy;
    for (int op = 1; op < n_op; ++op){
        entropy.first = op;
        entropy.second = entropy_of_op(model.data, op, model.q);
        entropy_of_ops.push_back(entropy);
    }

    // Sort the operators based on entropy from low to high
    sort(entropy_of_ops.begin(), entropy_of_ops.end(), comp_entropy);

    // Search the n most biased operators that are independent
    pair<int, int> new_ops;
    for (pair<int, double> op_mu : entropy_of_ops){
        // Keep track of the size at the start of the loop for deleting new combinations with dependent operators
        int n_comb = all_comb.size();
        for (int i = 0; i < n_comb; ++i){
            int op_nu = all_comb[i];
            // Combine phi_nu with phi_mu and phi_(-mu)
            new_ops = comb_ops(op_nu, op_mu.first, model.q);
            // Check if they are independent (combination != 0)
            if (!new_ops.first || !new_ops.second){
                all_comb.resize(n_comb);
                break;
            }
            // If independent, add new combinations
            all_comb.push_back(new_ops.first);
            all_comb.push_back(new_ops.second);
        }
        // If all combinations are independent add operator to basis
        if (!new_ops.first || !new_ops.second){continue;}
        model.best_basis.push_back(op_mu.first);

        // Stop search if n operators are found
        if (model.best_basis.size() == model.n){break;}
    }
}