#include "model.h"

int spin_value(vector<int>& state, int op, int q){
    // s = sum(alpha_j * mu_j)
    int spin = 0;
    // Read state from left to right (s1 is the first value in the vector)
    int j = 0;
    while (op){
        // alpha_j + mu_j (last bit of mu is the remainder)
        spin += state[j] * (op % q);

        // Next bit (bit shift to the right by doing integer division)
        op /= q;
        j += 1;
    }
    return spin % q;
}

double entropy(vector<double>& prob_distr){
    // H(x) = - sum [p(x) log(px)]
    double entropy = 0;
    for (double p : prob_distr){
        // Ignore p == 0 because assume 0 log 0 = 0
        if (p){
            entropy -= (p * log2(p));
        }
    }
    return entropy;
}

double entropy_of_op(vector<vector<int>>& data, int op, int q){
    // Variable for probability distribution (# entries = # spin values/states)
    vector<double> prob_distr;
    for (int i = 0; i < q; ++i){
        prob_distr.push_back(0);
    }

    int s;
    for (vector<int>& obs : data){
        // Determine the value of the spin operator for a given observation
        s = spin_value(obs, op, q);
        // Increase the number of occurences of that spin value by 1
        prob_distr[s] += 1; 
    }

    // Normalize the distribution
    int N = data.size();
    for (int i = 0; i < q; ++i){
        prob_distr[i] /= N;
    }

    // Calculate the entropy
    return entropy(prob_distr);
}

pair<int, int> comb_ops(int op_nu, int op_mu, int q){
    pair<int, int> new_ops;
    // phi_nu * phi_mu
    int comb_op = 0;
    // phi_nu * phi_(-mu)
    int comb_op_cc = 0;

    // Base^(j-th bit)
    int factor = 1;

    int nu_j;
    int mu_j;
    while (op_nu || op_mu){
        // last bit of operator is the remainder
        nu_j = op_nu % q;
        mu_j = op_mu % q;

        comb_op += ((nu_j + mu_j)%q) * factor;
        // nu_j - mu_j could be negative -> add q and do another modulo q (C++ uses truncated division)
        comb_op_cc += ((((nu_j - mu_j)%q)+q)%q) * factor;

        // Bitshift to the left
        factor *= q;
        op_nu /= q;
        op_mu /= q;
    }
    new_ops.first = comb_op;
    new_ops.second = comb_op_cc;

    return new_ops;
}

pair<int, int> get_spin_ops_index(std::vector<int>& vars, std::vector<int>& interaction, int q){
    pair<int, int> indices{0,0};
    int n = vars.size();
    int power;
    for (int i = 0; i < n; ++i){
        power = pow(q, vars[i]);
        indices.first += interaction[i] * power;
        indices.second += ((-interaction[i] + q) % q) * power;
    }
    return indices;
}