#include "sbm.h"

bool check_in_or_out(std::vector<int>& comm_structure, std::vector<int>& vars){
    int n_comms = comm_structure.size();
    int n_vars = vars.size();
    // Retrun True if there is only 1 variable
    if(n_vars == 1){return true;}
    // Check the community of the first variable
    int comm = 0;
    // First variable as bitstring
    int var_1 = 1 << vars[0];

    for (int i = 0; i < n_comms; ++i){
        // Check if the bit of the first variable is present in the ith community
        if (comm_structure[i] & var_1){
            comm = i;
        }
    }

    // Check if all the other variables are in the same community
    int var;
    for (int i = 1; i < n_vars; ++i){
        // ith variable as bitstring
        var = 1 << vars[i];
        if (! (comm_structure[comm] & var)){
            return false;
        }
    }
    return true;
}

int find_index(std::vector<int>& set, int index, int max_value){
    while(set[index] == max_value){
        index -= 1;
        max_value -= 1;

        // Break if all element have reached their max value (index will be -1)
        if(index == -1){break;}
    }
    return index;
}

std::vector<std::complex<double>> create_graph(std::vector<int>& comm_structure, double p_in, double p_out, std::vector<int>& interaction, std::complex<double> strength, int q, int n){
    // Model parameters (q^n values, initially all switched of)
    std::vector<std::complex<double>> g(pow(q, n), 0);
    // Variable for the indices in vector of model parameters for interactions
    pair<int, int> indices;
    // Uniform random variable
    double x;
    // Variable for probability
    double p;
    // Interaction order
    int k = interaction.size();
    // Variables in the first interaction of order k (variables 0 to k)
    std::vector<int> set;
    for (int i = 0; i < k; ++i){
        set.push_back(i);
    }
    // Start index at last element
    int index = k - 1;
    // Max value is the last variables
    int max_value = n - 1;

    // Iterate over all possible combinations of order k between n variables (n choos k)
    int j;
    while(true){
        // Check if the variables in the set are from the same community or not
        if(check_in_or_out(comm_structure, set)){
            // Form a link with probability p_in
            p = p_in;
        }
        else{
            // Form a link wit probability p_out
            p = p_out;
        }
        // Generate uniform random variable between 0 and 1
        x = ((double) rand() / (RAND_MAX));
        if (x < p){
            // Form a link -> Set the model parameter in the spin model
            indices = get_spin_ops_index(set, interaction, q);
            g[indices.first] = strength;
            g[indices.second] = std::conj(strength);
        }

        // Generate next interaction
        if(set[index] == max_value){
            j = find_index(set, index, max_value);
            // All combinations are generated
            if(j == -1){break;}
            // Increase all the values from j to the end by 1 to have the next set
            set[j] += 1;
            for(int i = j+1; i < k; ++i){
                set[i] = set[i-1] + 1;
            }
        }
        else{
            // Increase the last element by 1
            set[index] += 1;
        }
    }
    return g;
}

