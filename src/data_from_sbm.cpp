#include "stoch_block_model/sbm.h"

int main(){
    std::string file = "test_data";
    int q = 3;
    int n = 10;
    int N = 10000;

    // 1111100000, 0000011111
    std::vector<int> comm_structure = {31,992};

    // Interactions
    std::vector<int> interaction = {1,1};
    std::complex<double> strength = 0.25;

    // Model parameters
    std::vector<std::complex<double>> g;

    // Link probabilities
    double p_in;
    double p_out;

    // Number of repeats
    int n_runs = 50;

    // File name to write the data to
    std::string file_prefix = "../../data/sbm/N_" + std::to_string(N) + "_g11/g_" + std::to_string(static_cast<int>(strength.real() * 100)) + "/sbm_";
    std::string file_name;

    for (int i = 0; i < 11; ++i){
        p_in = i/10.0;
        for (int j = 0; j < 11; ++j){
            p_out = j/10.0;
            for (int r = 0; r < n_runs; ++r){
                file_name = file_prefix + "p_in_" + std::to_string(i*10) + "_p_out_" + std::to_string(j*10) + "_n10_q3_run_" + std::to_string(r);
                // Create spinmodel
                g = create_graph(comm_structure, p_in, p_out, interaction, strength, q, n);
                generate_samples(g, N, file_name, q, n);
            }
        }
    }
}