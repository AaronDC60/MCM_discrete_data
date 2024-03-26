#include <cstdlib> 
#include <complex>
#include <cmath>

#include "../model/model.h"

// Stochastic block model
bool check_in_or_out(std::vector<int>& comm_structure, std::vector<int>& vars);
int find_index(std::vector<int>& set, int index, int max_value);
std::vector<std::complex<double>> create_graph(std::vector<int>& comm_structure, double p_in, double p_out, std::vector<int>& interaction, std::complex<double> strength, int q, int n);

// Generating data
std::string int_to_string(int integer, int q, int n);
std::vector<std::complex<double>> fwht(std::vector<std::complex<double>> g, int q);
void generate_samples(std::vector<std::complex<double>> g, int N, std::string file, int q, int n);

