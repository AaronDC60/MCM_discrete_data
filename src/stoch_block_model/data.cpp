#include "sbm.h"

std::string int_to_string(int integer, int q, int n){
    std::string state(n, '0');
    int i = 0;
    while(integer){
        // Last bit of integer in bas q is the remainder
        state[i] += integer % q;

        // Next bit (bit shift to the right by doing integer division)
        integer /= q;
        ++i;
    }
    return state;
}

std::vector<std::complex<double>> fwht(std::vector<std::complex<double>> g, int q){
    int n = g.size();
    std::vector<std::complex<double>> wht_g(g);

    // Compute the factors (q different spin values)
    std::vector<complex<double>> factors(q);
    // 2pi/q
    std::complex<double> complex_factor(2 * M_PI / q, 0);
    // Imaginary unit
    std::complex<double> imag_unit(0,1);
    for (int i = 0; i < q; ++i){
        factors[i] = std::exp(std::complex<double>(i,0) * complex_factor * imag_unit);
    }

    std::vector<std::complex<double>> tmp;
    int h = 1;
    int increment;
    while(h < n){
        // Make hard copy
        tmp.assign(wht_g.begin(), wht_g.end());
        // Set all values back to zero./t
        std::fill(wht_g.begin(), wht_g.end(), 0);
        // Step size in the outer loop of the algorithm
        increment = h * q;
        for (int i = 0; i < n; i+=increment){
            for (int j = i; j < (i+h); ++j){
                for (int k = 0; k < q; ++k){
                    for (int l = 0; l < q; ++l){
                        wht_g[j + k*h] += tmp[j + l*h] * factors[(l*k)%q];
                    }
                }
            }
        }
        h *= q;
    }
    return wht_g;
}

void generate_samples(std::vector<std::complex<double>> g, int N, std::string file, int q, int n){
    // Number of states
    int n_states = g.size();

    // Construct the cumulative distribution function
    std::vector<double> cdf(n_states);
    // FWHT of the model parameters
    std::vector<std::complex<double>> wht_g = fwht(g, q);

    double total = 0;
    for (int i = 0; i < n_states; ++i){
        total += std::exp(wht_g[i].real());
        cdf[i] = total;
    }
    // Normalize the distribution
    for (int i = 0; i < n_states; ++i){
        cdf[i] /= total;
    }

    std::ofstream outputFile(file + ".dat");
    // Sample states
    double x;
    int state;
    for (int i = 0; i < N; ++i){
        // Generate uniform random value
        x = ((double) rand() / (RAND_MAX));
        // Determine state by checking where in the cdf x lies
        state = 0;
        while (x > cdf[state]){++state;}
        outputFile << int_to_string(state, q, n) << '\n';
    }
    outputFile.close();
}