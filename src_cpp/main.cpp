#include "model/model.h"
#include "search_algorithms/search.h"

#include <chrono>

int main(){
    // Should be user input
    string file = "../data/SC_voting/US_SupremeCourt_n9_N895.txt";
    int q = 2;

    mcm model;

    // Divide and conquer scheme
    model = data_processing(file);
    model.q = q;

    auto start = std::chrono::high_resolution_clock::now();
    divide_and_conquer_setup(model);
    cout << "Best MCM: " << endl;
    cout << model.best_mcm[0] << endl;
    cout << model.best_mcm[1] << endl;
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Divide and conquer (microseconds): " << duration.count() << endl; 

    // Greedy search
    model = data_processing(file);
    model.q = q;

    start = std::chrono::high_resolution_clock::now();
    greedy_search(model);
    cout << "Best MCM: " << endl;
    cout << model.best_mcm[0] << endl;
    cout << model.best_mcm[1] << endl;
    stop = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Divide and conquer (microseconds): " << duration.count() << endl; 

    cout << "Done" << endl;

    return 0;
}