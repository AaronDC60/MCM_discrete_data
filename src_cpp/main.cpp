#include "model/model.h"
#include "search_algorithms/search.h"

#include <chrono>


int main(){
    // Should be user input
    string file = "../../data/SC_voting/US_SupremeCourt_n9_N895.txt";
    int q = 2;

    mcm model;
    model = data_processing(file);
    model.q = q;

    // Create output file
    ofstream outputFile("output.dat");

    // Exhaustive search
    auto start = std::chrono::high_resolution_clock::now();
    exhaustive_search(model);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    outputFile << "Exhaustive search: " << duration.count() << "ms \n" << endl;

    outputFile << "Best MCM: " << endl;
    print_partition_to_file(outputFile, model.best_mcm);
    outputFile << "Best log-evidence: " << model.best_evidence << "\n" << endl;

    // Greedy search
    model.greedy_file.open("greedy_search.dat");
    start = std::chrono::high_resolution_clock::now();
    greedy_search(model);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    model.greedy_file.close();
    outputFile << "Greedy search: " << duration.count() << "ms \n" << endl;    

    outputFile << "Best MCM: " << endl;
    print_partition_to_file(outputFile, model.best_mcm);
    outputFile << "Best log-evidence: " << model.best_evidence << "\n" <<endl;

    // Divide and conquer
    model.divide_and_conquer_file.open("divide_and_conquer.dat");
    start = std::chrono::high_resolution_clock::now();
    divide_and_conquer(model);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    model.divide_and_conquer_file.close();
    outputFile << "Divide and conquer: " << duration.count() << "ms \n" << endl;    

    outputFile << "Best MCM: " << endl;
    print_partition_to_file(outputFile, model.best_mcm);
    outputFile << "Best log-evidence: " << model.best_evidence << "\n" << endl;

    outputFile << "Search done" << endl;
    outputFile.close();
    return 0;
}

