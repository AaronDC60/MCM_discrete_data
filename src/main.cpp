#include "model/model.h"
#include "search_algorithms/search.h"

#include <chrono>


int main(int argc, char* argv[]){
    // Process user input

    // Information about data
    string file;
    int q = 0;
    int n = 0;

    // Search method
    bool log_file = false;
    bool exhaustive = false;
    bool greedy = false;
    bool div_and_conq = false;

    string arg;
    for (int i = 0; i < argc; i++) {
        arg = argv[i];
        
        // File
        if (arg == "-f"){
            file = argv[i+1];
        }
        // Number of variables
        if (arg == "-n"){
            n = stoi(argv[i+1]);
        }
        // Number of states
        if (arg == "-q"){
            q = stoi(argv[i+1]);
        }
        // Log files to store the steps in the search process
        if (arg == "-l"){
            log_file = true;
        }
        // Search method
        if (arg == "-es"){
            exhaustive = true;
        }
        if (arg == "-gs"){
            greedy = true;
        }
        if (arg == "-dc"){
            div_and_conq = true;
        }
    }
    // Number of states and number of variables are mandatory
    if (!q){
        cout << "Argument for number of states (-q) is missing." << endl;
        return 0;
    }
    if (!n){
        cout << "Argument for number of variables (-n) is missing." << endl;
        return 0;
    }

    string path = "../../data/" + file + ".dat";

    // Read in data
    vector<vector<int>> data;
    data = data_processing(path, n);
    // Construct mcm model
    mcm model = create_model(data, q, n, log_file);

    // Create output file
    ofstream outputFile(file + "_output.dat");

    if (exhaustive){
        // Exhaustive search
        auto start = std::chrono::high_resolution_clock::now();
        exhaustive_search(model);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        outputFile << "Exhaustive search: " << duration.count() << "ms \n" << '\n';
        outputFile << "Number of best MCMs found : " << model.best_mcm.size() << "\n\n";
        outputFile << "Best MCM(s): " << endl;
        for(int i = 0; i < model.best_mcm.size(); ++i){
            print_partition_to_file(outputFile, model.best_mcm[i]);
            outputFile << "\n";
        }
        outputFile << "Best log-evidence: " << model.best_evidence << "\n" << '\n';
    }

    if (greedy){
        // Greedy search
        model.best_mcm.clear();
        if(log_file){
            model.greedy_file.open(file + "_greedy_search.dat");
        }

        auto start = std::chrono::high_resolution_clock::now();
        greedy_search(model);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        if(log_file){
            model.greedy_file.close();
        }
        outputFile << "Greedy search: " << duration.count() << "ms \n" << endl;    

        outputFile << "Best MCM: " << endl;
        print_partition_to_file(outputFile, model.best_mcm[0]);
        outputFile << "Best log-evidence: " << model.best_evidence << "\n" <<endl;
    }

    if (div_and_conq){
        // Divide and conquer
        model.best_mcm.clear();
        if(log_file){
            model.divide_and_conquer_file.open(file + "_divide_and_conquer.dat");
        }

        auto start = std::chrono::high_resolution_clock::now();
        divide_and_conquer(model);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        if(log_file){
            model.divide_and_conquer_file.close();
        }
        outputFile << "Divide and conquer: " << duration.count() << "ms \n" << endl;    

        outputFile << "Best MCM: " << endl;
        print_partition_to_file(outputFile, model.best_mcm[0]);
        outputFile << "Best log-evidence: " << model.best_evidence << "\n" << endl;
    }

    // Close output file
    outputFile << "Search done" << endl;
    outputFile.close();
    return 0;
}

