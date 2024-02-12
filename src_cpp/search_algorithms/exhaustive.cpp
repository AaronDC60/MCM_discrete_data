#include "search.h"

void exhaustive_search(mcm& model){
    vector<int> partition;
    double log_evidence;
    // Initialize arrays to keep track of the next partition to generate
    int a[model.n];
    int b[model.n];
    for (int i = 0; i < model.n; i++){
        a[i] = 0;
        b[i] = 1;
    }
    // Tracker for first value that is different (from the right)
    int j = model.n - 1;

    while (j != 0){
        j = generate_next_partition(a, b, j, model.n);
        if (j == 0){
            // All possible partitions are generated
            break;
        }
        partition = convert_partition(a, model.n);
        log_evidence = calc_evidence(partition, model);
        // Check if this the best log evidence found so far
        if (log_evidence > model.best_evidence){
            // Update the current best
            model.best_evidence = log_evidence;
            model.best_mcm = partition;
        }
    }
}