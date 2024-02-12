#include "search.h"

void greedy_search(mcm& model){
    // Start from the independent model
    int n_partitions = model.n;
    vector<int> partition;
    partition.assign(n_partitions, 0);
    int element = 1;
    for (int i = 0; i < n_partitions; i++){
        partition[i] += element;
        element <<= 1;
    }

    // Write to file
    model.greedy_file << "Start greedy merging procedure \n" << endl;
    print_partition_to_file(model.greedy_file, partition);

    // Variables to store the calculated evidences
    double evidence_i;
    double evidence_j;
    double evidence_diff;

    // Variables to store the best values
    int best_i;
    int best_j;
    double best_evidence_diff = 1;

    // Size of the communities to merge
    int r_i;
    int r_j;

    while (best_evidence_diff > 0){
        best_evidence_diff = 0;
        for (int i = 0; i < n_partitions; i++){
            if (partition[i] == 0){continue;}
            r_i = community_size(partition[i]);
            evidence_i = calc_evidence_icc(partition[i], model, r_i);
            for (int j = i+1; j < n_partitions; j++){
                if (partition[j] == 0){continue;}
                r_j = community_size(partition[j]);
                evidence_j = calc_evidence_icc(partition[j], model, r_j);
                // Calculate difference in evidence between merged and separate partitions
                evidence_diff = calc_evidence_icc(partition[i] + partition[j], model, r_i + r_j) - evidence_i - evidence_j;
                // Check if this is the best merge so far
                if (evidence_diff > best_evidence_diff){
                    best_evidence_diff = evidence_diff;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        // Check if merging increased evidence
        if (best_evidence_diff == 0){
            break;
        }
        else{
            // Merge the two communities that results in the biggest increase in evidence
            partition[best_i] += partition[best_j];
            partition[best_j] = 0;

            // Write to file
            model.greedy_file << "\nMerging communities " << best_i << " and " << best_j << " Evidence difference: "<<  best_evidence_diff << endl;
            print_partition_to_file(model.greedy_file, partition);
        }
    }
    // Store the best MCM found using the greedy merging scheme
    model.best_mcm = partition;
    model.best_evidence = calc_evidence(partition, model);
}