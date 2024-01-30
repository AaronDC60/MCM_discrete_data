#include "search.h"

int find_j(int* a, int* b, int n){
    // Find the first element from right to left that is different in a and b
    int j = n-2;
    while(a[j] == b[j]){j--;}
    return j;
}

int generate_next_partition(int* a, int* b, int j, int n){
    // Compare the last bit
    if (a[n-1] != b[n-1]){
        // Increase the last bit of 'a' by 1 to generate new partition
        a[n-1] += 1;
        return 1;
    }
    // Find the first bit that is different (starting from the right)
    j = find_j(a, b, n);
    if (j == 0){
        // All bits are the same, all possible partitions are generated
        return 0;
    }
    // Increase the first bit from the right in 'a' that is different from 'b' by 1
    a[j] += 1;
    if (a[j] == b[j]){
        // Change the number of partitions the next member is allowed to be in
        b[j+1] = b[j] + 1;
    }
    // Set values for all member to the right
    int i = j+1;
    while (i < n){
        // Move member back to the first partition
        a[i] = 0;
        // Change the number of partitions this member is allowed to be in
        b[i] = b[j+1];
        i++;
    }
    return 1;
}

vector<int> convert_partition(int* a, int n){
    vector<int> partition;
    partition.assign(n, 0);
    int element = 1;
    for (int i = 0; i < n; i++){
            partition[a[i]] += element;
            element <<= 1;
        }
    return partition;
}

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