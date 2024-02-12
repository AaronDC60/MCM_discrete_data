#include "model.h"

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

string community_as_string(int community, int n){
    // String representation of community
    string comm(n, '0');
    for (int j = 0 ; j < n; j++){
        if (community & 1){
            comm[j] = '1';
        }
        community >>= 1;
    }
    return comm;
}

void print_partition_to_file(ofstream& file, vector<int>& partition){
    int n = partition.size();
    int i = 0;
    for (int community : partition){
        // Ignore empty communities
        if (! community){continue;}
        file << "Community " << i << " : " << community_as_string(community, n) << endl;
        i++;
    }
}

void print_partition_to_terminal(vector<int>& partition){
    // Number of variables
    int n = partition.size();
    // Counter for community number
    int i = 0;
    for (int community : partition){
        // Ignore empty communities
        if (! community){continue;}
        // String representation of community
        string comm(n, '0');
        for (int j = 0 ; j < n; j++){
            if (community & 1){
                comm[j] = '1';
            }
            community >>= 1;
        }
        cout << "Community " << i << " : " << comm << endl;
        i++;
    }
    cout << "\n";
}

int community_size(int community){
    int size = 0;
    while(community){
        // Check if last bit is equal to 1
        if (community & 1){
            size += 1;
        }
        // Bitshift to the right
        community >>= 1;
    }
    return size;
}