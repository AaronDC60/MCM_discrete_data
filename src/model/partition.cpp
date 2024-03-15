#include "model.h"

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
