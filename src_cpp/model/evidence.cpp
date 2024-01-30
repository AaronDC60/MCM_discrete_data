#include "model.h"

// Count all the different observations in the dataset for a given community
map<string, int> count_observations(vector<string> data, int community){
    // Trackers for the bit count
    int i;
    int j;
    map<string, int> counts;
    // Loop over the entire dataset
    for (string obs : data){
        // Extract the substring corresponding to the community
        string sub_obs;
        i = 0;
        j = community;
        while(j){
            if (j & 1){
                sub_obs += obs[i];
            }
            i++;
            j >>= 1;
        }
        // Increase count of the substring
        counts[sub_obs] += 1;   
    }
    return counts;
}

double calc_evidence_icc(int community, mcm& model){
    double log_evidence = 0;
    // Contributions from the different observations
    map<string, int> counts = count_observations(model.data, community);
    map<string, int>::iterator count_iter = counts.begin();
    while (count_iter != counts.end()){
        log_evidence += (lgamma(count_iter->second + 0.5) - 0.5 * log(M_PI));
        count_iter++;
    }
    // Calculate prefactor
    int r = counts.begin()->first.size();
    log_evidence += lgamma(pow(model.q, r)/2) - lgamma(model.N + pow(model.q, r)/2);
    return log_evidence;
}

double calc_evidence(vector<int> partition, mcm& model){
    double log_evidence = 0;
    // Iterate over all the ICCs in the partition
    for (int community : partition){
        if (community){
            log_evidence += calc_evidence_icc(community, model);
        }
    }
    return log_evidence;
}


