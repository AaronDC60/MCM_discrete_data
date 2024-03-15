#include "model.h"

// Count all the different observations in the dataset for a given community
map<int, int> count_observations(mcm& model, int community){
    // Trackers for the bit count
    int i;
    int j;
    map<int, int> counts;
    // Loop over the entire dataset
    for (const vector<int>& obs : model.data){
        // Extract the substring corresponding to the community
        int state = 0;
        i = 0;
        j = community;
        while(j){
            if (j & 1){
                //state += (obs[i] * pow(model.q, i));
                state += (obs[i] * model.pow_q[i]);
            }
            i++;
            j >>= 1;
        }
        // Increase count of the substring
        counts[state] += 1;   
    }
    return counts;
}

double calc_evidence_icc(int community, mcm& model, int r){
    double log_evidence = 0;
    // Contributions from the different observations
    map<int, int> counts = count_observations(model, community);
    map<int, int>::iterator count_iter = counts.begin();
    while (count_iter != counts.end()){
        log_evidence += (lgamma(count_iter->second + 0.5) - 0.5 * log(M_PI));
        count_iter++;
    }
    // Calculate prefactor
    log_evidence += lgamma(pow(model.q, r)/2) - lgamma(model.N + pow(model.q, r)/2);
    return log_evidence;
}

// Input pass by reference
double calc_evidence(vector<int> partition, mcm& model){
    double log_evidence = 0;
    int r;
    // Iterate over all the ICCs in the partition
    for (int community : partition){
        if (community){
            r = community_size(community);
            log_evidence += calc_evidence_icc(community, model, r);
        }
    }
    return log_evidence;
}
