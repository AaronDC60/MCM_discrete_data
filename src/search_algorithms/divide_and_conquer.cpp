#include "search.h"

int find_member_i(int community, int i){
    int counter = -1;
    int position = 1;
    while(community){
        if(community & 1){
            counter += 1;
        }
        if (counter == i){
            return position;
        }
        community >>= 1;
        position <<= 1;
    }
    return position;
}

int index_of_member(int member){
    int i=0;
    while((member&1) == false){
        member >>= 1;
        i++;
    }
    return i;
}

void divide_and_conquer(mcm&model){
    // Start from complete model
    int n_members = model.n;
    vector<int> partition;
    partition.assign(n_members, 0);
    int element = 1;
    for (int i = 0; i < n_members; i++){
        // Put every component in first community
        partition[0] += element;
        element <<= 1;
    }
    model.best_mcm.push_back(partition);
    model.best_evidence = calc_evidence(model.best_mcm[0], model);

    if(model.log_file){
        model.divide_and_conquer_file << "Start divide and conquer procedure" << endl;
    }

    // Start recursive algorithm with moving element from first to the second community
    divide_and_conquer_recursive(0, 1, model, 1);
}

int divide_and_conquer_recursive(int move_from, int move_to, mcm& model, int first_empty){
    // Number of member in the community that we want to split
    int n_members_1 = community_size(model.best_mcm[0][move_from]);
    int n_members_2 = 0;
    // Check if the community can be split
    if (n_members_1 == 1){return first_empty;}

    // Hard copy of the starting partition
    vector<int> partition = model.best_mcm[0];

    // Variables for the difference in evidence before and after split
    double best_evidence_diff = 0;
    double best_evidence_diff_tmp;
    double evidence_diff;

    // Variables to represent the unsplit and split communities
    int unsplit_community;
    int community_1;
    int community_2;

    // Variables to store the best split
    int best_community_1;
    int best_community_2;
    // Variable to indicate which member is moving
    int member;

    double evidence_unsplit_community = calc_evidence_icc(partition[move_from], model, n_members_1);
    n_members_1 -= 1;
    n_members_2 += 1;

    while (n_members_1 > 1){
        // Initial values
        best_evidence_diff_tmp = -DBL_MAX;
        unsplit_community = partition[move_from];
        community_1 = unsplit_community;
        community_2 = partition[move_to];

        if(model.log_file){
            // Write to file
            model.divide_and_conquer_file << "\nStart moving members from community " << move_from << " to community " << move_to << endl;
            print_partition_to_file(model.divide_and_conquer_file, partition);
        }

        for (int i = 0; i <= n_members_1; i++){
            // Move member i from first partition to second
            member = find_member_i(unsplit_community, i);
            community_1 -= member;
            community_2 += member;

            // Calculate difference in evidence from splitting
            evidence_diff = calc_evidence_icc(community_1, model, n_members_1) + calc_evidence_icc(community_2, model, n_members_2) - evidence_unsplit_community;

            if (evidence_diff > best_evidence_diff_tmp){
                // Store best split (irrelevant if it is an overall increase or not)
                best_evidence_diff_tmp = evidence_diff;
                best_community_1 = community_1;
                best_community_2 = community_2;
                // Update partition
                partition[move_from] = best_community_1;
                partition[move_to] = best_community_2;

                if(model.log_file){
                    // Write to file
                    model.divide_and_conquer_file << "\nBest split (intermediate): moving member " << index_of_member(member) << " from community " << move_from << " to community " << move_to << " Evidence difference: " << best_evidence_diff_tmp << endl;
                    print_partition_to_file(model.divide_and_conquer_file, partition);
                }
            }

            // Reset community 1 and 2
            community_1 += member;
            community_2 -= member;
        }
        // Update number of members
        n_members_1 -= 1;
        n_members_2 += 1;

        // Check if the split results in an overall improvement of the evidence
        if (best_evidence_diff_tmp > best_evidence_diff){
            best_evidence_diff = best_evidence_diff_tmp;

            model.best_mcm[0][move_from] = best_community_1;
            model.best_mcm[0][move_to] = best_community_2;
            model.best_evidence = calc_evidence(model.best_mcm[0], model);

            if(model.log_file){
                // Write to file
                model.divide_and_conquer_file << "\nNew best split" << endl;
                print_partition_to_file(model.divide_and_conquer_file, partition);
            }
        }
    }
    // Stop if there was no improvement
    if (model.best_mcm[0][move_to] == 0){
        return first_empty;
    }
    first_empty += 1;
    // Continue with a split of the first subpart
    first_empty = divide_and_conquer_recursive(move_from, first_empty, model, first_empty);
    // Continue with a split of the second subpart
    first_empty = divide_and_conquer_recursive(move_to, first_empty, model, first_empty);

    return first_empty;
}