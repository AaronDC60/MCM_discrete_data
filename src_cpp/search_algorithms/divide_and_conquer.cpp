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

void divide_and_conquer_setup(mcm&model){
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
    model.best_mcm = partition;

    model.members.assign(n_members, 0);
    model.members[0] = n_members;

    // Make copy of the vector containing the number of elements per community
    vector<int> members = model.members;
    // Start recursive algorithm with moving element from first to the second community
    divide_and_conquer(0, 1, model, members, 1);
}

int divide_and_conquer(int move_from, int move_to, mcm& model, vector<int> members, int first_empty){
    // Number of member in the community that we want to split
    int n_members = members[move_from];
    vector<int> partition = model.best_mcm;

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

    first_empty += 1;
    double evidence_unsplit_community = calc_evidence_icc(partition[move_from], model);

    while (n_members > 2){
        // Initial values
        best_evidence_diff_tmp = -DBL_MAX;
        unsplit_community = partition[move_from];
        community_1 = unsplit_community;
        community_2 = partition[move_to];

        for (int i = 0; i < n_members; i++){
            // Move member i from first partition to second
            member = find_member_i(unsplit_community, i);
            community_1 -= member;
            community_2 += member;

            // Calculate difference in evidence from splitting
            evidence_diff = calc_evidence_icc(community_1, model) + calc_evidence_icc(community_2, model) - evidence_unsplit_community;

            cout << "moving member " << index_of_member(member) << " from " << move_from << " to " << move_to << endl;

            if (evidence_diff > best_evidence_diff_tmp){
                // Store best split (irrelevant if it is an overall increase or not)
                best_evidence_diff_tmp = evidence_diff;
                best_community_1 = community_1;
                best_community_2 = community_2;
                // Update partition
                partition[move_from] = best_community_1;
                partition[move_to] = best_community_2;
                cout << "New intermediate best: moving member " << index_of_member(member) << " from " << move_from << " to " << move_to << " LogE: " << calc_evidence(partition, model) << endl;
            }

            // Reset community 1 and 2
            community_1 += member;
            community_2 -= member;
        }
        // Update number of members
        n_members -= 1;
        members[move_from] -= 1;
        members[move_to] += 1;

        // Check if the split results in an overall improvement of the evidence
        if (best_evidence_diff_tmp > best_evidence_diff){
            best_evidence_diff = best_evidence_diff_tmp;
            cout << "Overall improvement: Community " << move_from << " becomes " << best_community_1 << " and community " << move_to << " becomes " << best_community_2 << endl;
            model.members[move_from] = members[move_from];
            model.members[move_to] = members[move_to];

            model.best_mcm[move_from] = best_community_1;
            model.best_mcm[move_to] = best_community_2;
        }
    }
    members = model.members;
    // Stop if there was no improvement
    if (model.best_mcm[move_to] == 0){
        return first_empty;
    }
    // Continue with a split of the first subpart
    if (model.members[move_from] != 1){
        first_empty = divide_and_conquer(move_from, first_empty, model, members, first_empty);
    }
    // Continue with a split of the second subpart
    if (model.members[move_to] != 1){
        first_empty = divide_and_conquer(move_to, first_empty, model, members, first_empty);
    }
    return first_empty;
}