#include "../model/model.h"

// Search algorithms
//void find_best_mcm(mcm& model, string method);
void exhaustive_search(mcm& model);
void greedy_search(mcm& model);

void divide_and_conquer_setup(mcm&model);
int divide_and_conquer(int move_from, int move_to, mcm& model, vector<int> members, int first_empty);

int find_member_i(int community, int i);
int index_of_member(int member);

vector<int> convert_partition(int* a, int n);