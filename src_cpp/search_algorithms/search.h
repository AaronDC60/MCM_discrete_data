#include "../model/model.h"

// Search algorithms
void exhaustive_search(mcm& model);

void greedy_search(mcm& model);

void divide_and_conquer(mcm&model);
int divide_and_conquer_recursive(int move_from, int move_to, mcm& model, int first_empty);

int find_member_i(int community, int i);
int index_of_member(int member);