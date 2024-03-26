#include "gtest/gtest.h"
#include "../model/model.h"

TEST(spin_op, spin_value){
    vector<int> state;
    // Base 2
    state = {0,0};
    EXPECT_EQ(spin_value(state, 1, 2), 0);
    state = {0,1};
    EXPECT_EQ(spin_value(state, 1, 2), 0);
    EXPECT_EQ(spin_value(state, 2, 2), 1);
    EXPECT_EQ(spin_value(state, 3, 2), 1);
    state = {1,0};
    EXPECT_EQ(spin_value(state, 3, 2), 1);
    state = {1,1};
    EXPECT_EQ(spin_value(state, 1, 2), 1);
    EXPECT_EQ(spin_value(state, 3, 2), 0);

    // Base 3
    state = {1,0};
    EXPECT_EQ(spin_value(state, 1, 3), 1);
    state = {2,0};
    EXPECT_EQ(spin_value(state, 1, 3), 2);
    EXPECT_EQ(spin_value(state, 4, 3), 2);
    state = {2,1};
    EXPECT_EQ(spin_value(state, 4, 3), 0);
    EXPECT_EQ(spin_value(state, 7, 3), 1);
}

TEST(spin_op, entropy){
    vector<double> prob_distr = {1,0};
    EXPECT_EQ(entropy(prob_distr), 0);

    prob_distr = {0.5, 0.5};
    EXPECT_EQ(entropy(prob_distr), 1);

    prob_distr = {0.25, 0.25, 0.25, 0.25};
    EXPECT_EQ(entropy(prob_distr), 2);

    prob_distr = {0.75, 0.25};
    EXPECT_FLOAT_EQ(entropy(prob_distr), 0.8112781244591);
}

TEST(spin_op, entropy_op){
    vector<vector<int>> data = {{0,1,0,0},
                                {0,1,2,2},
                                {0,1,1,1},
                                {0,2,0,0}};
    EXPECT_EQ(entropy_of_op(data, 0, 3), 0);
    EXPECT_EQ(entropy_of_op(data, 1, 3), 0);
    EXPECT_EQ(entropy_of_op(data, 2, 3), 0);
    EXPECT_EQ(entropy_of_op(data, 45, 3), 0);
    EXPECT_EQ(entropy_of_op(data, 63, 3), 0);
    EXPECT_FLOAT_EQ(entropy_of_op(data, 3, 3), 0.8112781244591);
    EXPECT_FLOAT_EQ(entropy_of_op(data, 6, 3), 0.8112781244591);
}

TEST(spin_op, comb_op){
    pair<int, int> result;
    // Base 2
    result = comb_ops(0,7,2);
    EXPECT_EQ(result.first, 7);
    EXPECT_EQ(result.second, 7);

    result = comb_ops(1,2,2);
    EXPECT_EQ(result.first, 3);
    EXPECT_EQ(result.second, 3);

    result = comb_ops(3,6,2);
    EXPECT_EQ(result.first, 5);
    EXPECT_EQ(result.second, 5);

    // Base 3
    result = comb_ops(0,7,3);
    EXPECT_EQ(result.first, 7);
    EXPECT_EQ(result.second, 5);

    result = comb_ops(3,4,3);
    EXPECT_EQ(result.first, 7);
    EXPECT_EQ(result.second, 2);
    
    result = comb_ops(25,16,3);
    EXPECT_EQ(result.first, 5);
    EXPECT_EQ(result.second, 9);
}

TEST(spin_op, indices){
    pair<int, int> indices;

    // 2 States

    // First-order interaction
    vector<int> vars = {0};
    vector<int> interaction = {1};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 1);
    EXPECT_EQ(indices.second, 1);

    vars = {1};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 2);
    EXPECT_EQ(indices.second, 2);

    vars = {2};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 4);
    EXPECT_EQ(indices.second, 4);

    // Second-order interaction
    vars = {0,1};
    interaction = {1,1};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 3);
    EXPECT_EQ(indices.second, 3);

    vars = {0,2};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 5);
    EXPECT_EQ(indices.second, 5);

    vars = {1,2};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 6);
    EXPECT_EQ(indices.second, 6);

    // Third-order interaction
    vars = {0,1,2};
    interaction = {1,1,1};
    indices = get_spin_ops_index(vars, interaction, 2);
    EXPECT_EQ(indices.first, 7);
    EXPECT_EQ(indices.second, 7);

    // 3 States

    // First-order interaction
    vars = {0};
    interaction = {1};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 1);
    EXPECT_EQ(indices.second, 2);

    interaction = {2};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 2);
    EXPECT_EQ(indices.second, 1);

    vars = {1};
    interaction = {1};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 3);
    EXPECT_EQ(indices.second, 6);

    vars = {2};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 9);
    EXPECT_EQ(indices.second, 18);  

    // Second-order interaction
    vars = {0,1};
    interaction = {1,1};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 4);
    EXPECT_EQ(indices.second, 8);

    interaction = {2,2};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 8);
    EXPECT_EQ(indices.second, 4);

    interaction = {1,2};
    indices = get_spin_ops_index(vars, interaction, 3);
    EXPECT_EQ(indices.first, 7);
    EXPECT_EQ(indices.second, 5);
}
