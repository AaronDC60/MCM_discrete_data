#include "gtest/gtest.h"
#include "../model/model.h"

TEST(gt, transform_state){
    vector<int> gt = {4,5};
    vector<int> state;

    state = {0,0};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 0);

    state = {0,1};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 1);
    EXPECT_EQ(state[1], 2);

    state = {0,2};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 2);
    EXPECT_EQ(state[1], 1);

    state = {1,0};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 1);
    EXPECT_EQ(state[1], 1);

    state = {1,1};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 2);
    EXPECT_EQ(state[1], 0);

    state = {1,2};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 2);

    state = {2,0};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 2);
    EXPECT_EQ(state[1], 2);

    state = {2,1};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 0);
    EXPECT_EQ(state[1], 1);

    state = {2,2};
    gt_state(state, gt, 3, 2);
    EXPECT_EQ(state[0], 1);
    EXPECT_EQ(state[1], 0);
}

TEST(gt, transform_data){
    vector<int> gt = {4,5};
    vector<vector<int>> data = {{0,0}, {0,1}, {0,2},
                                {1,0}, {1,1}, {1,2},
                                {2,0}, {2,1}, {2,2}};
    vector<vector<int>> expected_transform = {{0,0}, {1,2}, {2,1},
                                              {1,1}, {2,0}, {0,2},
                                              {2,2}, {0,1}, {1,0}};
    transform_data(data, gt, 3, 2);

    for (int i = 0; i < 9; ++i){
        EXPECT_EQ(data[i][0], expected_transform[i][0]);
        EXPECT_EQ(data[i][1], expected_transform[i][1]);
    }
}

TEST(gt, best_basis){
    vector<vector<int>> data = {{0,1,0,0},
                                {0,1,2,2},
                                {0,1,1,1},
                                {0,2,0,0}};
    mcm model = create_model(data, 3, 4, false);
    find_best_basis(model);

    EXPECT_EQ(model.best_basis.size(), 4);
    EXPECT_EQ(model.best_basis[0], 1);
    EXPECT_EQ(model.best_basis[1], 46);
    EXPECT_EQ(model.best_basis[2], 71);
    EXPECT_EQ(model.best_basis[3], 27);
}