#include "gtest/gtest.h"
#include "../search_algorithms/search.h"

TEST(search, greedy){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    greedy_search(model);

    // Expected results
    vector<int> mcm = {7,0,0};
    double evidence = -23.324842793537613;

    EXPECT_EQ(model.best_mcm[0], mcm);
    EXPECT_FLOAT_EQ(model.best_evidence, evidence);
}

TEST(search, divide_and_conquer){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    divide_and_conquer(model);

    // Expected results
    vector<int> mcm = {7,0,0};
    double evidence = -23.324842793537613;

    EXPECT_EQ(model.best_mcm[0], mcm);
    EXPECT_FLOAT_EQ(model.best_evidence, evidence);
}

TEST(search, exhaustive){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    exhaustive_search(model);

    // Expected results
    vector<int> mcm = {7,0,0};
    double evidence = -23.324842793537613;

    EXPECT_EQ(model.best_mcm[0], mcm);
    EXPECT_FLOAT_EQ(model.best_evidence, evidence);
}

TEST(search, n_solutions){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test_2.dat", 10);
    mcm model = create_model(data, 3, 10, false);

    exhaustive_search(model);

    // Expected results
    vector<int> mcm = {31,992,0,0,0,0,0,0,0,0};
    double evidence = -1650.1673437747404;
    int n_same_mcms = 126; // (10 choose 5)/2

    EXPECT_EQ(model.best_mcm.size(), n_same_mcms);
    EXPECT_EQ(model.best_mcm[0], mcm);
    EXPECT_FLOAT_EQ(model.best_evidence, evidence);
}