#include "gtest/gtest.h"
#include "../model/model.h"
#include "../search_algorithms/search.h"

TEST(evidence, count_obs){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    unordered_map<int, int> counts;
    // Community = 1
    counts = count_observations(model, 1);
    EXPECT_EQ(counts.size(), 3);
    EXPECT_EQ(counts[0], 1);
    EXPECT_EQ(counts[1], 2);
    EXPECT_EQ(counts[2], 4);

    // Community = 2
    counts = count_observations(model, 2);
    EXPECT_EQ(counts.size(), 3);
    EXPECT_EQ(counts[0], 1);
    EXPECT_EQ(counts[3], 5);
    EXPECT_EQ(counts[6], 1);

    // Community = 4
    counts = count_observations(model, 4);
    EXPECT_EQ(counts.size(), 3);
    EXPECT_EQ(counts[0], 4);
    EXPECT_EQ(counts[9], 2);
    EXPECT_EQ(counts[18], 1);

    // Community = 3
    counts = count_observations(model, 3);
    EXPECT_EQ(counts.size(), 5);
    EXPECT_EQ(counts[2], 1);
    EXPECT_EQ(counts[3], 1);
    EXPECT_EQ(counts[4], 1);
    EXPECT_EQ(counts[5], 3);
    EXPECT_EQ(counts[7], 1);

    // Community = 7
    counts = count_observations(model, 7);
    EXPECT_EQ(counts.size(), 6);
    EXPECT_EQ(counts[2], 1);
    EXPECT_EQ(counts[5], 2);
    EXPECT_EQ(counts[7], 1);
    EXPECT_EQ(counts[13], 1);
    EXPECT_EQ(counts[14], 1);
    EXPECT_EQ(counts[21], 1);
}

TEST(evidence, icc){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    EXPECT_FLOAT_EQ(calc_evidence_icc(1, model, 1), -8.769507120030227);
    EXPECT_FLOAT_EQ(calc_evidence_icc(2, model, 1), -7.670894831362117);
    EXPECT_FLOAT_EQ(calc_evidence_icc(4, model, 1), -8.769507120030227);
    EXPECT_FLOAT_EQ(calc_evidence_icc(3, model, 2), -15.982243968542207);
    EXPECT_FLOAT_EQ(calc_evidence_icc(7, model, 3), -23.324842793537606); 
}

TEST(evidence, total){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    vector<int> partition = {1,2,4};
    EXPECT_EQ(calc_evidence(partition, model), calc_evidence_icc(1, model, 1) + calc_evidence_icc(2, model, 1) + calc_evidence_icc(4, model, 1));

    partition = {7,0,0};
    EXPECT_EQ(calc_evidence(partition, model), calc_evidence_icc(7, model, 3));

    partition = {3,4,0};
    EXPECT_EQ(calc_evidence(partition, model), calc_evidence_icc(3, model, 2) + calc_evidence_icc(4, model, 1));
}

TEST(evidence, storage){
    // Read in test data + create model
    vector<vector<int>> data = data_processing("../tests/test.dat", 3);
    mcm model = create_model(data, 3, 3, false);

    // Calculate the log-evidence of 1 community
    get_evidence_icc(3, model);

    EXPECT_EQ(model.evidence_storage.size(), 1);
    EXPECT_EQ(model.evidence_storage[3], calc_evidence_icc(3, model,2));

    // Exhaustive search
    exhaustive_search(model);

    // All partitions must be stored
    EXPECT_EQ(model.evidence_storage.size(), 7);
}