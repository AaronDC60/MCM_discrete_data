#include "gtest/gtest.h"
#include "../model/model.h"

TEST(data, read_in){
    int n = 3;
    // Read in data
    vector<vector<int>> data = data_processing("../tests/test.dat", n);

    // Check the number of observations
    EXPECT_EQ(data.size(), 7);

    // Check the number of variables
    for(vector<int> obs : data){
        EXPECT_EQ(obs.size(), n);
    }

    // Check individual observations
    vector<int> obs = {2,1,0};
    EXPECT_EQ(data[0], obs);
    EXPECT_EQ(data[5], obs);
    
    obs = {2,0,0};
    EXPECT_EQ(data[1], obs);

    obs = {1,2,0};
    EXPECT_EQ(data[2], obs);

    obs = {2,1,1};
    EXPECT_EQ(data[3], obs);

    obs = {1,1,1};
    EXPECT_EQ(data[4], obs);

    obs = {0,1,2};
    EXPECT_EQ(data[6], obs);
}
