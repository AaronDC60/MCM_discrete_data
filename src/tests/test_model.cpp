#include "gtest/gtest.h"
#include "../model/model.h"

TEST(model, construction){
    int q = 3;
    int n = 3;
    // Read in data
    vector<vector<int>> data = data_processing("../tests/test.dat", n);
    // Construct model
    mcm model = create_model(data, q, n, false);

    // Check initializations
    EXPECT_EQ(model.N, 7);
    EXPECT_EQ(model.q, 3);
    EXPECT_EQ(model.n, 3);

    // Check powers of q
    vector<int> powers = {1,3,9};
    for(int i = 0; i < n; i++){
        EXPECT_EQ(powers[i], model.pow_q[i]) << "Wrong power at index " << i;
    }
}