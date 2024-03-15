#include "gtest/gtest.h"
#include "../model/model.h"

TEST(partition, comm_size){
    EXPECT_EQ(community_size(0), 0);
    EXPECT_EQ(community_size(1), 1);
    EXPECT_EQ(community_size(2), 1);
    EXPECT_EQ(community_size(3), 2);
    EXPECT_EQ(community_size(4), 1);
    EXPECT_EQ(community_size(5), 2);
    EXPECT_EQ(community_size(6), 2);
    EXPECT_EQ(community_size(7), 3);
    EXPECT_EQ(community_size(8), 1);
}

TEST(partition, comm_string){
    EXPECT_EQ(community_as_string(0, 1), "0");
    EXPECT_EQ(community_as_string(0, 2), "00");
    EXPECT_EQ(community_as_string(1, 3), "100");
    EXPECT_EQ(community_as_string(2, 3), "010");
    EXPECT_EQ(community_as_string(3, 3), "110");
    EXPECT_EQ(community_as_string(4, 3), "001");
    EXPECT_EQ(community_as_string(5, 3), "101");
    EXPECT_EQ(community_as_string(6, 3), "011");
    EXPECT_EQ(community_as_string(7, 3), "111");
    EXPECT_EQ(community_as_string(8, 4), "0001");
}

TEST(partition, conversion){
    vector<int> result;
    vector<int> expected_result;

    int a[3] = {0,0,0};
    result = convert_partition(a, 3);
    expected_result = {7,0,0};

    for(int i = 0; i < 3; i++){
        EXPECT_EQ(result[i], expected_result[i]) << "Partition wrong at index " << i;
    }

    a[1] = 1;
    a[2] = 2;
    result = convert_partition(a, 3);
    expected_result = {1,2,4};

    for(int i = 0; i < 3; i++){
        EXPECT_EQ(result[i], expected_result[i]) << "Partition wrong at index " << i;
    }

    a[0] = 1;
    result = convert_partition(a, 3);
    expected_result = {0,3,4};

    for(int i = 0; i < 3; i++){
        EXPECT_EQ(result[i], expected_result[i]) << "Partition wrong at index " << i;
    }
}