#include "gtest/gtest.h"
#include "../stoch_block_model/sbm.h"

TEST(sbm, check_comm){
    // 1111100000, 0000011111
    std::vector<int> comm_structure = {31,992};

    std::vector<int> vars = {0};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {1};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));

    vars = {0,0};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {0,1};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {0,2};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {0,3};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {0,4};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {1,2};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {1,3};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {1,4};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {2,3};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {2,4};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {3,4};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {4,1};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));

    vars = {0,5};
    EXPECT_FALSE(check_in_or_out(comm_structure, vars));
    vars = {0,6};
    EXPECT_FALSE(check_in_or_out(comm_structure, vars));
    vars = {0,7};
    EXPECT_FALSE(check_in_or_out(comm_structure, vars));
    vars = {0,8};
    EXPECT_FALSE(check_in_or_out(comm_structure, vars));
    vars = {0,9};
    EXPECT_FALSE(check_in_or_out(comm_structure, vars));

    vars = {0,1,3};
    EXPECT_TRUE(check_in_or_out(comm_structure, vars));
    vars = {0,1,5};
    EXPECT_FALSE(check_in_or_out(comm_structure, vars));
}

TEST(sbm, find_index){
    int index = 2;
    int max_value = 5;
    std::vector<int> set = {0,1,2};

    EXPECT_EQ(find_index(set, index, max_value), 2);
    set = {0,1,5};
    EXPECT_EQ(find_index(set, index, max_value), 1);
    set = {0,4,5};
    EXPECT_EQ(find_index(set, index, max_value), 0);
    set = {3,4,5};
    EXPECT_EQ(find_index(set, index, max_value), -1);
}

TEST(sbm, create_graph){
    // 1100, 0011
    std::vector<int> comm_structure = {12,3};

    // 2 States

    // First-order interaction

    // Only intracommunity links (p_in = 1, p_out = 0)
    vector<int> interaction = {1};
    std::vector<std::complex<double>> g = create_graph(comm_structure, 1, 0, interaction, 0.75, 2, 4);
    std::vector<std::complex<double>> expected_g(16, 0);
    expected_g[1] = 0.75;
    expected_g[2] = 0.75;
    expected_g[4] = 0.75;
    expected_g[8] = 0.75;

    EXPECT_EQ(g.size(), 16);
    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[i]) << "Parameters wrong at index " << i;
    }
    std::fill(expected_g.begin(), expected_g.end(), 0);

    // Only intercommunity links (p_in = 0, p_out = 1)
    g = create_graph(comm_structure, 0, 1, interaction, 0.75, 2, 4);
    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[0]) << "Parameters wrong at index " << i;
    }    

    // Second-order interaction

    // Only intracommunity links (p_in = 1, p_out = 0)
    interaction = {1,1};
    g = create_graph(comm_structure, 1, 0, interaction, 0.8, 2, 4);
    expected_g[3] = 0.8;
    expected_g[12] = 0.8;

    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[i]) << "Parameters wrong at index " << i;
    }
    std::fill(expected_g.begin(), expected_g.end(), 0);

    // Only intercommunity links (p_in = 0, p_out = 1)
    g = create_graph(comm_structure, 0, 1, interaction, 0.8, 2, 4);
    expected_g[5] = 0.8;
    expected_g[6] = 0.8;
    expected_g[9] = 0.8;
    expected_g[10] = 0.8; 

    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[i]) << "Parameters wrong at index " << i;
    }
    std::fill(expected_g.begin(), expected_g.end(), 0);

    // Third-order interaction

    // Only intracommunity links (p_in = 1, p_out = 0)
    interaction = {1,1,1};
    g = create_graph(comm_structure, 1, 0, interaction, 0.8, 2, 4);
    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[i]) << "Parameters wrong at index " << i;
    }

    // Only intercommunity links (p_in = 0, p_out = 1)
    g = create_graph(comm_structure, 0, 1, interaction, 1, 2, 4);
    expected_g[7] = 1;
    expected_g[11] = 1;
    expected_g[13] = 1;
    expected_g[14] = 1;

    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[i]) << "Parameters wrong at index " << i;
    }
    std::fill(expected_g.begin(), expected_g.end(), 0);

    // 3 states
    expected_g.resize(81);
    std::fill(expected_g.begin(), expected_g.end(), 0);

    std::complex<double> strength(1, 0.25);
    // Second-order interaction, only intracommunity links (p_in = 1, p_out = 0)
    interaction = {2,1};
    g = create_graph(comm_structure, 1, 0, interaction, strength, 3, 4);
    expected_g[5] = strength;
    expected_g[7] = std::conj(strength);
    expected_g[45] = strength;
    expected_g[63] = std::conj(strength);

    for(int i = 0; i < 16; ++i){
        EXPECT_EQ(g[i], expected_g[i]) << "Parameters wrong at index " << i;
    }
}

TEST(data, int_to_str){
    // Base 2
    EXPECT_EQ(int_to_string(0,2,3), "000");
    EXPECT_EQ(int_to_string(1,2,4), "1000");
    EXPECT_EQ(int_to_string(2,2,4), "0100");
    EXPECT_EQ(int_to_string(3,2,4), "1100");
    EXPECT_EQ(int_to_string(19,2,5), "11001");

    // Base 3
    EXPECT_EQ(int_to_string(0,3,3), "000");
    EXPECT_EQ(int_to_string(1,3,4), "1000");
    EXPECT_EQ(int_to_string(2,3,4), "2000");
    EXPECT_EQ(int_to_string(3,3,4), "0100");
    EXPECT_EQ(int_to_string(4,3,4), "1100");
    EXPECT_EQ(int_to_string(5,3,4), "2100");
    EXPECT_EQ(int_to_string(73,3,4), "1022");
}

TEST(data, fwht){
    // Base 2
    std::vector<std::complex<double>> g = {1,0,1,0,0,1,1,0};
    std::vector<double> expected_wht_g = {4,2,0,-2,0,2,0,2};
    std::vector<std::complex<double>> wht = fwht(g, 2);
    for (int i = 0; i < 8; ++i){
        EXPECT_NEAR(wht[i].real(), expected_wht_g[i], 1E-10);
        EXPECT_NEAR(wht[i].imag(), 0, 1E-10);
    }

    // Base 4
    g = {2,1,0,-1};
    std::vector<std::complex<double>> expected_wht = {std::complex<double>(2,0),std::complex<double>(2,2),std::complex<double>(2,0),std::complex<double>(2,-2)};
    std::vector<std::complex<double>> wht_g = fwht(g, 4);

    for (int i = 0; i < 4; ++i){
        EXPECT_FLOAT_EQ(wht_g[i].real(), expected_wht[i].real());
        EXPECT_FLOAT_EQ(wht_g[i].imag(), expected_wht[i].imag());
    }

}