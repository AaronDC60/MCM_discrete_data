cmake_minimum_required(VERSION 3.26.4)

project(mcm_discrete)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Optimization flag
set(CMAKE_CXX_FLAGS "-O3")

# Tests
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(Model)
add_subdirectory(Search_Algorithms)
add_subdirectory(stoch_block_model)

add_executable(${PROJECT_NAME} main.cpp)
#add_executable(${PROJECT_NAME} timing.cpp)
#add_executable(${PROJECT_NAME} data_from_sbm.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Model Search_Algorithms SBM)
target_include_directories(${PROJECT_NAME} PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/Model"
                          "${PROJECT_SOURCE_DIR}/Search_Algorithms"
                          "${PROJECT_SOURCE_DIR}/stoch_block_model"
                          )
