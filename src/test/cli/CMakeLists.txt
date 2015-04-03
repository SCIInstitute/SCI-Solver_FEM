########################################################################

# Tests for cli
include_directories(${FEM_Solver_SOURCE_DIR}/externals/gtest)
include_directories(${FEM_Solver_SOURCE_DIR}/externals/gtest/include)
add_definitions(-DBINARY_DIR="${FEM_Solver_BINARY_DIR}/test/cli")
add_definitions(-DTEST_DATA_DIR="${FEM_Solver_SOURCE_DIR}/test/test_data/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin)
#cli
cxx_test(cli_basic gtest_main)