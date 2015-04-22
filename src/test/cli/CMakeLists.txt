########################################################################

# Definitions for Tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin)
add_definitions(-DTEST_BINARY_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cli")
add_definitions(-DTEST_DATA_DIR="${CMAKE_SOURCE_DIR}/test/test_data/")
#cli
add_executable(cli_basic cli_basic.cc)
if (NOT WIN32)
  target_link_libraries(cli_basic ${GTEST_LIBRARY} -lpthread)
else ()
  target_link_libraries(cli_basic ${GTEST_LIBRARY})
endif()
add_test(cli_basic ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cli_basic)
add_dependencies(cli_basic gtest)
