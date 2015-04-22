#include <cstdlib>
#include <cstdio>
#include "gtest/gtest.h"
TEST(CLIRegressionTests, Basic) {
  //make sure there is a command interpreter
  ASSERT_EQ(0,(int)!(std::system(NULL)));
  //setup the line that calls the command line interface
  std::string log = "basic_output.txt";
  std::string output = " > " + std::string(TEST_DATA_DIR) + log + " 2>&1";
  std::string line = (std::string(TEST_BINARY_DIR) + output);
  //make sure there was no error from the command line
  //ASSERT_EQ(0, std::system(line.c_str()));
  //compare all of the related files
  //EXPECT_NO_FATAL_FAILURE(checkoutput());
}
