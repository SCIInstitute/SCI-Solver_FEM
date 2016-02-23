#include "gtest/gtest.h"
TEST(SanityTests, sanity) {
  //make sure there is a command interpreter
  ASSERT_EQ(0,(int)!(std::system(NULL)));
}
