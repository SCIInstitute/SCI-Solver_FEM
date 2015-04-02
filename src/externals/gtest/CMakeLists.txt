########################################################################
# CMake build script for Google Test.
# BUILD_SHARED_LIBS is a standard CMake variable, but we declare it here to
# make it prominent in the GUI.
option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." OFF)
# When other libraries are using a shared version of runtime libraries,
# Google Test also has to use one.
option(
  gtest_force_shared_crt
  "Use shared (DLL) run-time lib even when Google Test is built as static lib."
  OFF)
option(gtest_build_tests "Build all of gtest's own tests." OFF)
option(gtest_disable_pthreads "Disable uses of pthreads in gtest." OFF)
########################################################################
# Define helper functions and macros used by Google Test.
include(cmake/internal_utils.cmake)
config_compiler_and_linker()  # Defined in internal_utils.cmake.
# Where Google Test's .h files can be found.
include_directories(
  ${${PROJECT_NAME}_SOURCE_DIR}/externals/gtest/include
  ${${PROJECT_NAME}_SOURCE_DIR}/externals/gtest)
# Where Google Test's libraries can be found.
link_directories(${gtest_SOURCE_DIR}/src)
########################################################################
# Defines the gtest & gtest_main libraries.  User tests should link
# with one of them.
cxx_library(gtest "${cxx_strict}" src/gtest-all.cc)
cxx_library(gtest_main "${cxx_strict}" src/gtest_main.cc)
target_link_libraries(gtest_main gtest)
########################################################################
# Google Test's own tests.
if (gtest_build_tests)
  add_subdirectory(test)
endif()
