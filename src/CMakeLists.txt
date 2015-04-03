cmake_minimum_required(VERSION 2.8)
project(FEM_Solver)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


SET(BUILD_FEM_SOLVER_EXAMPLES false CACHE BOOL "Checking this will build sample executables that demonstrate FEM Solver usage")

find_package(CUDA REQUIRED)

include_directories(
  "externals"
  )

add_subdirectory(externals)
add_subdirectory(core)

if (${BUILD_FEM_SOLVER_EXAMPLES})
  add_subdirectory(examples)
endif()

########################################################################
# All of the testing and test framework options
INCLUDE(CTest)
if (BUILD_TESTING)
  ENABLE_TESTING()
  # Use gcov for code coverage (useful for seeing test completeness)
  OPTION(USE_GCOV "Enable code coverage for GNUCXX Platforms" OFF)
  if (USE_GCOV)
    if(CMAKE_COMPILER_IS_GNUCXX)
      INCLUDE(${PROJECT_SOURCE_DIR}/CodeCoverage.cmake)
      SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
      SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    endif()
  endif()
  ADD_SUBDIRECTORY(externals/gtest)
  ADD_SUBDIRECTORY(test)
endif()
