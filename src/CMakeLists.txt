cmake_minimum_required(VERSION 2.8)
project(FEM_Solver)

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

SET(BUILD_EXAMPLES false CACHE BOOL "Build sample executables")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
###################
# Externals
include(ExternalProject)
set(EXTERNAL_DIR "${CMAKE_BINARY_DIR}/externals")
# Metis
set(METIS_INSTALL_DIR "${EXTERNAL_DIR}/metis")
ExternalProject_Add( metis
  PREFIX "${METIS_INSTALL_DIR}"
  GIT_REPOSITORY "https://github.com/cibc-internal/metis-4.0.3.git"
  INSTALL_DIR "${METIS_INSTALL_DIR}/include"
  BINARY_DIR "${METIS_INSTALL_DIR}/lib"
  INSTALL_COMMAND ""
  )
ExternalProject_Add_Step( metis copy_deps
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${METIS_INSTALL_DIR}/src/metis/Lib ${METIS_INSTALL_DIR}/include
  DEPENDEES install
)
include_directories("${METIS_INSTALL_DIR}/include")
SET(METIS_LIBRARY "${METIS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}metis${CMAKE_STATIC_LIBRARY_SUFFIX}")
# cusp
set(CUSP_INSTALL_DIR "${EXTERNAL_DIR}/cusp")
ExternalProject_Add( cusp
  PREFIX "${CUSP_INSTALL_DIR}"
  GIT_REPOSITORY "https://github.com/cibc-internal/cusp.git"
  INSTALL_DIR "${CUSP_INSTALL_DIR}/include"
  BINARY_DIR "${CUSP_INSTALL_DIR}/lib"
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  CONFIGURE_COMMAND ""
  )
include_directories("${CUSP_INSTALL_DIR}/src")
###################
# Core / Examples
add_subdirectory(core)

if (${BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()
###################
# Testing
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
  ADD_SUBDIRECTORY(test)
endif()
