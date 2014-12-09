cmake_minimum_required(VERSION 2.8)
project(FEM_Solver)

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
