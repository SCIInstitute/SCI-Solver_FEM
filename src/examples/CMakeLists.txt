include_directories("${CMAKE_SOURCE_DIR}/core/include")
include_directories("${CMAKE_SOURCE_DIR}")

CUDA_ADD_EXECUTABLE(Example1 example1.cu)
TARGET_LINK_LIBRARIES(Example1 FEM_SOLVER FEM_CORE)

CUDA_ADD_EXECUTABLE(Example2 example2.cu)
TARGET_LINK_LIBRARIES(Example2 FEM_SOLVER FEM_CORE)