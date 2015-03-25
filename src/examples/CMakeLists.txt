project(Example1)

include_directories(
	"${CMAKE_SOURCE_DIR}/core/include"
)

CUDA_ADD_EXECUTABLE(Example1 example1.cu)
TARGET_LINK_LIBRARIES(Example1 FEM_CORE)
TARGET_LINK_LIBRARIES(Example1 pthread) 
