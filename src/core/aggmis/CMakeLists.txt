SET (AGGMIS_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL STRING)

include_directories(${AGGMIS_INCLUDE})

set(AGGMIS_CUDA_SOURCES
	
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_Aggregation_CPU.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_MergeSplitConditioner_CPU.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_Types.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_Aggregation_GPU.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_MergeSplitConditioner.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/FractionalTimer.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_BasicTest.cu"        
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_Metrics.cu"                 
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/tetmesh.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_FileIO.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_MIS_CPU.cu"                    
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/Timer.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_GraphHelpers.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_MIS_GPU.cu"              
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/TriMesh_connectivity.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_IOHelpers.cu"   
#	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/AggMIS_Testing.cu"                    
	"${CMAKE_CURRENT_SOURCE_DIR}/cuda/TriMesh_io.cu"
)

CUDA_ADD_LIBRARY(FEM_AGGMIS SHARED ${AGGMIS_CUDA_SOURCES})

