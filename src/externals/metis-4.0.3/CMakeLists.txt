cmake_minimum_required(VERSION 2.8)
project(METIS)

set(METIS_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "path to METIS")
SET (METIS_LIBRARY "metis" CACHE INTERNAL STRING)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Lib)

set(METIS_SOURCES
	Lib/balance.c 
	Lib/bucketsort.c
	Lib/ccgraph.c
	Lib/coarsen.c
	Lib/compress.c
	Lib/debug.c
	Lib/estmem.c
	Lib/fm.c
	Lib/fortran.c
	Lib/frename.c
	Lib/graph.c
	Lib/initpart.c
	Lib/kmetis.c
	Lib/kvmetis.c
	Lib/kwayfm.c
	Lib/kwayrefine.c
	Lib/kwayvolfm.c
	Lib/kwayvolrefine.c
	Lib/match.c
	Lib/mbalance2.c
	Lib/mbalance.c
	Lib/mcoarsen.c
	Lib/memory.c
	Lib/mesh.c
	Lib/meshpart.c
	Lib/mfm2.c
	Lib/mfm.c
	Lib/mincover.c
	Lib/minitpart2.c
	Lib/minitpart.c
	Lib/mkmetis.c
	Lib/mkwayfmh.c
	Lib/mkwayrefine.c
	Lib/mmatch.c
	Lib/mmd.c
	Lib/mpmetis.c
	Lib/mrefine2.c
	Lib/mrefine.c
	Lib/mutil.c
	Lib/myqsort.c
	Lib/ometis.c
	Lib/parmetis.c
	Lib/pmetis.c
	Lib/pqueue.c
	Lib/refine.c
	Lib/separator.c
	Lib/sfm.c
	Lib/srefine.c
	Lib/stat.c
	Lib/subdomains.c
	Lib/timing.c
	Lib/util.c
)

set(METIS_HEADERS
	Lib/defs.h
	Lib/macros.h
	Lib/metis.h
	Lib/proto.h
	Lib/rename.h
	Lib/struct.h
)

add_library(${METIS_LIBRARY} STATIC ${METIS_SOURCES} ${METIS_HEADERS})