# gtest external download
set(GTEST_INSTALL_DIR "${EXTERNAL_DIR}/gtest")
if (WIN32)
   set(GTEST_SHARED_ARG ON)
else()
   set(GTEST_SHARED_ARG OFF)
endif()
ExternalProject_Add( gtest
  PREFIX "${GTEST_INSTALL_DIR}"
  SVN_REPOSITORY "http://googletest.googlecode.com/svn/trunk/"
  INSTALL_DIR "${GTEST_INSTALL_DIR}/include"
  BINARY_DIR "${GTEST_INSTALL_DIR}/lib"
  INSTALL_COMMAND ""
  CMAKE_ARGS 
    -Dgtest_build_tests:BOOL=OFF
    -Dgtest_force_shared_crt:BOOL=${GTEST_SHARED_ARG}
)                   
ExternalProject_Add_Step( gtest copy_deps
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${GTEST_INSTALL_DIR}/src/gtest/include ${GTEST_INSTALL_DIR}/include
  DEPENDEES install
)
include_directories("${GTEST_INSTALL_DIR}/include")
SET(GTEST_LIBRARY "${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX};${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}")
########################################################################

# Definitions for Tests
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/bin)
add_definitions(-DTEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data/")
include_directories(${CMAKE_SOURCE_DIR}/core/include)
include_directories(${CMAKE_SOURCE_DIR})
#test macro
function(NEWTEST name)
  add_executable(${name} ${name}.cc)
  if (NOT WIN32)
    target_link_libraries(${name} ${GTEST_LIBRARY} FEM_SOLVER FEM_CORE -lpthread)
  else ()
    target_link_libraries(${name} ${GTEST_LIBRARY} FEM_SOLVER FEM_CORE )
  endif()
  add_test(${name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}) 
  add_dependencies(${name} gtest FEM_SOLVER FEM_CORE)
endfunction()
#the tests
newtest(sanity)