# gtest external download
set(GTEST_INSTALL_DIR "${EXTERNAL_DIR}/gtest")
ExternalProject_Add( gtest
  PREFIX "${GTEST_INSTALL_DIR}"
  SVN_REPOSITORY "http://googletest.googlecode.com/svn/trunk/"
  INSTALL_DIR "${GTEST_INSTALL_DIR}/include"
  BINARY_DIR "${GTEST_INSTALL_DIR}/lib"
  INSTALL_COMMAND ""
  CMAKE_ARGS "-Dgtest_build_tests:BOOL=OFF"
)                   
ExternalProject_Add_Step( gtest copy_deps
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${GTEST_INSTALL_DIR}/src/gtest/include ${GTEST_INSTALL_DIR}/include
  DEPENDEES install
)
include_directories("${GTEST_INSTALL_DIR}/include")
SET(GTEST_LIBRARY "${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX};${GTEST_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_subdirectory(cli)
