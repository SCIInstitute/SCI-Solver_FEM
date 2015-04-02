#Add the actual tests
cxx_test(gtest-death-test_test gtest_main)
cxx_test(gtest_environment_test gtest)
cxx_test(gtest-filepath_test gtest_main)
cxx_test(gtest-linked_ptr_test gtest_main)
cxx_test(gtest-listener_test gtest_main)
cxx_test(gtest_main_unittest gtest_main)
cxx_test(gtest-message_test gtest_main)
cxx_test(gtest_no_test_unittest gtest)
cxx_test(gtest-options_test gtest_main)
cxx_test(gtest-param-test_test gtest
  gtest-param-test2_test.cc)
cxx_test(gtest-port_test gtest_main)
cxx_test(gtest_pred_impl_unittest gtest_main)
cxx_test(gtest_premature_exit_test gtest
  gtest_premature_exit_test.cc)
cxx_test(gtest-printers_test gtest_main)
cxx_test(gtest_prod_test gtest_main
  production.cc)
cxx_test(gtest_repeat_test gtest)
cxx_test(gtest_sole_header_test gtest_main)
cxx_test(gtest_stress_test gtest)
cxx_test(gtest-test-part_test gtest_main)
cxx_test(gtest_throw_on_failure_ex_test gtest)
cxx_test(gtest-typed-test_test gtest_main
  gtest-typed-test2_test.cc)
cxx_test(gtest_unittest gtest_main)
cxx_test(gtest-unittest-api_test gtest)
############################################################
# C++ tests built with non-standard compiler flags.
# MSVC 7.1 does not support STL with exceptions disabled.
if (NOT MSVC OR MSVC_VERSION GREATER 1310)
  cxx_library(gtest_no_exception "${cxx_no_exception}"
    ../src/gtest-all.cc)
  cxx_library(gtest_main_no_exception "${cxx_no_exception}"
    ../src/gtest-all.cc ../src/gtest_main.cc)
endif()
cxx_library(gtest_main_no_rtti "${cxx_no_rtti}"
  ../src/gtest-all.cc ../src/gtest_main.cc)

cxx_test_with_flags(gtest-death-test_ex_nocatch_test
  "${cxx_exception} -DGTEST_ENABLE_CATCH_EXCEPTIONS_=0"
  gtest gtest-death-test_ex_test.cc)
cxx_test_with_flags(gtest-death-test_ex_catch_test
  "${cxx_exception} -DGTEST_ENABLE_CATCH_EXCEPTIONS_=1"
  gtest gtest-death-test_ex_test.cc)

cxx_test_with_flags(gtest_no_rtti_unittest "${cxx_no_rtti}"
  gtest_main_no_rtti gtest_unittest.cc)

cxx_shared_library(gtest_dll "${cxx_default}"
  ../src/gtest-all.cc ../src/gtest_main.cc)

cxx_executable_with_flags(gtest_dll_test_ "${cxx_default}"
  gtest_dll gtest_all_test.cc)
set_target_properties(gtest_dll_test_
  PROPERTIES
  COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1")

if (NOT MSVC OR NOT MSVC_VERSION EQUAL 1600)
  # The C++ Standard specifies tuple_element<int, class>.
  # Yet MSVC 10's <utility> declares tuple_element<size_t, class>.
  # That declaration conflicts with our own standard-conforming
  # tuple implementation.  Therefore using our own tuple with
  # MSVC 10 doesn't compile.
  cxx_library(gtest_main_use_own_tuple "${cxx_use_own_tuple}"
    ../src/gtest-all.cc ../src/gtest_main.cc)

  cxx_test_with_flags(gtest-tuple_test "${cxx_use_own_tuple}"
    gtest_main_use_own_tuple gtest-tuple_test.cc)

  cxx_test_with_flags(gtest_use_own_tuple_test "${cxx_use_own_tuple}"
    gtest_main_use_own_tuple
    gtest-param-test_test.cc gtest-param-test2_test.cc)
endif()
