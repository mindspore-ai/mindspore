set(test_src_file "${CMAKE_BINARY_DIR}/test_bisheng/test_bisheng_toolchain.cc")
set(test_cmake_file "${CMAKE_BINARY_DIR}/test_bisheng/CMakeLists.txt")
file(WRITE ${test_src_file} ""
        "#include <sycl/sycl.hpp> \n"
        "int main () { \n "
        "  return 0; \n"
        "} \n")
file(WRITE ${test_cmake_file} ""
        "project(TEST_BISHENG_COMPILER) \n"
        "set(CMAKE_CXX_COMPILER clang++) \n"
        "set(CMAKE_CXX_FLAGS \"-fsycl -fsycl-targets=ascend_910-cce\") \n"
        "add_executable(test_bisheng \${CMAKE_SOURCE_DIR}/test_bisheng_toolchain.cc) \n")
message("Checking bisheng toolchain ...")
try_compile(try_result ${CMAKE_BINARY_DIR}/test_bisheng ${CMAKE_BINARY_DIR}/test_bisheng test_bisheng
        OUTPUT_VARIABLE out)
if(${try_result})
    message("Bisheng toolchain seems to work.")
    add_subdirectory(impl)
    file(GLOB SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM SRC_LIST "bisheng_kernel_build.cc" "custom_bisheng_kernel.cc" "bisheng_kernel_factory.cc")
    add_library(bisheng_kernels SHARED ${SRC_LIST})
    target_link_libraries(bisheng_kernels PRIVATE mindspore_ascend bisheng_kernels_impl)
    set_target_properties(bisheng_kernels PROPERTIES INSTALL_RPATH $ORIGIN)
    install(TARGETS bisheng_kernels bisheng_kernels_impl OPTIONAL
            LIBRARY DESTINATION lib/plugin/ascend
            )
else()
    message(WARNING "Ignore bisheng ops because bisheng toolchain test failed, message: \n ${out}")
endif()

