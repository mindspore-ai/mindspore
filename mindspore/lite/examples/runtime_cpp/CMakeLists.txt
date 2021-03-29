cmake_minimum_required(VERSION 3.14)
project(RuntimeCpp)
option(SUPPORT_NPU "if support npu" off)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE RUNTIME_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(runtime_cpp ${RUNTIME_CPP})
find_library(log-lib log)
if(SUPPORT_NPU)
    target_link_libraries(
            runtime_cpp
            -Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive
            hiai
            hiai_ir
            hiai_ir_build
            ${log-lib}
    )
else()
    target_link_libraries(
            runtime_cpp
            -Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive
            ${log-lib}
    )
endif()

SET(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/build/tmp)

INSTALL(TARGETS runtime_cpp
        DESTINATION bin)

INSTALL(FILES
        ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so
        DESTINATION lib)

INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/model/mobilenetv2.ms
        DESTINATION model)

if(SUPPORT_NPU)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libhiai.so
            DESTINATION lib)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libhiai_ir.so
            DESTINATION lib)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libhiai_ir_build.so
            DESTINATION lib)
endif()

set(CPACK_GENERATOR "TGZ")

set(CPACK_PACKAGE_FILE_NAME "runtime_cpp_demo")

set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)

include(CPack)
