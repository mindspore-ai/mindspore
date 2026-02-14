set(TOOLCHAIN_PATH ${ASCEND_PATH}/toolkit/toolchain)
set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/hcc/bin/aarch64-target-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/hcc/bin/aarch64-target-linux-gnu-g++)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS ${CMAKE_C_COMPILER} AND EXISTS ${CMAKE_CXX_COMPILER})
    find_program(CCACHE_EXECUTABLE ccache)
    if(CCACHE_EXECUTABLE)
        message(STATUS "using ccache to compile aicpu kernels")
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
    endif()
    if(ENABLE_SYM_FILE)
        string(REPLACE "-ggdb" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "-g" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()

    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/securec_arm.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/protobuf_arm.cmake)

    string(REGEX REPLACE "-B.*mold" " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/customize/op_info_cfg)
        add_subdirectory(customize)
    endif()
    add_subdirectory(cpu_kernel)
else()
    message("[WARNING]Due to the lack of cross compiler, the aicpu operator is not compiled.
    If you need to use a custom aicpu operator, please confirm whether the relevant directory
    [${CMAKE_C_COMPILER}] and [${CMAKE_CXX_COMPILER}] is exists")
endif()
