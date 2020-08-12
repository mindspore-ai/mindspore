project(nnacl)

set(NNACL_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
include_directories(NNACL_DIR)

########################### optimized files ###########################
file(GLOB OPTIMIZED_ASSEMBLY
        ${NNACL_DIR}/assembly/opt/*.s
        ${NNACL_DIR}/assembly/opt/*.S
        )

file(GLOB FP16_SRC
        ${NNACL_DIR}/fp16/*.c
        ${NNACL_DIR}/../fp16/*.cc
        )

########################### share library build ########################
set(OPTIMIZED_OPS ${NNACL_DIR}/opt_op_handler.c)

set_property(SOURCE ${OPTIMIZED_ASSEMBLY} PROPERTY LANGUAGE C)
list(APPEND OPTIMIZED_OPS ${OPTIMIZED_ASSEMBLY} ${FP16_SRC})

if (PLATFORM_ARM64)
    string(REPLACE "-fvisibility=hidden" "-fvisibility=default" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a+dotprod+fp16")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod+fp16")
    add_library(optimize SHARED ${OPTIMIZED_OPS})
    target_link_libraries(
            optimize
            mindspore-lite
    )
    set_target_properties(optimize PROPERTIES CLEAN_DIRECT_OUTPUT 1)

    add_custom_command(TARGET optimize POST_BUILD
            COMMAND ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/aarch64-linux-android/bin/strip
            ${TOP_DIR}/build/src/runtime/kernel/arm/nnacl/liboptimize.so)

    add_custom_command(TARGET optimize POST_BUILD
            COMMAND rm -rf ${TOP_DIR}/output/lib/liboptimize.so
            COMMAND mkdir -pv ${TOP_DIR}/output/lib
            COMMAND cp ${TOP_DIR}/build/src/runtime/kernel/arm/nnacl/liboptimize.so ${TOP_DIR}/output/lib)
endif ()
