project(optimize)

set(OPTIMIZED_OP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../)
include_directories(OPTIMIZED_OP_DIR)

########################### optimized files ###########################
file(GLOB OPTIMIZED_ASSEMBLY
        ${OPTIMIZED_OP_DIR}/assembly/opt/*.s
        ${OPTIMIZED_OP_DIR}/assembly/opt/*.S
        )

file(GLOB FP16_SRC
        ${OPTIMIZED_OP_DIR}/fp16/*.cc
        ${OPTIMIZED_OP_DIR}/../fp16/*.cc
        )

########################### share library build ########################
set(OPTIMIZED_OPS ${OPTIMIZED_OP_DIR}/opt_op_handler.c)

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
            ${LITE_DIR}/build/src/runtime/kernel/arm/opclib/liboptimize.so)

    add_custom_command(TARGET optimize POST_BUILD
            COMMAND rm -rf ${LITE_DIR}/output/lib/liboptimize.so
            COMMAND mkdir -pv ${LITE_DIR}/output/lib
            COMMAND cp ${LITE_DIR}/build/src/runtime/kernel/arm/opclib/liboptimize.so ${LITE_DIR}/output/lib)
endif ()
