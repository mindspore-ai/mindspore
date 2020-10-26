set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
set(TEST_DIR ${TOP_DIR}/mindspore/lite/test)
set(LITE_DIR ${TOP_DIR}/mindspore/lite)

include_directories(${TOP_DIR})
include_directories(${TEST_DIR})
add_compile_definitions(ENABLE_NNACL_INFER_SHAPE)

STRING(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
STRING(REPLACE " -fvisibility=hidden " " -fvisibility=default " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

### cpu kernel
file(GLOB KERNEL_OP_SRC
        ${LITE_DIR}/internal/src/kernel/*.cc
        ${LITE_DIR}/internal/src/kernel/common/*.cc
        ${LITE_DIR}/internal/src/kernel/fp32/*.cc
        ${LITE_DIR}/internal/src/kernel/fp32_grad/*.cc
        ${LITE_DIR}/nnacl/*.c
        ${LITE_DIR}/nnacl/fp32/*.c
        ${LITE_DIR}/nnacl/fp32_grad/*.c
        ${LITE_DIR}/nnacl/int8/*.c
        ${LITE_DIR}/nnacl/quantization/*.c
        )

if (PLATFORM_ARM64)
    # assembly
    file(GLOB TEST_ASSEMBLY_SRC ${LITE_DIR}/nnacl/assembly/arm64/*.s
            ${LITE_DIR}/nnacl/assembly/arm64/*.S)

    set_property(SOURCE ${TEST_ASSEMBLY_SRC} PROPERTY LANGUAGE C)
    set(KERNEL_OP_SRC
            ${KERNEL_OP_SRC}
            ${TEST_ASSEMBLY_SRC}
            )
endif()

### runtime framework
set(TEST_LITE_SRC
        ${LITE_DIR}/internal/src/common/string.cc
        ${LITE_DIR}/internal/src/lite_session.cc
        ${LITE_DIR}/internal/src/allocator.cc
        ${LITE_DIR}/internal/src/ms_tensor.cc
        ${LITE_DIR}/internal/src/common/string.cc
        ${LITE_DIR}/internal/src/common/vector.cc
        ${TOP_DIR}/mindspore/core/utils/log_adapter.cc
        ${TOP_DIR}/mindspore/core/gvar/logging_level.cc
        )

### test src
file(GLOB_RECURSE TEST_CASE_KERNEL_SRC
    ${TEST_DIR}/ut/internal/*.cc
)

file(GLOB_RECURSE TEST_CASE_KERNEL_TRAIN_SRC
    ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32_grad/*.cc
)

set(TEST_SRC
    ${TEST_LITE_SRC}
    ${TEST_CASE_KERNEL_SRC}
    ${KERNEL_OP_SRC}
    ${TEST_DIR}/common/common_test.cc
    ${TEST_DIR}/main.cc
)

add_executable(lite-test-internal ${TEST_SRC})

target_link_libraries(lite-test-internal dl ${GTEST_LIBRARY})
