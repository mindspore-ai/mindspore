SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(MICRO_WRAPPER_SRC
        ${LITE_DIR}/src/runtime/thread_pool.c
        ${MICRO_DIR}/wrapper/fp32/matmul_fp32_wrapper.c
        ${MICRO_DIR}/wrapper/int8/matmul_int8_wrapper.c
        ${MICRO_DIR}/wrapper/int8/conv_init_int8_wrapper.c
        ${MICRO_DIR}/wrapper/int8/conv1x1_init_int8_wrapper.c
        ${MICRO_DIR}/wrapper/int8/conv1x1_run_int8_wrapper.c
        )

list(APPEND FILE_SET ${MICRO_WRAPPER_SRC})