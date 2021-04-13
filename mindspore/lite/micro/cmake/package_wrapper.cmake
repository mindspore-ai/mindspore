include_directories(${LITE_DIR}/micro/coder/)
set(WRAPPER_DIR ${LITE_DIR}/micro/coder/wrapper/)

set(WRAPPER_SRC
        ${WRAPPER_DIR}/base/common_wrapper.c
        ${WRAPPER_DIR}/base/detection_post_process_base_wrapper.c
        ${WRAPPER_DIR}/base/optimize_handler_wrapper.c
        ${WRAPPER_DIR}/fp32/matmul_fp32_wrapper.c
        ${WRAPPER_DIR}/int8/matmul_int8_wrapper.c
        ${WRAPPER_DIR}/int8/add_int8_wrapper.c
        ${WRAPPER_DIR}/int8/concat_int8_wrapper.c
        ${WRAPPER_DIR}/int8/convolution_int8_wrapper.c
        ${WRAPPER_DIR}/int8/conv_init_int8_wrapper.c
        ${WRAPPER_DIR}/int8/conv1x1_init_int8_wrapper.c
        ${WRAPPER_DIR}/int8/conv1x1_run_int8_wrapper.c
        ${WRAPPER_DIR}/int8/conv3x3_run_int8_wrapper.c
        ${WRAPPER_DIR}/int8/convolution_depthwise_int8_wrapper.c
        ${WRAPPER_DIR}/int8/resize_int8_wrapper.c
        ${WRAPPER_DIR}/int8/slice_int8_wrapper.c
        ${WRAPPER_DIR}/int8/batchnorm_int8_wrapper.c
        )

list(APPEND FILE_SET ${WRAPPER_SRC})
