#### classify all .h .c .cc files to FILE_SET
set(CODER_SRC
        ${MICRO_DIR}/coder/coder.cc
        ${MICRO_DIR}/coder/context.cc
        ${MICRO_DIR}/coder/graph.cc
        ${MICRO_DIR}/coder/session.cc
        ${MICRO_DIR}/coder/train.cc
        )

set(CODER_ALLOCATOR_SRC
        ${MICRO_DIR}/coder/allocator/allocator.cc
        ${MICRO_DIR}/coder/allocator/memory_manager.cc
        )

set(CODER_GENERATOR_SRC
        ${MICRO_DIR}/coder/generator/generator.cc
        ${MICRO_DIR}/coder/generator/inference/inference_generator.cc
        ${MICRO_DIR}/coder/generator/train/train_generator.cc
        ${MICRO_DIR}/coder/generator/component/benchmark_component.cc
        ${MICRO_DIR}/coder/generator/component/common_component.cc
        ${MICRO_DIR}/coder/generator/component/weight_component.cc
        ${MICRO_DIR}/coder/generator/component/cmake_component.cc
        ${MICRO_DIR}/coder/generator/component/train_component.cc
        )

set(CODER_OPCODERS_SRC
        ${MICRO_DIR}/coder/opcoders/file_collector.cc
        ${MICRO_DIR}/coder/opcoders/op_coder.cc
        ${MICRO_DIR}/coder/opcoders/op_coder_builder.cc
        ${MICRO_DIR}/coder/opcoders/op_coder_register.cc
        #### serializer
        ${MICRO_DIR}/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.cc
        ${MICRO_DIR}/coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.cc
        #### base coder
        ${MICRO_DIR}/coder/opcoders/base/conv2d_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/dtype_cast_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/full_connection_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/quant_dtype_cast_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/reduce_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/softmax_base_coder.cc
        #### cmsis int8 coder
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/add_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/conv2d_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/conv2d_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/dwconv_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/fullconnection_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/mul_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/pooling_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/reshape_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/cmsis-nn/int8/softmax_int8_coder.cc
        #### nnacl fp32 coder
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/activation_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/addn_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/arithmetic_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/arithmetic_self_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/assign_add_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/batchnorm_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/concat_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/convolution_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/convolution_depthwise_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/expand_dims_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/full_connection_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/gather_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/matmul_fp32_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/matmul_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/nchw2nhwc_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/nhwc2nchw_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/pad_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/pooling_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/power_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/reduce_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/reshape_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/scale_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/slice_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/softmax_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/squeeze_dims_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/tile_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/transpose_fp32_coder.cc
        #### nnacl int8 coder
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/add_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/concat_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/fullconnection_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/matmul_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/conv2d_1x1_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/conv2d_3x3_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/conv2d_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/deconvolution_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/pooling_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/reduce_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/reshape_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/softmax_int8_coder.cc
        )

set(CODER_UTILS_SRC
        ${MICRO_DIR}/coder/utils/coder_utils.cc
        ${MICRO_DIR}/coder/utils/dir_utils.cc
        ${MICRO_DIR}/coder/utils/type_cast.cc
        )

set(LITE_SRC
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/string_util.cc
        ${LITE_DIR}/src/runtime/allocator.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/common/log_adapter.cc
        ### src/ops for parameter and infer shape
        ${LITE_DIR}/src/ops/batch_norm.cc
        ${LITE_DIR}/src/ops/conv2d.cc
        ${LITE_DIR}/src/ops/primitive_c.cc
        ${LITE_DIR}/src/ops/slice.cc
        ${LITE_DIR}/src/ops/while.cc
        ### populate operator parameter
        ${LITE_DIR}/src/ops/populate/conv2d_populate.cc
        ### tools
        ${LITE_DIR}/tools/common/flag_parser.cc
        )
set(LITE_KERNEL_SRC
        ### nnacl
        ${LITE_DIR}/nnacl/base/minimal_filtering_generator.c
        ${LITE_DIR}/nnacl/fp32/winograd_utils.c
        ${LITE_DIR}/nnacl/fp32/pack_fp32.c
        ${LITE_DIR}/nnacl/int8/quantize.c
        ${LITE_DIR}/nnacl/int8/pack_int8.c
        ${LITE_DIR}/nnacl/int8/matmul_int8.c
        ${LITE_DIR}/nnacl/int8/fixed_point.c
        ${LITE_DIR}/nnacl/fp32/matmul_fp32.c
        ${LITE_DIR}/nnacl/int8/conv3x3_int8.c
        ${LITE_DIR}/nnacl/int8/conv1x1_int8.c
        ${LITE_DIR}/nnacl/base/conv1x1_base.c
        ${LITE_DIR}/nnacl/int8/deconv_int8.c
        ${LITE_DIR}/nnacl/int8/common_func_int8.c
        )

list(APPEND FILE_SET ${CODER_SRC} ${CODER_UTILS_SRC} ${CODER_OPCODERS_SRC} ${CODER_GENERATOR_SRC}
        ${CODER_ALLOCATOR_SRC} ${LITE_SRC} ${LITE_KERNEL_SRC})

