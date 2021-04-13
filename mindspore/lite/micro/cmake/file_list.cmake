#### classify all .h .c .cc files to FILE_SET
set(CODER_SRC
        ${MICRO_DIR}/coder/coder.cc
        ${MICRO_DIR}/coder/context.cc
        ${MICRO_DIR}/coder/graph.cc
        ${MICRO_DIR}/coder/session.cc
        ${MICRO_DIR}/coder/train.cc
        ${MICRO_DIR}/coder/utils/coder_utils.cc
        ${MICRO_DIR}/coder/utils/dir_utils.cc
        ${MICRO_DIR}/coder/utils/type_cast.cc
        )

set(CODER_ALLOCATOR_SRC
        ${MICRO_DIR}/coder/allocator/allocator.cc
        ${MICRO_DIR}/coder/allocator/memory_manager.cc
        )

set(CODER_GENERATOR_SRC
        ${MICRO_DIR}/coder/generator/generator.cc
        ${MICRO_DIR}/coder/generator/inference/inference_generator.cc
        ${MICRO_DIR}/coder/generator/train/train_generator.cc
        ${MICRO_DIR}/coder/generator/component/common_component.cc
        ${MICRO_DIR}/coder/generator/component/weight_component.cc
        ${MICRO_DIR}/coder/generator/component/cmake_component.cc
        ${MICRO_DIR}/coder/generator/component/train_component.cc
        ${MICRO_DIR}/coder/generator/component/parallel_component.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/cmake_lists.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/debug_utils.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/msession.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/mtensor.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/mstring.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/model.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/license.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/load_input.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/thread_pool.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/benchmark.cc
        )

set(MINDSPORE_CORE
        ${TOP_DIR}/mindspore/core/gvar/logging_level.cc
        )

set(CODER_OPCODERS_SRC
        ${MICRO_DIR}/coder/opcoders/file_collector.cc
        ${MICRO_DIR}/coder/opcoders/op_coder.cc
        ${MICRO_DIR}/coder/opcoders/op_coder_builder.cc
        ${MICRO_DIR}/coder/opcoders/op_coder_register.cc
        #### serializer
        ${MICRO_DIR}/coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.cc
        ${MICRO_DIR}/coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.cc
        ${MICRO_DIR}/coder/opcoders/serializers/nnacl_serializer/nnacl_stream_utils.cc
        #### base coder
        ${MICRO_DIR}/coder/opcoders/base/conv2d_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/dtype_cast_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/full_connection_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/quant_dtype_cast_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/reduce_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/resize_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/reshape_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/softmax_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/detection_post_process_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/base/strided_slice_base_coder.cc
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
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/biasadd_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/concat_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/conv2d_delegate_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/convolution_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/convolution_winograd_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/convolution_depthwise_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/full_connection_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/gather_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/lstm_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/matmul_fp32_base_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/matmul_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/pad_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/pooling_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/power_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/reduce_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/scale_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/softmax_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/tile_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/transpose_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/splice_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/exp_fp32_coder.cc
        #### nnacl int8 coder
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/activation_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/add_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/batchnorm_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/concat_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/fullconnection_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/matmul_base_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/matmul_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/conv2d_1x1_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/conv2d_3x3_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/conv2d_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/convolution_depthwise_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/deconvolution_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/pooling_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/resize_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/reduce_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/reshape_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/softmax_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/sub_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/detection_post_process_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/sigmoid_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/relux_int8_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/div_int8_coder.cc
        #### nnacl dequant coder
        ${MICRO_DIR}/coder/opcoders/nnacl/dequant/de_quant.cc
        )

set(LITE_SRC
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/prim_util.cc
        ${LITE_DIR}/src/common/tensor_util.cc
        ${LITE_DIR}/src/runtime/infer_manager.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/sub_graph_split.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/dequant.cc
        ${LITE_DIR}/src/huffman_decode.cc
        ${LITE_DIR}/src/common/log_adapter.cc
        ${LITE_DIR}/src/common/utils.cc
        ### populate operator parameter
        ${LITE_DIR}/src/ops/ops_utils.cc
        ${LITE_DIR}/src/ops/populate/default_populate.cc
        ${LITE_DIR}/src/ops/populate/conv2d_populate.cc
        ${LITE_DIR}/src/ops/populate/arithmetic_populate.cc
        ${LITE_DIR}/src/ops/populate/add_populate.cc
        ${LITE_DIR}/src/ops/populate/concat_populate.cc
        ${LITE_DIR}/src/ops/populate/conv2d_populate.cc
        ${LITE_DIR}/src/ops/populate/detection_post_process_populate.cc
        ${LITE_DIR}/src/ops/populate/depthwise_conv2d_populate.cc
        ${LITE_DIR}/src/ops/populate/full_connection_populate.cc
        ${LITE_DIR}/src/ops/populate/pooling_populate.cc
        ${LITE_DIR}/src/ops/populate/quant_dtype_cast_populate.cc
        ${LITE_DIR}/src/ops/populate/resize_populate.cc
        ${LITE_DIR}/src/ops/populate/reshape_populate.cc
        ${LITE_DIR}/src/ops/populate/batch_norm_populate.cc
        ${LITE_DIR}/src/ops/populate/slice_populate.cc
        ${LITE_DIR}/src/ops/populate/while_populate.cc
        ${LITE_DIR}/src/ops/populate/matmul_populate.cc
        ${LITE_DIR}/src/ops/populate/bias_add_populate.cc
        ${LITE_DIR}/src/ops/populate/activation_populate.cc
        ${LITE_DIR}/src/ops/populate/softmax_populate.cc
        ${LITE_DIR}/src/ops/populate/splice_populate.cc
        ${LITE_DIR}/src/ops/populate/transpose_populate.cc
        ${LITE_DIR}/src/ops/populate/sub_populate.cc
        ${LITE_DIR}/src/ops/populate/power_populate.cc
        ${LITE_DIR}/src/ops/populate/mul_populate.cc
        ${LITE_DIR}/src/ops/populate/arithmetic_self_populate.cc
        ${LITE_DIR}/src/ops/populate/div_populate.cc
        ${LITE_DIR}/src/ops/populate/erf_populate.cc
        ${LITE_DIR}/src/ops/populate/exp_populate.cc
        ${LITE_DIR}/src/ops/populate/strided_slice_populate.cc
        ${LITE_DIR}/src/ops/populate/lstm_populate.cc
        ${LITE_DIR}/src/ops/populate/squeeze_populate.cc
        ### tools
        ${LITE_DIR}/tools/common/flag_parser.cc
        )

set(NNACL_DIR ${TOP_DIR}/mindspore/ccsrc/backend/kernel_compiler/cpu/nnacl)
set(LITE_KERNEL_SRC
        ### nnacl
        ${NNACL_DIR}/common_func.c
        ${NNACL_DIR}/base/minimal_filtering_generator.c
        ${NNACL_DIR}/base/arithmetic_base.c
        ${NNACL_DIR}/base/slice_base.c
        ${NNACL_DIR}/fp32/winograd_utils.c
        ${NNACL_DIR}/fp32/pack_fp32.c
        ${NNACL_DIR}/int8/quantize.c
        ${NNACL_DIR}/int8/pack_int8.c
        ${NNACL_DIR}/int8/matmul_int8.c
        ${NNACL_DIR}/int8/fixed_point.c
        ${NNACL_DIR}/fp32/matmul_fp32.c
        ${NNACL_DIR}/int8/arithmetic_int8.c
        ${NNACL_DIR}/int8/add_int8.c
        ${NNACL_DIR}/int8/concat_int8.c
        ${NNACL_DIR}/int8/conv_int8.c
        ${NNACL_DIR}/int8/conv3x3_int8.c
        ${NNACL_DIR}/int8/conv1x1_int8.c
        ${NNACL_DIR}/base/conv1x1_base.c
        ${NNACL_DIR}/int8/conv_depthwise_int8.c
        ${NNACL_DIR}/int8/deconv_int8.c
        ${NNACL_DIR}/int8/common_func_int8.c
        ${NNACL_DIR}/int8/slice_int8.c
        ${NNACL_DIR}/int8/batchnorm_int8.c
        ${NNACL_DIR}/int8/sub_int8.c
        ${NNACL_DIR}/int8/quant_dtype_cast_int8.c
        ${NNACL_DIR}/int8/sigmoid_int8.c
        ${NNACL_DIR}/int8/resize_int8.c
        ### infer
        ${NNACL_DIR}/infer/adam_infer.c
        ${NNACL_DIR}/infer/add_sub_grad_infer.c
        ${NNACL_DIR}/infer/addn_infer.c
        ${NNACL_DIR}/infer/apply_momentum_infer.c
        ${NNACL_DIR}/infer/argmin_max_infer.c
        ${NNACL_DIR}/infer/arithmetic_compare_infer.c
        ${NNACL_DIR}/infer/arithmetic_grad_infer.c
        ${NNACL_DIR}/infer/arithmetic_infer.c
        ${NNACL_DIR}/infer/assign_add_infer.c
        ${NNACL_DIR}/infer/assign_infer.c
        ${NNACL_DIR}/infer/batch_to_space_infer.c
        ${NNACL_DIR}/infer/bias_grad_infer.c
        ${NNACL_DIR}/infer/binary_cross_entropy_infer.c
        ${NNACL_DIR}/infer/bn_grad_infer.c
        ${NNACL_DIR}/infer/broadcast_to_infer.c
        ${NNACL_DIR}/infer/cast_infer.c
        ${NNACL_DIR}/infer/common_infer.c
        ${NNACL_DIR}/infer/concat_infer.c
        ${NNACL_DIR}/infer/constant_of_shape_infer.c
        ${NNACL_DIR}/infer/conv2d_grad_filter_infer.c
        ${NNACL_DIR}/infer/conv2d_grad_input_infer.c
        ${NNACL_DIR}/infer/conv2d_infer.c
        ${NNACL_DIR}/infer/deconv2d_infer.c
        ${NNACL_DIR}/infer/dedepthwise_conv2d_infer.c
        ${NNACL_DIR}/infer/depthwise_conv2d_infer.c
        ${NNACL_DIR}/infer/detection_post_process_infer.c
        ${NNACL_DIR}/infer/expand_dims_infer.c
        ${NNACL_DIR}/infer/fill_infer.c
        ${NNACL_DIR}/infer/full_connection_infer.c
        ${NNACL_DIR}/infer/fused_batchnorm_infer.c
        ${NNACL_DIR}/infer/gather_infer.c
        ${NNACL_DIR}/infer/gather_nd_infer.c
        ${NNACL_DIR}/infer/group_conv2d_grad_input_infer.c
        ${NNACL_DIR}/infer/infer_register.c
        ${NNACL_DIR}/infer/lsh_projection_infer.c
        ${NNACL_DIR}/infer/lstm_infer.c
        ${NNACL_DIR}/infer/matmul_infer.c
        ${NNACL_DIR}/infer/max_min_grad_infer.c
        ${NNACL_DIR}/infer/mean_infer.c
        ${NNACL_DIR}/infer/pooling_grad_infer.c
        ${NNACL_DIR}/infer/pooling_infer.c
        ${NNACL_DIR}/infer/power_infer.c
        ${NNACL_DIR}/infer/quant_dtype_cast_infer.c
        ${NNACL_DIR}/infer/range_infer.c
        ${NNACL_DIR}/infer/rank_infer.c
        ${NNACL_DIR}/infer/reduce_infer.c
        ${NNACL_DIR}/infer/reshape_infer.c
        ${NNACL_DIR}/infer/resize_infer.c
        ${NNACL_DIR}/infer/roi_pooling_infer.c
        ${NNACL_DIR}/infer/select_infer.c
        ${NNACL_DIR}/infer/sgd_infer.c
        ${NNACL_DIR}/infer/shape_infer.c
        ${NNACL_DIR}/infer/slice_infer.c
        ${NNACL_DIR}/infer/softmax_cross_entropy_infer.c
        ${NNACL_DIR}/infer/softmax_infer.c
        ${NNACL_DIR}/infer/space_to_batch_infer.c
        ${NNACL_DIR}/infer/space_to_batch_nd_infer.c
        ${NNACL_DIR}/infer/space_to_depth_infer.c
        ${NNACL_DIR}/infer/sparse_softmax_cross_entropy_with_logits_infer.c
        ${NNACL_DIR}/infer/sparse_to_dense_infer.c
        ${NNACL_DIR}/infer/split_infer.c
        ${NNACL_DIR}/infer/squeeze_infer.c
        ${NNACL_DIR}/infer/strided_slice_grad_infer.c
        ${NNACL_DIR}/infer/strided_slice_infer.c
        ${NNACL_DIR}/infer/tile_infer.c
        ${NNACL_DIR}/infer/topk_infer.c
        ${NNACL_DIR}/infer/transpose_infer.c
        ${NNACL_DIR}/infer/unsorted_segment_sum_infer.c
        ${NNACL_DIR}/infer/unsqueeze_infer.c
        ${NNACL_DIR}/infer/where_infer.c
        ${NNACL_DIR}/infer/while_infer.c
        ${NNACL_DIR}/infer/splice_infer.c
        )

#### sse
if("${X86_64_SIMD}" STREQUAL "sse")
    set(SSE_SRC
            ${NNACL_DIR}/intrinsics/sse/MatMul_Sse.c
            )
    set_property(SOURCE ${SSE_SRC} PROPERTY LANGUAGE C)
endif()

#### avx
if("${X86_64_SIMD}" STREQUAL "avx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1 -mavx -mavx2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1 -mavx -mavx2")
    set(AVX_SRC
            ${NNACL_DIR}/intrinsics/avx/common_utils.c
            ${NNACL_DIR}/intrinsics/sse/MatMul_Sse.c
            ${NNACL_DIR}/assembly/avx/MatmulAvx.S
            )
    set_property(SOURCE ${AVX_SRC} PROPERTY LANGUAGE C)
endif()

list(APPEND FILE_SET ${CODER_SRC} ${CODER_OPCODERS_SRC} ${CODER_GENERATOR_SRC}
        ${CODER_ALLOCATOR_SRC} ${LITE_SRC} ${LITE_KERNEL_SRC} ${MINDSPORE_CORE} ${SSE_SRC} ${AVX_SRC})

