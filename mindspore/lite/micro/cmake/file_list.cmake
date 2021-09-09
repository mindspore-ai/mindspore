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
        ${MICRO_DIR}/coder/generator/component/const_blocks/cmake_lists.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/debug_utils.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/msession.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/mtensor.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/mmodel.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/license.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/load_input.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/calib_output.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/thread_pool.cc
        ${MICRO_DIR}/coder/generator/component/const_blocks/benchmark.cc
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
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/resize_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/scale_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/softmax_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/tile_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/transpose_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/splice_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/exp_fp32_coder.cc
        ${MICRO_DIR}/coder/opcoders/nnacl/fp32/deconv2d_fp32_coder.cc
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
        ${MICRO_DIR}/coder/opcoders/nnacl/int8/transpose_int8_coder.cc
        #### nnacl dequant coder
        ${MICRO_DIR}/coder/opcoders/nnacl/dequant/de_quant.cc
        #### custom
        ${MICRO_DIR}/coder/opcoders/custom/custom_coder.cc
        )

set(LITE_SRC
        ${LITE_DIR}/src/cxx_api/tensor_utils.cc
        ${LITE_DIR}/src/cxx_api/types.cc
        ${LITE_DIR}/src/cxx_api/tensor/tensor_impl.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/prim_util.cc
        ${LITE_DIR}/src/common/string_util.cc
        ${LITE_DIR}/src/common/lite_utils.cc
        ${LITE_DIR}/src/common/tensor_util.cc
        ${LITE_DIR}/src/runtime/infer_manager.cc
        ${LITE_DIR}/src/registry/register_kernel_interface.cc
        ${LITE_DIR}/src/registry/kernel_interface_registry.cc
        ${LITE_DIR}/src/registry/register_kernel.cc
        ${LITE_DIR}/src/registry/register_kernel_impl.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/ms_tensor.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/tensor.cc
        ${LITE_DIR}/src/weight_decoder.cc
        ${LITE_DIR}/src/huffman_decode.cc
        ${LITE_DIR}/src/common/log.cc
        ${LITE_DIR}/src/common/utils.cc
        ${LITE_DIR}/../core/utils/status.cc
        ### tools
        ${LITE_DIR}/tools/common/flag_parser.cc
        )

set(REGISTRY_SRC
        ${MICRO_DIR}/coder/opcoders/kernel_registry.cc
        )

list(APPEND FILE_SET ${CODER_SRC} ${CODER_OPCODERS_SRC} ${CODER_GENERATOR_SRC}
        ${CODER_ALLOCATOR_SRC} ${LITE_SRC} ${REGISTRY_SRC})
