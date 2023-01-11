include(CMakePackageConfigHelpers)

set(RUNTIME_PKG_NAME ${PKG_NAME_PREFIX}-${RUNTIME_COMPONENT_NAME})

set(CONVERTER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/converter)
set(OBFUSCATOR_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/obfuscator)
set(CROPPER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/cropper)
if(WIN32)
    set(BUILD_DIR ${TOP_DIR}/build)
else()
    set(BUILD_DIR ${TOP_DIR}/mindspore/lite/build)
endif()
set(TEST_CASE_DIR ${TOP_DIR}/mindspore/lite/test/build)
set(EXTENDRT_BUILD_DIR ${TOP_DIR}/mindspore/lite/build/src/extendrt)

set(RUNTIME_DIR ${RUNTIME_PKG_NAME}/runtime)
set(RUNTIME_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include)
set(RUNTIME_LIB_DIR ${RUNTIME_PKG_NAME}/runtime/lib)
set(PROVIDERS_LIB_DIR ${RUNTIME_PKG_NAME}/providers)
set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include/dataset)
set(TURBO_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/libjpeg-turbo)
set(GLOG_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/glog)
set(SECUREC_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/securec)
set(MINDSPORE_LITE_LIB_NAME libmindspore-lite)
set(MINDSPORE_LITE_EXTENDRT_LIB_NAME libmindspore-lite)
set(MINDSPORE_CORE_LIB_NAME libmindspore_core)
set(MINDSPORE_GE_LITERT_LIB_NAME libmsplugin-ge-litert)
set(BENCHMARK_NAME benchmark)
set(MSLITE_NNIE_LIB_NAME libmslite_nnie)
set(MSLITE_PROPOSAL_LIB_NAME libmslite_proposal)
set(MICRO_NNIE_LIB_NAME libmicro_nnie)
set(DPICO_ACL_ADAPTER_LIB_NAME libdpico_acl_adapter)
set(BENCHMARK_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark)

set(MINDSPORE_LITE_TRAIN_LIB_NAME libmindspore-lite-train)
set(BENCHMARK_TRAIN_NAME benchmark_train)
set(BENCHMARK_TRAIN_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark_train)
file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so*)

include(${TOP_DIR}/cmake/package_micro.cmake)

function(__install_white_list_ops)
    install(FILES
            ${TOP_DIR}/mindspore/core/ops/abs.h
            ${TOP_DIR}/mindspore/core/ops/adam.h
            ${TOP_DIR}/mindspore/core/ops/addn.h
            ${TOP_DIR}/mindspore/core/ops/all.h
            ${TOP_DIR}/mindspore/core/ops/apply_momentum.h
            ${TOP_DIR}/mindspore/core/ops/assert.h
            ${TOP_DIR}/mindspore/core/ops/assign.h
            ${TOP_DIR}/mindspore/core/ops/assign_add.h
            ${TOP_DIR}/mindspore/core/ops/audio_spectrogram.h
            ${TOP_DIR}/mindspore/core/ops/batch_norm.h
            ${TOP_DIR}/mindspore/core/ops/batch_to_space.h
            ${TOP_DIR}/mindspore/core/ops/batch_to_space_nd.h
            ${TOP_DIR}/mindspore/core/ops/bias_add.h
            ${TOP_DIR}/mindspore/core/ops/binary_cross_entropy.h
            ${TOP_DIR}/mindspore/core/ops/broadcast.h
            ${TOP_DIR}/mindspore/core/ops/cast.h
            ${TOP_DIR}/mindspore/core/ops/ceil.h
            ${TOP_DIR}/mindspore/core/ops/clip.h
            ${TOP_DIR}/mindspore/core/ops/concat.h
            ${TOP_DIR}/mindspore/core/ops/attention.h
            ${TOP_DIR}/mindspore/core/ops/cos.h
            ${TOP_DIR}/mindspore/core/ops/constant_of_shape.h
            ${TOP_DIR}/mindspore/core/ops/crop.h
            ${TOP_DIR}/mindspore/core/ops/custom_extract_features.h
            ${TOP_DIR}/mindspore/core/ops/custom_normalize.h
            ${TOP_DIR}/mindspore/core/ops/custom_predict.h
            ${TOP_DIR}/mindspore/core/ops/depend.h
            ${TOP_DIR}/mindspore/core/ops/depth_to_space.h
            ${TOP_DIR}/mindspore/core/ops/detection_post_process.h
            ${TOP_DIR}/mindspore/core/ops/dropout.h
            ${TOP_DIR}/mindspore/core/ops/elu.h
            ${TOP_DIR}/mindspore/core/ops/eltwise.h
            ${TOP_DIR}/mindspore/core/ops/equal.h
            ${TOP_DIR}/mindspore/core/ops/expand_dims.h
            ${TOP_DIR}/mindspore/core/ops/fake_quant_with_min_max_vars.h
            ${TOP_DIR}/mindspore/core/ops/fake_quant_with_min_max_vars_per_channel.h
            ${TOP_DIR}/mindspore/core/ops/fake_quant_with_min_max_vars.h
            ${TOP_DIR}/mindspore/core/ops/fft_real.h
            ${TOP_DIR}/mindspore/core/ops/fft_imag.h
            ${TOP_DIR}/mindspore/core/ops/flatten.h
            ${TOP_DIR}/mindspore/core/ops/floor.h
            ${TOP_DIR}/mindspore/core/ops/floor_div.h
            ${TOP_DIR}/mindspore/core/ops/floor_mod.h
            ${TOP_DIR}/mindspore/core/ops/fill.h
            ${TOP_DIR}/mindspore/core/ops/fused_batch_norm.h
            ${TOP_DIR}/mindspore/core/ops/gather.h
            ${TOP_DIR}/mindspore/core/ops/gather_nd.h
            ${TOP_DIR}/mindspore/core/ops/greater.h
            ${TOP_DIR}/mindspore/core/ops/greater_equal.h
            ${TOP_DIR}/mindspore/core/ops/hashtable_lookup.h
            ${TOP_DIR}/mindspore/core/ops/instance_norm.h
            ${TOP_DIR}/mindspore/core/ops/leaky_relu.h
            ${TOP_DIR}/mindspore/core/ops/less.h
            ${TOP_DIR}/mindspore/core/ops/less_equal.h
            ${TOP_DIR}/mindspore/core/ops/log.h
            ${TOP_DIR}/mindspore/core/ops/logical_and.h
            ${TOP_DIR}/mindspore/core/ops/logical_not.h
            ${TOP_DIR}/mindspore/core/ops/logical_or.h
            ${TOP_DIR}/mindspore/core/ops/lp_normalization.h
            ${TOP_DIR}/mindspore/core/ops/lrn.h
            ${TOP_DIR}/mindspore/core/ops/lsh_projection.h
            ${TOP_DIR}/mindspore/core/ops/lstm.h
            ${TOP_DIR}/mindspore/core/ops/maximum.h
            ${TOP_DIR}/mindspore/core/ops/switch_layer.h
            ${TOP_DIR}/mindspore/core/ops/mfcc.h
            ${TOP_DIR}/mindspore/core/ops/minimum.h
            ${TOP_DIR}/mindspore/core/ops/mod.h
            ${TOP_DIR}/mindspore/core/ops/neg.h
            ${TOP_DIR}/mindspore/core/ops/not_equal.h
            ${TOP_DIR}/mindspore/core/ops/non_max_suppression.h
            ${TOP_DIR}/mindspore/core/ops/one_hot.h
            ${TOP_DIR}/mindspore/core/ops/ones_like.h
            ${TOP_DIR}/mindspore/core/ops/prior_box.h
            ${TOP_DIR}/mindspore/core/ops/quant_dtype_cast.h
            ${TOP_DIR}/mindspore/core/ops/rank.h
            ${TOP_DIR}/mindspore/core/ops/range.h
            ${TOP_DIR}/mindspore/core/ops/reciprocal.h
            ${TOP_DIR}/mindspore/core/ops/real_div.h
            ${TOP_DIR}/mindspore/core/ops/reshape.h
            ${TOP_DIR}/mindspore/core/ops/resize.h
            ${TOP_DIR}/mindspore/core/ops/reverse_sequence.h
            ${TOP_DIR}/mindspore/core/ops/reverse_v2.h
            ${TOP_DIR}/mindspore/core/ops/rfft.h
            ${TOP_DIR}/mindspore/core/ops/roi_pooling.h
            ${TOP_DIR}/mindspore/core/ops/round.h
            ${TOP_DIR}/mindspore/core/ops/rsqrt.h
            ${TOP_DIR}/mindspore/core/ops/scatter_nd.h
            ${TOP_DIR}/mindspore/core/ops/sgd.h
            ${TOP_DIR}/mindspore/core/ops/shape.h
            ${TOP_DIR}/mindspore/core/ops/sigmoid_cross_entropy_with_logits.h
            ${TOP_DIR}/mindspore/core/ops/sin.h
            ${TOP_DIR}/mindspore/core/ops/skip_gram.h
            ${TOP_DIR}/mindspore/core/ops/smooth_l1_loss.h
            ${TOP_DIR}/mindspore/core/ops/softmax.h
            ${TOP_DIR}/mindspore/core/ops/softmax_cross_entropy_with_logits.h
            ${TOP_DIR}/mindspore/core/ops/space_to_batch.h
            ${TOP_DIR}/mindspore/core/ops/space_to_batch_nd.h
            ${TOP_DIR}/mindspore/core/ops/space_to_depth.h
            ${TOP_DIR}/mindspore/core/ops/sparse_softmax_cross_entropy_with_logits.h
            ${TOP_DIR}/mindspore/core/ops/sparse_to_dense.h
            ${TOP_DIR}/mindspore/core/ops/split.h
            ${TOP_DIR}/mindspore/core/ops/sqrt.h
            ${TOP_DIR}/mindspore/core/ops/squeeze.h
            ${TOP_DIR}/mindspore/core/ops/square.h
            ${TOP_DIR}/mindspore/core/ops/squared_difference.h
            ${TOP_DIR}/mindspore/core/ops/stack.h
            ${TOP_DIR}/mindspore/core/ops/strided_slice.h
            ${TOP_DIR}/mindspore/core/ops/switch.h
            ${TOP_DIR}/mindspore/core/ops/tensor_list_from_tensor.h
            ${TOP_DIR}/mindspore/core/ops/tensor_list_get_item.h
            ${TOP_DIR}/mindspore/core/ops/tensor_list_reserve.h
            ${TOP_DIR}/mindspore/core/ops/tensor_list_set_item.h
            ${TOP_DIR}/mindspore/core/ops/tensor_list_stack.h
            ${TOP_DIR}/mindspore/core/ops/transpose.h
            ${TOP_DIR}/mindspore/core/ops/unique.h
            ${TOP_DIR}/mindspore/core/ops/unsorted_segment_sum.h
            ${TOP_DIR}/mindspore/core/ops/unsqueeze.h
            ${TOP_DIR}/mindspore/core/ops/unstack.h
            ${TOP_DIR}/mindspore/core/ops/where.h
            ${TOP_DIR}/mindspore/core/ops/zeros_like.h
            ${TOP_DIR}/mindspore/core/ops/select.h
            ${TOP_DIR}/mindspore/core/ops/scatter_nd_update.h
            ${TOP_DIR}/mindspore/core/ops/gru.h
            ${TOP_DIR}/mindspore/core/ops/non_zero.h
            ${TOP_DIR}/mindspore/core/ops/invert_permutation.h
            ${TOP_DIR}/mindspore/core/ops/size.h
            ${TOP_DIR}/mindspore/core/ops/random_standard_normal.h
            ${TOP_DIR}/mindspore/core/ops/crop_and_resize.h
            ${TOP_DIR}/mindspore/core/ops/erf.h
            ${TOP_DIR}/mindspore/core/ops/is_finite.h
            ${TOP_DIR}/mindspore/core/ops/lin_space.h
            ${TOP_DIR}/mindspore/core/ops/uniform_real.h
            ${TOP_DIR}/mindspore/core/ops/splice.h
            ${TOP_DIR}/mindspore/core/ops/log_softmax.h
            ${TOP_DIR}/mindspore/core/ops/call.h
            ${TOP_DIR}/mindspore/core/ops/custom.h
            ${TOP_DIR}/mindspore/core/ops/cumsum.h
            ${TOP_DIR}/mindspore/core/ops/split_with_overlap.h
            ${TOP_DIR}/mindspore/core/ops/ragged_range.h
            ${TOP_DIR}/mindspore/core/ops/glu.h
            ${TOP_DIR}/mindspore/core/ops/tensor_array.h
            ${TOP_DIR}/mindspore/core/ops/tensor_array_read.h
            ${TOP_DIR}/mindspore/core/ops/tensor_array_write.h
            ${TOP_DIR}/mindspore/core/ops/affine.h
            ${TOP_DIR}/mindspore/core/ops/all_gather.h
            ${TOP_DIR}/mindspore/core/ops/reduce_scatter.h
            ${TOP_DIR}/mindspore/core/ops/dynamic_quant.h
            ${TOP_DIR}/mindspore/core/ops/random_normal.h
            ${TOP_DIR}/mindspore/core/ops/nllloss.h
            ${TOP_DIR}/mindspore/core/ops/op_name.h
            ${TOP_DIR}/mindspore/core/ops/tuple_get_item.h
            ${TOP_DIR}/mindspore/core/ops/add.h
            ${TOP_DIR}/mindspore/core/ops/div.h
            ${TOP_DIR}/mindspore/core/ops/mul.h
            ${TOP_DIR}/mindspore/core/ops/tuple_get_item.h
            ${TOP_DIR}/mindspore/core/ops/scale.h
            ${TOP_DIR}/mindspore/core/ops/sub.h
            ${TOP_DIR}/mindspore/core/ops/avg_pool.h
            ${TOP_DIR}/mindspore/core/ops/exp.h
            ${TOP_DIR}/mindspore/core/ops/conv2d_transpose.h
            ${TOP_DIR}/mindspore/core/ops/conv2d.h
            ${TOP_DIR}/mindspore/core/ops/pow.h
            ${TOP_DIR}/mindspore/core/ops/topk.h
            ${TOP_DIR}/mindspore/core/ops/reduce.h
            ${TOP_DIR}/mindspore/core/ops/arg_max.h
            ${TOP_DIR}/mindspore/core/ops/max_pool.h
            ${TOP_DIR}/mindspore/core/ops/prelu.h
            ${TOP_DIR}/mindspore/core/ops/tile.h
            ${TOP_DIR}/mindspore/core/ops/make_tuple.h
            ${TOP_DIR}/mindspore/core/ops/base_operator.h
            ${TOP_DIR}/mindspore/core/ops/return.h
            ${TOP_DIR}/mindspore/core/ops/pad.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/ops
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
    install(FILES
            ${TOP_DIR}/mindspore/core/ops/fusion/activation.h
            ${TOP_DIR}/mindspore/core/ops/fusion/add_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/adder_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/arg_max_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/arg_min_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/avg_pool_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/conv2d_backprop_filter_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/conv2d_backprop_input_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/conv2d_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/conv2d_transpose_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/div_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/embedding_lookup_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/exp_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/full_connection.h
            ${TOP_DIR}/mindspore/core/ops/fusion/layer_norm_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/l2_normalize_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/mat_mul_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/max_pool_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/mul_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/pad_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/partial_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/pow_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/prelu_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/reduce_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/scale_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/slice_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/sub_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/tile_fusion.h
            ${TOP_DIR}/mindspore/core/ops/fusion/topk_fusion.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/ops/fusion
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
endfunction()
# full mode will also package the files of lite_cv mode.
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
    # full header files
    install(FILES
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/constants.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/data_helper.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/iterator.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/samplers.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/transforms.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_lite.h
            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/liteapi/include/datasets.h
        DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})

    if(PLATFORM_ARM64)
        if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE) AND MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
                    DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/kernels-dvpp-image/utils/libdvpp_utils.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE) AND MSLITE_ENABLE_ACL)
                install(FILES ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
                        DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/kernels-dvpp-image/utils/libdvpp_utils.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.3.0 DESTINATION ${TURBO_DIR}/lib
                RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.2.0 DESTINATION ${TURBO_DIR}/lib
                RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()

    # lite_cv header files
    install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv
            DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "wrapper")
    install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "vision.h" EXCLUDE)
    if(PLATFORM_ARM64)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.3.0 DESTINATION ${TURBO_DIR}/lib RENAME libjpeg.so.62
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.2.0 DESTINATION ${TURBO_DIR}/lib RENAME libturbojpeg.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite")
    install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(PLATFORM_ARM64)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62.3.0
                DESTINATION ${TURBO_DIR}/lib RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0.2.0
                DESTINATION ${TURBO_DIR}/lib RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite_cv")
    if(PLATFORM_ARM64)
        install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(WIN32)
    install(FILES ${TOP_DIR}/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
else()
    install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
endif()
if(NOT PLATFORM_MCU)
    install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${RUNTIME_INC_DIR}/third_party
            COMPONENT ${RUNTIME_COMPONENT_NAME})
endif()
if(PLATFORM_ARM64)
    if(SUPPORT_NPU)
        install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(EXISTS "${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so")
            install(FILES ${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so
                    DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/${MINDSPORE_LITE_EXTENDRT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${EXTENDRT_BUILD_DIR}/delegate/graph_executor/litert/${MINDSPORE_GE_LITERT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR}
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/convert/libruntime_convert_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/kernel/ascend/libascend_kernel_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/delegate/ascend_ge/libascend_ge_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_GPU_BACKEND STREQUAL tensorrt)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/delegate/tensorrt/libtensorrt_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/litert/kernel/ascend/libascend_kernel_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/android-aarch64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/core/mindapi/base/format.h
            ${TOP_DIR}/mindspore/core/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/core/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED OR MSLITE_ENABLE_CONVERTER OR TARGET_HIMIX)
        __install_micro_wrapper()
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${GLOG_DIR} RENAME libmindspore_glog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(MSLITE_ENABLE_TOOLS)
        if(NOT MSLITE_COMPILE_TWICE)
            install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(TARGET_HIMIX)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3559A")
                    install(FILES ${TOP_DIR}/mindspore/lite/build/tools/benchmark/nnie/${MSLITE_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${TOP_DIR}/mindspore/lite/build/tools/benchmark/nnie_proposal/${MSLITE_PROPOSAL_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${TOP_DIR}/mindspore/lite/build/nnie_micro/${MICRO_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            elseif(TARGET_MIX210)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "SD3403" AND (NOT MSLITE_ENABLE_ACL))
                    install(FILES ${TOP_DIR}/mindspore/lite/build/tools/benchmark/dpico/${DPICO_ACL_ADAPTER_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_ENABLE_CONVERTER)
            install(FILES ${TOP_DIR}/mindspore/lite/include/converter.h DESTINATION ${CONVERTER_ROOT_DIR}/include
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION
                    ${CONVERTER_ROOT_DIR}/include/registry COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${API_HEADER}  DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${MINDAPI_BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/base
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${MINDAPI_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/ir
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            __install_white_list_ops()
            install(DIRECTORY ${TOP_DIR}/mindspore/lite/build/schema/
                    DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                    COMPONENT ${RUNTIME_COMPONENT_NAME}
                    FILES_MATCHING PATTERN "*.h" PATTERN "schema_generated.h" EXCLUDE)
            install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(DIRECTORY ${glog_LIBPATH}/../include/glog/
                    DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(DIRECTORY ${TOP_DIR}/third_party/securec/include/
                    DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/tools/converter/libmindspore_converter.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(DIRECTORY ${TOP_DIR}/third_party/proto/ DESTINATION ${CONVERTER_ROOT_DIR}/third_party/proto
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib
                    RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(TARGETS mindspore_core DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_core.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgcodecs.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgproc.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(MSLITE_ENABLE_ACL)
                set(LITE_ACL_DIR ${TOP_DIR}/mindspore/lite/build/tools/converter/adapter/acl)
                install(FILES ${LITE_ACL_DIR}/mindspore_shared_lib/libmindspore_shared_lib.so
                        DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
                if(MSLITE_ENABLE_RUNTIME_CONVERT)
                    install(FILES ${LITE_ACL_DIR}/mindspore_shared_lib/libmindspore_shared_lib.so
                            DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR}
                            RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(TARGETS mindspore_core DESTINATION ${CONVERTER_ROOT_DIR}/lib
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
                install(FILES ${LITE_ACL_DIR}/libascend_pass_plugin.so DESTINATION ${CONVERTER_ROOT_DIR}/lib
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()

            if(MSLITE_ENABLE_DPICO_ATC_ADAPTER)
                install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/adapter/dpico/libdpico_atc_adapter.so
                        DESTINATION ${CONVERTER_ROOT_DIR}/providers/SD3403 COMPONENT ${RUNTIME_COMPONENT_NAME})
                if(MSLITE_ENABLE_TOOLS)
                    install(TARGETS ${BECHCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()

            if(MSLITE_ENABLE_RUNTIME_GLOG)
                install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${RUNTIME_INC_DIR}/third_party/glog
                        COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
                install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${GLOG_DIR}
                        RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            if(MSLITE_ENABLE_RUNTIME_CONVERT)
                install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})

                install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_core.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgproc.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
    endif()
    if(MSLITE_ENABLE_TESTCASES)
        install(FILES ${TOP_DIR}/mindspore/lite/build/test/lite-test DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/build/src/ DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.so")
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/build/minddata/ DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.so")
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TEST_CASE_DIR})
        if(SUPPORT_NPU)
            install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${TEST_CASE_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${TEST_CASE_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${TEST_CASE_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(EXISTS "${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so")
                install(FILES ${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so
                        DESTINATION ${TEST_CASE_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
    endif()
elseif(PLATFORM_ARM32)
    if(SUPPORT_NPU)
        install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(EXISTS "${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so")
            install(FILES ${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so
                    DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/${MINDSPORE_LITE_EXTENDRT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${EXTENDRT_BUILD_DIR}/delegate/graph_executor/litert/${MINDSPORE_GE_LITERT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR}
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/convert/libruntime_convert_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/kernel/ascend/libascend_kernel_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/delegate/ascend_ge/libascend_ge_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_GPU_BACKEND STREQUAL tensorrt)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/delegate/tensorrt/libtensorrt_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/litert/kernel/ascend/libascend_kernel_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/android-aarch32/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/core/mindapi/base/format.h
            ${TOP_DIR}/mindspore/core/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/core/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED OR MSLITE_ENABLE_CONVERTER OR TARGET_OHOS_LITE OR TARGET_HIMIX)
        __install_micro_wrapper()
    endif()
    if(MSLITE_ENABLE_TOOLS AND NOT TARGET_OHOS_LITE)
        if(NOT MSLITE_COMPILE_TWICE)
            install(TARGETS ${BENCHMARK_NAME} RUNTIME
                    DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(TARGET_HIMIX)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3516D" OR ${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3519A")
                    install(FILES ${TOP_DIR}/mindspore/lite/build/tools/benchmark/nnie/${MSLITE_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${TOP_DIR}/mindspore/lite/build/tools/benchmark/nnie_proposal/${MSLITE_PROPOSAL_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${TOP_DIR}/mindspore/lite/build/nnie_micro/${MICRO_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
elseif(WIN32)
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB LIB_LIST ${CXX_DIR}/libstdc++-6.dll ${CXX_DIR}/libwinpthread-1.dll
            ${CXX_DIR}/libssp-0.dll ${CXX_DIR}/libgcc_s_*-1.dll)
    if(MSLITE_ENABLE_CONVERTER)
        install(FILES ${TOP_DIR}/mindspore/lite/include/converter.h DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/converter_lite/converter_lite.exe
                DESTINATION ${CONVERTER_ROOT_DIR}/converter COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/libmindspore_converter.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/registry/libmslite_converter_plugin.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        file(GLOB GLOG_LIB_LIST ${glog_LIBPATH}/../bin/*.dll)
        install(FILES ${GLOG_LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        file(GLOB_RECURSE OPENCV_LIB_LIST
                ${opencv_LIBPATH}/../bin/libopencv_core*
                ${opencv_LIBPATH}/../bin/libopencv_imgcodecs*
                ${opencv_LIBPATH}/../bin/libopencv_imgproc*
                )
        install(FILES ${OPENCV_LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(NOT MSVC AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
            __install_micro_wrapper()
            __install_micro_codegen()
        endif()
    endif()
    if(MSLITE_ENABLE_TOOLS)
        if(MSVC)
            install(FILES ${TOP_DIR}/build/mindspore/tools/benchmark/${BENCHMARK_NAME}.exe
                    DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        else()
            install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        file(GLOB GLOG_LIB_LIST ${glog_LIBPATH}/../bin/*.dll)
        install(FILES ${GLOG_LIB_LIST} DESTINATION ${GLOG_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    install(FILES ${TOP_DIR}/build/mindspore/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/core/mindapi/base/format.h
            ${TOP_DIR}/mindspore/core/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/core/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(MSVC)
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.lib DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll.lib DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
elseif(PLATFORM_MCU)
    __install_micro_wrapper()
    __install_micro_codegen()
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
else()
    install(FILES ${TOP_DIR}/mindspore/lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/core/mindapi/base/format.h
            ${TOP_DIR}/mindspore/core/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/core/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/${MINDSPORE_LITE_EXTENDRT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${EXTENDRT_BUILD_DIR}/delegate/graph_executor/litert/${MINDSPORE_GE_LITERT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR}
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/convert/libruntime_convert_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/kernel/ascend/libascend_kernel_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/delegate/ascend_ge/libascend_ge_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_GPU_BACKEND STREQUAL tensorrt)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/extendrt/delegate/tensorrt/libtensorrt_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${fast_transformers_LIBPATH}/libtransformer-shared.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore/lite/build/src/litert/kernel/ascend/libascend_kernel_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/bin/linux-x64/msobfuscator
                DESTINATION ${OBFUSCATOR_ROOT_DIR} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/linux-x64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${GLOG_DIR} RENAME libmindspore_glog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${RUNTIME_INC_DIR}/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    endif()
    if(MSLITE_ENABLE_CONVERTER)
        install(FILES ${TOP_DIR}/mindspore/lite/include/converter.h DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/include/kernel_interface.h DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${CONVERTER_ROOT_DIR}/include/registry
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${API_HEADER}  DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${MINDAPI_BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/base
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${MINDAPI_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        __install_white_list_ops()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/build/schema/ DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "schema_generated.h" EXCLUDE)
        install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/third_party/proto/ DESTINATION ${CONVERTER_ROOT_DIR}/third_party/proto
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/third_party/securec/include/
                DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/tools/converter/libmindspore_converter.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_core.so.4.5
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgcodecs.so.4.5
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgproc.so.4.5
                COMPONENT ${RUNTIME_COMPONENT_NAME})

        if(MSLITE_ENABLE_ACL)
            set(LITE_ACL_DIR ${TOP_DIR}/mindspore/lite/build/tools/converter/adapter/acl)
            install(FILES ${LITE_ACL_DIR}/mindspore_shared_lib/libmindspore_shared_lib.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(MSLITE_ENABLE_RUNTIME_CONVERT)
                install(FILES ${LITE_ACL_DIR}/mindspore_shared_lib/libmindspore_shared_lib.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR}
                        RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(TARGETS mindspore_core DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            install(FILES ${LITE_ACL_DIR}/libascend_pass_plugin.so DESTINATION ${CONVERTER_ROOT_DIR}/lib
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()

        if(MSLITE_ENABLE_DPICO_ATC_ADAPTER)
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/adapter/dpico/libdpico_atc_adapter.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/providers/SD3403 COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(MSLITE_ENABLE_TOOLS)
                install(TARGETS ${BECHCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()

        if(MSLITE_ENABLE_RUNTIME_GLOG)
            install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${RUNTIME_INC_DIR}/third_party/glog
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0
                    DESTINATION ${GLOG_DIR} RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_ENABLE_RUNTIME_CONVERT)
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})

            install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                    DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_core.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                    DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                    DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgproc.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
            __install_micro_wrapper()
            __install_micro_codegen()
        endif()
    endif()
    if(MSLITE_ENABLE_TOOLS)
        if(NOT MSLITE_COMPILE_TWICE)
            install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
        if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
            install(TARGETS cropper RUNTIME DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/cropper/cropper_mapping_cpu.cfg
                DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/cropper/cropper_mapping_gpu.cfg
                DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/cropper/cropper_mapping_npu.cfg
                DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(SUPPORT_TRAIN)
                install(FILES ${TOP_DIR}/mindspore/lite/build/tools/cropper/cropper_mapping_cpu_train.cfg
                    DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
    endif()
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    install(FILES
            ${TOP_DIR}/mindspore/core/ops/abs.h
            ${TOP_DIR}/mindspore/core/ops/batch_norm.h
            ${TOP_DIR}/mindspore/core/ops/ceil.h
            ${TOP_DIR}/mindspore/core/ops/concat.h
            ${TOP_DIR}/mindspore/core/ops/equal.h
            ${TOP_DIR}/mindspore/core/ops/flatten.h
            ${TOP_DIR}/mindspore/core/ops/gather.h
            ${TOP_DIR}/mindspore/core/ops/gather_nd.h
            ${TOP_DIR}/mindspore/core/ops/maximum.h
            ${TOP_DIR}/mindspore/core/ops/minimum.h
            ${TOP_DIR}/mindspore/core/ops/reshape.h
            ${TOP_DIR}/mindspore/core/ops/softmax.h
            ${TOP_DIR}/mindspore/core/ops/strided_slice.h
            ${TOP_DIR}/mindspore/core/ops/transpose.h
            ${TOP_DIR}/mindspore/core/ops/base_operator.h
            ${TOP_DIR}/mindspore/core/ops/custom.h
            ${TOP_DIR}/mindspore/core/ops/add.h
            ${TOP_DIR}/mindspore/core/ops/arg_max.h
            ${TOP_DIR}/mindspore/core/ops/arg_min.h
            ${TOP_DIR}/mindspore/core/ops/avg_pool.h
            ${TOP_DIR}/mindspore/core/ops/conv2d.h
            ${TOP_DIR}/mindspore/core/ops/conv2d_transpose.h
            ${TOP_DIR}/mindspore/core/ops/div.h
            ${TOP_DIR}/mindspore/core/ops/mat_mul.h
            ${TOP_DIR}/mindspore/core/ops/max_pool.h
            ${TOP_DIR}/mindspore/core/ops/mul.h
            ${TOP_DIR}/mindspore/core/ops/pad.h
            ${TOP_DIR}/mindspore/core/ops/prelu.h
            ${TOP_DIR}/mindspore/core/ops/topk.h
            ${TOP_DIR}/mindspore/core/ops/relu.h
            ${TOP_DIR}/mindspore/core/ops/sigmoid.h
            DESTINATION ${RUNTIME_INC_DIR}/ops
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/core/mindapi/base/format.h
            ${TOP_DIR}/mindspore/core/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/core/mindapi/base/types.h
            ${TOP_DIR}/mindspore/core/mindapi/base/macros.h
            ${TOP_DIR}/mindspore/core/mindapi/base/shared_ptr.h
            ${TOP_DIR}/mindspore/core/mindapi/base/type_traits.h
            ${TOP_DIR}/mindspore/core/mindapi/base/base.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/core/mindapi/ir/common.h
            ${TOP_DIR}/mindspore/core/mindapi/ir/primitive.h
            ${TOP_DIR}/mindspore/core/mindapi/ir/value.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/ir
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/src/litert/cxx_api/kernel_executor/kernel_executor.h DESTINATION
            ${RUNTIME_INC_DIR}/api COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(TARGETS kernel_executor DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(TARGETS mindspore_core DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(MSLITE_ENABLE_CONVERTER)
        install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR}
            RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL ${RUNTIME_COMPONENT_NAME})
set(CPACK_PACKAGE_FILE_NAME ${PKG_NAME_PREFIX})

if(WIN32)
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output)
else()
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output/tmp)
endif()
set(CPACK_PACKAGE_CHECKSUM SHA256)
include(CPack)
