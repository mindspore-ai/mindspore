include(CMakePackageConfigHelpers)

set(RUNTIME_PKG_NAME ${PKG_NAME_PREFIX}-${RUNTIME_COMPONENT_NAME})

set(CODEGEN_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/codegen)
set(CONVERTER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/converter)
set(OBFUSCATOR_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/obfuscator)
set(CROPPER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/cropper)
set(TEST_CASE_DIR ${TOP_DIR}/mindspore/lite/test/build)

set(RUNTIME_DIR ${RUNTIME_PKG_NAME}/runtime)
set(RUNTIME_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include)
set(RUNTIME_LIB_DIR ${RUNTIME_PKG_NAME}/runtime/lib)
set(PROVIDERS_LIB_DIR ${RUNTIME_PKG_NAME}/providers)
set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include/dataset)
set(TURBO_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/libjpeg-turbo)
set(GLOG_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/glog)
set(SECUREC_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/securec)
set(MINDSPORE_LITE_LIB_NAME libmindspore-lite)
set(MINDSPORE_CORE_LIB_NAME libmindspore_core)
set(BENCHMARK_NAME benchmark)
set(MSLITE_NNIE_LIB_NAME libmslite_nnie)
set(MSLITE_PROPOSAL_LIB_NAME libmslite_proposal)
set(MICRO_NNIE_LIB_NAME libmicro_nnie)
set(DPICO_ACL_ADAPTER_LIB_NAME libdpico_acl_adapter)
set(BENCHMARK_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark)
set(MICRO_DIR ${TOP_DIR}/mindspore/lite/tools/converter/micro)

set(MINDSPORE_LITE_TRAIN_LIB_NAME libmindspore-lite-train)
set(BENCHMARK_TRAIN_NAME benchmark_train)
set(BENCHMARK_TRAIN_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark_train)
file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)

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

function(__install_micro_wrapper)
    file(GLOB NNACL_FILES GLOB ${NNACL_DIR}/*.h)
    install(FILES ${NNACL_FILES} DESTINATION ${CODEGEN_ROOT_DIR}/include/nnacl COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${NNACL_DIR}/base DESTINATION ${CODEGEN_ROOT_DIR}/include/nnacl
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${NNACL_DIR}/int8 DESTINATION ${CODEGEN_ROOT_DIR}/include/nnacl
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${NNACL_DIR}/fp32 DESTINATION ${CODEGEN_ROOT_DIR}/include/nnacl
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${NNACL_DIR}/intrinsics DESTINATION ${CODEGEN_ROOT_DIR}/include/nnacl
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${MICRO_DIR}/coder/wrapper DESTINATION ${CODEGEN_ROOT_DIR}/include
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(TARGETS wrapper ARCHIVE DESTINATION ${CODEGEN_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
endfunction()

function(__install_micro_codegen)
    set(MICRO_CMSIS_DIR ${CMAKE_BINARY_DIR}/cmsis/CMSIS)
    install(DIRECTORY ${MICRO_CMSIS_DIR}/Core/Include DESTINATION ${CODEGEN_ROOT_DIR}/third_party/include/CMSIS/Core
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${MICRO_CMSIS_DIR}/DSP/Include DESTINATION ${CODEGEN_ROOT_DIR}/third_party/include/CMSIS/DSP
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(DIRECTORY ${MICRO_CMSIS_DIR}/NN/Include DESTINATION ${CODEGEN_ROOT_DIR}/third_party/include/CMSIS/NN
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    install(TARGETS cmsis_nn ARCHIVE DESTINATION ${CODEGEN_ROOT_DIR}/third_party/lib
            COMPONENT ${RUNTIME_COMPONENT_NAME})
endfunction()

if(WIN32)
    install(FILES ${TOP_DIR}/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
else()
    install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
endif()
install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${RUNTIME_INC_DIR}/third_party COMPONENT ${RUNTIME_COMPONENT_NAME})
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
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "registry*" EXCLUDE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "registry*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_MODEL_OBF)
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
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED OR MSLITE_ENABLE_CONVERTER)
        __install_micro_wrapper()
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${GLOG_DIR} RENAME libglog.so.0
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
                endif()
            elseif(TARGET_MIX210)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "SD3403")
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
            install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${CONVERTER_ROOT_DIR}/include
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h"
                    PATTERN "train*" EXCLUDE PATTERN "delegate.h" EXCLUDE PATTERN "lite_session.h" EXCLUDE)
            install(FILES ${API_HEADER}  DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${MINDAPI_BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/mindapi/base
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${MINDAPI_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/mindapi/ir
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${ABSTRACT_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/abstract
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${API_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/api/ir
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/base
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${IR_DTYPE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ir/dtype
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ir
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(DIRECTORY ${TOP_DIR}/mindspore/core/ops/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ops
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(FILES ${UTILS_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/utils
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
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
            install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libglog.so.0
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
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
                    install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR} RENAME libglog.so.0
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(TARGETS mindspore_core DESTINATION ${CONVERTER_ROOT_DIR}/lib
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
                if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "cloud" AND MSLITE_ENABLE_RUNTIME_CONVERT)
                    file(GLOB DATA_ENGINE_LIB_LIST ${LITE_ACL_DIR}/_c_dataengine/*.so)
                    file(GLOB DATA_RECORD_LIB_LIST ${LITE_ACL_DIR}/_c_mindrecord/*.so)
                    install(FILES ${DATA_ENGINE_LIB_LIST}
                            DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${DATA_RECORD_LIB_LIST}
                            DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.3.0
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.2.0
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.so.8.0.0
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libtinyxml2.so.8 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${icu4c_LIBPATH}/libicuuc.so.69.1
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libicuuc.so.69 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${icu4c_LIBPATH}/libicudata.so.69.1
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libicudata.so.69 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${icu4c_LIBPATH}/libicui18n.so.69.1
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libicui18n.so.69 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${grpc_LIBPATH}/libmindspore_grpc++.so.1.36.1 DESTINATION ${RUNTIME_LIB_DIR}
                            RENAME libmindspore_grpc++.so.1 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${grpc_LIBPATH}/libmindspore_grpc.so.15.0.0 DESTINATION
                            ${RUNTIME_LIB_DIR} RENAME libmindspore_grpc.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${grpc_LIBPATH}/libmindspore_gpr.so.15.0.0 DESTINATION
                            ${RUNTIME_LIB_DIR} RENAME libmindspore_gpr.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${grpc_LIBPATH}/libmindspore_upb.so.15.0.0 DESTINATION
                            ${RUNTIME_LIB_DIR} RENAME libmindspore_upb.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${grpc_LIBPATH}/libmindspore_address_sorting.so.15.0.0 DESTINATION ${RUNTIME_LIB_DIR}
                            RENAME libmindspore_address_sorting.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    ## Public header files for minddata
                    install(
                            FILES ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/config.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/constants.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/text.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/transforms.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_lite.h
                            ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
                            DESTINATION ${RUNTIME_INC_DIR}/dataset COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
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
                install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${GLOG_DIR} RENAME libglog.so.0
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
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
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "registry*" EXCLUDE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "registry*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_MODEL_OBF)
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
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED OR MSLITE_ENABLE_CONVERTER)
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
                    if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3516D")
                        install(FILES
                                ${TOP_DIR}/mindspore/lite/providers/nnie/third_patry/${MICRO_NNIE_LIB_NAME}.so
                                DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                                COMPONENT ${RUNTIME_COMPONENT_NAME})
                    endif()
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
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/converter_lite.exe
                DESTINATION ${CONVERTER_ROOT_DIR}/converter COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/registry/libmslite_converter_plugin.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/../bin/libglog.dll DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        file(GLOB_RECURSE OPENCV_LIB_LIST
                ${opencv_LIBPATH}/../bin/libopencv_core*
                ${opencv_LIBPATH}/../bin/libopencv_imgcodecs*
                ${opencv_LIBPATH}/../bin/libopencv_imgproc*
                )
        install(FILES ${OPENCV_LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(NOT MSVC)
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
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "registry*" EXCLUDE)
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "registry*" EXCLUDE)
    endif()
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
else()
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "registry*" EXCLUDE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "registry*" EXCLUDE)
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
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/bin/linux-x64/msobfuscator
                DESTINATION ${OBFUSCATOR_ROOT_DIR} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/linux-x64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${GLOG_DIR} RENAME libglog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(MSLITE_ENABLE_CONVERTER)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h"
                PATTERN "train*" EXCLUDE PATTERN "delegate.h" EXCLUDE PATTERN "lite_session.h" EXCLUDE)
        install(FILES ${API_HEADER}  DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${MINDAPI_BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/mindapi/base
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${MINDAPI_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/mindapi/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${ABSTRACT_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/abstract
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${API_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/api/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/base
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${IR_DTYPE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ir/dtype
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/core/ops/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ops
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${UTILS_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/core/utils
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/build/schema/ DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "schema_generated.h" EXCLUDE)
        install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/third_party/securec/include/
                DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libglog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
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
                install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${RUNTIME_LIB_DIR} RENAME libglog.so.0
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(TARGETS mindspore_core DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "cloud" AND MSLITE_ENABLE_RUNTIME_CONVERT)
                file(GLOB DATA_ENGINE_LIB_LIST ${LITE_ACL_DIR}/_c_dataengine/*.so)
                file(GLOB DATA_RECORD_LIB_LIST ${LITE_ACL_DIR}/_c_mindrecord/*.so)
                install(FILES ${DATA_ENGINE_LIB_LIST}
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${DATA_RECORD_LIB_LIST}
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.3.0
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.2.0
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.so.8.0.0
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libtinyxml2.so.8 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${icu4c_LIBPATH}/libicuuc.so.67.1
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libicuuc.so.67 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${icu4c_LIBPATH}/libicudata.so.67.1
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libicudata.so.67 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${icu4c_LIBPATH}/libicui18n.so.67.1
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libicui18n.so.67 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${grpc_LIBPATH}/libmindspore_grpc++.so.1.36.1 DESTINATION ${RUNTIME_LIB_DIR}
                        RENAME libmindspore_grpc++.so.1 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${grpc_LIBPATH}/libmindspore_grpc.so.15.0.0 DESTINATION
                        ${RUNTIME_LIB_DIR} RENAME libmindspore_grpc.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${grpc_LIBPATH}/libmindspore_gpr.so.15.0.0 DESTINATION
                        ${RUNTIME_LIB_DIR} RENAME libmindspore_gpr.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${grpc_LIBPATH}/libmindspore_upb.so.15.0.0 DESTINATION
                        ${RUNTIME_LIB_DIR} RENAME libmindspore_upb.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${grpc_LIBPATH}/libmindspore_address_sorting.so.15.0.0 DESTINATION ${RUNTIME_LIB_DIR}
                        RENAME libmindspore_address_sorting.so.15 COMPONENT ${RUNTIME_COMPONENT_NAME})
                ## Public header files for minddata
                install(
                        FILES ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/config.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/constants.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/text.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/transforms.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_lite.h
                        ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
                        DESTINATION ${RUNTIME_INC_DIR}/dataset COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
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
            install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${GLOG_DIR} RENAME libglog.so.0
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
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
        __install_micro_wrapper()
        __install_micro_codegen()
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
