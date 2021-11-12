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
set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include/dataset)
set(TURBO_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/libjpeg-turbo)
set(SECUREC_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/securec)
set(MINDSPORE_LITE_LIB_NAME libmindspore-lite)
set(MINDSPORE_CORE_LIB_NAME libmindspore_core)
set(BENCHMARK_NAME benchmark)
set(BENCHMARK_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark)

set(MINDSPORE_LITE_TRAIN_LIB_NAME libmindspore-lite-train)
set(BENCHMARK_TRAIN_NAME benchmark_train)
set(BENCHMARK_TRAIN_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark_train)

# full mode will also package the files of lite_cv mode.
if(BUILD_MINDDATA STREQUAL "full")
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
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
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

if(BUILD_MINDDATA STREQUAL "wrapper")
    install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "vision.h" EXCLUDE)
    if(PLATFORM_ARM64)
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
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

if(BUILD_MINDDATA STREQUAL "lite")
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

if(BUILD_MINDDATA STREQUAL "lite_cv")
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
    install(DIRECTORY ${TOP_DIR}/mindspore/lite/micro/coder/wrapper DESTINATION ${CODEGEN_ROOT_DIR}/include
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
    install(TARGETS codegen RUNTIME DESTINATION ${CODEGEN_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
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
    endif()
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "*registry.h" EXCLUDE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "*registry.h" EXCLUDE)
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
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    __install_micro_wrapper()
    if(MSLITE_ENABLE_TOOLS)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_TESTCASES)
        install(FILES ${TOP_DIR}/mindspore/lite/build/test/lite-test DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
elseif(PLATFORM_ARM32)
    if(SUPPORT_NPU)
        install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "*registry.h" EXCLUDE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "*registry.h" EXCLUDE)
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
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    __install_micro_wrapper()
    if(MSLITE_ENABLE_TOOLS AND NOT TARGET_OHOS_LITE)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
elseif(WIN32)
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB LIB_LIST ${CXX_DIR}/libstdc++-6.dll ${CXX_DIR}/libwinpthread-1.dll
            ${CXX_DIR}/libssp-0.dll ${CXX_DIR}/libgcc_s_*-1.dll)
    if(MSLITE_ENABLE_CONVERTER)
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h"
                PATTERN "train*" EXCLUDE PATTERN "delegate.h" EXCLUDE PATTERN "lite_session.h" EXCLUDE)
        install(FILES ${LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/registry/libmslite_converter_plugin.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/../bin/libglog.dll DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/core/abstract/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/abstract
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/base/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/base
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/ir/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/ops/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ops
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/utils/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/utils
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
        install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/ccsrc/backend/optimizer/common/pass.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/converter/model_parser.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/converter/dump_graph.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/converter/ops/ops_def.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/build/mindspore/schema/ DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/third_party/securec/include/
                DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${protobuf_INC}/google DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${eigen3_INC}/eigen3 DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/mindspore_core/${MINDSPORE_CORE_LIB_NAME}.a
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/mindspore_core/gvar/libmindspore_gvar.a
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/securec/src/libsecurec.a DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${protobuf_LIBPATH}/libprotobuf.a DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        __install_micro_wrapper()
        __install_micro_codegen()
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
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "*registry.h" EXCLUDE)
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "*registry.h" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/build/mindspore/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
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
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "*registry.h" EXCLUDE
                PATTERN "framework.h" EXCLUDE)
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE
                PATTERN "*registry.h" EXCLUDE PATTERN "framework.h" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
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
    if(MSLITE_ENABLE_CONVERTER)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h"
                PATTERN "train*" EXCLUDE PATTERN "delegate.h" EXCLUDE PATTERN "lite_session.h" EXCLUDE)
        install(DIRECTORY ${TOP_DIR}/mindspore/core/abstract/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/abstract
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/base/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/base
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/ir/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/ops/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/ops
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/mindspore/core/utils/ DESTINATION ${CONVERTER_ROOT_DIR}/include/core/utils
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
        install(DIRECTORY ${TOP_DIR}/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/ccsrc/backend/optimizer/common/pass.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/converter/model_parser.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/converter/dump_graph.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/converter/ops/ops_def.h
                DESTINATION ${CONVERTER_ROOT_DIR}/include COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/build/schema/ DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${TOP_DIR}/third_party/securec/include/
                DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${protobuf_INC}/google DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${openssl_INC}/openssl DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${eigen3_INC}/eigen3 DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/mindspore_core/${MINDSPORE_CORE_LIB_NAME}.a
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/mindspore_core/gvar/libmindspore_gvar.a
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${protobuf_LIBPATH}/libprotobuf.a DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${openssl_LIBPATH}/libcrypto.a DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libglog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_NNIE)
            install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libglog.so
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        __install_micro_wrapper()
        __install_micro_codegen()
    endif()
    if(MSLITE_ENABLE_TOOLS)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
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
