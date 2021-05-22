include(CMakePackageConfigHelpers)

set(RUNTIME_PKG_NAME ${PKG_NAME_PREFIX}-${RUNTIME_COMPONENT_NAME})

set(CODEGEN_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/codegen)
set(CONVERTER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/converter)
set(OBFUSCATOR_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/obfuscator)
set(CROPPER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/cropper)

if(SUPPORT_TRAIN)
    set(RUNTIME_DIR ${RUNTIME_PKG_NAME}/train)
    set(RUNTIME_INC_DIR ${RUNTIME_PKG_NAME}/train/include)
    set(RUNTIME_LIB_DIR ${RUNTIME_PKG_NAME}/train/lib)
    set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/train/include/dataset)
    set(TURBO_DIR ${RUNTIME_PKG_NAME}/train/third_party/libjpeg-turbo)
    set(SECUREC_DIR ${RUNTIME_PKG_NAME}/train/third_party/securec)
    set(MINDSPORE_LITE_LIB_NAME libmindspore-lite-train)
    set(BENCHMARK_NAME benchmark_train)
    set(BENCHMARK_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark_train)
else()
    set(RUNTIME_DIR ${RUNTIME_PKG_NAME}/inference)
    set(RUNTIME_INC_DIR ${RUNTIME_PKG_NAME}/inference/include)
    set(RUNTIME_LIB_DIR ${RUNTIME_PKG_NAME}/inference/lib)
    set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/inference/include/dataset)
    set(TURBO_DIR ${RUNTIME_PKG_NAME}/inference/third_party/libjpeg-turbo)
    set(SECUREC_DIR ${RUNTIME_PKG_NAME}/inference/third_party/securec)
    set(MINDSPORE_LITE_LIB_NAME libmindspore-lite)
    set(BENCHMARK_NAME benchmark)
    set(BENCHMARK_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark)
endif()

if(BUILD_MINDDATA STREQUAL "full")
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
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite_static.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite_static.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite_static.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.3.0 DESTINATION ${TURBO_DIR}/lib
                RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.2.0 DESTINATION ${TURBO_DIR}/lib
                RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/securec/src/libsecurec.a
                DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
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

if(WIN32)
    install(FILES ${TOP_DIR}/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
else()
    install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
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
    endif()
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/registry/libmslite_kernel_reg.so
            DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/android-aarch64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
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
    if(ENABLE_TOOLS)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
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
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/registry/libmslite_kernel_reg.so
            DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/android-aarch32/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
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
    if(ENABLE_TOOLS)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
elseif(WIN32)
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB LIB_LIST ${CXX_DIR}/libstdc++-6.dll ${CXX_DIR}/libwinpthread-1.dll
            ${CXX_DIR}/libssp-0.dll ${CXX_DIR}/libgcc_s_*-1.dll)
    if(ENABLE_CONVERTER)
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/registry/libmslite_converter_plugin_reg.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/registry/libmslite_kernel_reg.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/../bin/libglog.dll DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
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
    endif()
    if(ENABLE_TOOLS)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${LIB_LIST} DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${flatbuffers_INC} DESTINATION ${RUNTIME_INC_DIR}/third_party/
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/src/registry/libmslite_kernel_reg.dll DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
else()
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/registry/libmslite_kernel_reg.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/bin/linux-x64/msobfuscator
                DESTINATION ${OBFUSCATOR_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/tools/obfuscator/lib/linux-x64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(ENABLE_CONVERTER)
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/registry/libmslite_converter_plugin_reg.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/src/registry/libmslite_kernel_reg.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libglog.so.0.4.0 DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libglog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
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
    endif()
    if(ENABLE_TOOLS)
        install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
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
