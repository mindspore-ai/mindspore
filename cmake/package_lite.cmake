include(CMakePackageConfigHelpers)

set(RUNTIME_PKG_NAME ${MAIN_DIR}-${RUNTIME_COMPONENT_NAME})
set(CONVERTER_PKG_NAME ${MAIN_DIR}-${CONVERTER_COMPONENT_NAME})

set(RUNTIME_ROOT_DIR ${RUNTIME_PKG_NAME}/)
set(CONVERTER_ROOT_DIR ${CONVERTER_PKG_NAME}/)
set(RUNTIME_LIB_DIR ${RUNTIME_PKG_NAME}/lib)
set(RUNTIME_INC_DIR ${RUNTIME_PKG_NAME}/include)
set(CONVERTER_LIB_DIR ${CONVERTER_PKG_NAME}/lib)

set(TURBO_DIR ${RUNTIME_PKG_NAME}/minddata/third_party/libjpeg-turbo)

set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/minddata/include)
set(MIND_DATA_LIB_DIR ${RUNTIME_PKG_NAME}/minddata/lib)

set(LIB_DIR_RUN_X86 ${RUNTIME_PKG_NAME}/lib)

if(BUILD_MINDDATA STREQUAL "full" OR BUILD_MINDDATA STREQUAL "wrapper")
    install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "vision.h" EXCLUDE)
    if(PLATFORM_ARM64)
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.3.0 DESTINATION ${TURBO_DIR}/lib  RENAME libjpeg.so.62
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.2.0 DESTINATION ${TURBO_DIR}/lib RENAME libturbojpeg.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(BUILD_MINDDATA STREQUAL "lite")
    install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(PLATFORM_ARM64)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
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
                DESTINATION ${MIND_DATA_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${TOP_DIR}/mindspore/lite/build/minddata/libminddata-lite.so DESTINATION ${MIND_DATA_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(PLATFORM_ARM64)
    install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(SUPPORT_NPU)
        install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${RUNTIME_PKG_NAME}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${RUNTIME_PKG_NAME}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${RUNTIME_PKG_NAME}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_TOOLS)
        install(TARGETS benchmark RUNTIME DESTINATION ${RUNTIME_PKG_NAME}/benchmark COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
elseif(PLATFORM_ARM32)
    install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(SUPPORT_TRAIN)
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore/lite/include/ DESTINATION ${RUNTIME_INC_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "train*" EXCLUDE)
    endif()
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/core/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_TOOLS)
        install(TARGETS benchmark RUNTIME DESTINATION ${RUNTIME_PKG_NAME}/benchmark COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
elseif(WIN32)
    install(FILES ${TOP_DIR}/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB LIB_LIST ${CXX_DIR}/libstdc++-6.dll ${CXX_DIR}/libwinpthread-1.dll
            ${CXX_DIR}/libssp-0.dll ${CXX_DIR}/libgcc_s_seh-1.dll)
    if(ENABLE_CONVERTER)
        install(FILES ${TOP_DIR}/build/.commit_id DESTINATION ${CONVERTER_PKG_NAME}
                COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_PKG_NAME}/converter
                COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${CONVERTER_PKG_NAME}/converter COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/mindspore_core/gvar/libmindspore_gvar.dll
                DESTINATION ${CONVERTER_PKG_NAME}/converter COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/../bin/libglog.dll DESTINATION ${CONVERTER_PKG_NAME}/converter
                COMPONENT ${CONVERTER_COMPONENT_NAME})
    endif()
    if(ENABLE_TOOLS)
        install(TARGETS benchmark RUNTIME DESTINATION ${RUNTIME_PKG_NAME}/benchmark COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${RUNTIME_PKG_NAME}/benchmark COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${flatbuffers_INC} DESTINATION ${RUNTIME_PKG_NAME}/third_party/flatbuffers
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
        set(WIN_LIB_DIR_RUN_X86 ${RUNTIME_PKG_NAME}/benchmark)
        install(FILES ${TOP_DIR}/build/mindspore/src/libmindspore-lite.a DESTINATION ${WIN_LIB_DIR_RUN_X86}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/libmindspore-lite.dll.a DESTINATION ${WIN_LIB_DIR_RUN_X86}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/libmindspore-lite.dll DESTINATION ${WIN_LIB_DIR_RUN_X86}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
else()
    install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
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
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.so DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/lite/build/src/libmindspore-lite.a DESTINATION ${RUNTIME_LIB_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(ENABLE_CONVERTER)
        install(FILES ${TOP_DIR}/mindspore/lite/build/.commit_id DESTINATION ${CONVERTER_PKG_NAME}
                COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_PKG_NAME}/converter
                COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/converter/mindspore_core/gvar/libmindspore_gvar.so
                DESTINATION ${CONVERTER_PKG_NAME}/lib COMPONENT ${CONVERTER_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/libglog.so.0.4.0
                DESTINATION ${CONVERTER_PKG_NAME}/third_party/glog/lib RENAME libglog.so.0
                COMPONENT ${CONVERTER_COMPONENT_NAME})
    endif()
    if(ENABLE_TOOLS)
        install(TARGETS benchmark RUNTIME DESTINATION ${RUNTIME_PKG_NAME}/benchmark COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS cropper RUNTIME DESTINATION ${RUNTIME_PKG_NAME}/cropper COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore/lite/build/tools/cropper/cropper_mapping_cpu.cfg
                DESTINATION ${RUNTIME_PKG_NAME}/cropper COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    set(CPACK_COMPONENTS_ALL ${RUNTIME_COMPONENT_NAME})
else()
    set(CPACK_COMPONENTS_ALL ${RUNTIME_COMPONENT_NAME} ${CONVERTER_COMPONENT_NAME})
endif()
set(CPACK_PACKAGE_FILE_NAME ${MAIN_DIR})
if(WIN32)
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output)
else()
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output/tmp)
endif()
set(CPACK_PACKAGE_CHECKSUM SHA256)
include(CPack)
