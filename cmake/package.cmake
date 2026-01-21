# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# set package information
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_GENERATOR "External")
set(CPACK_CMAKE_GENERATOR "Ninja")
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${BUILD_PATH}/package/mindspore)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${BUILD_PATH}/package/mindspore)
set(CPACK_PACK_ROOT_DIR ${BUILD_PATH}/package/)
set(CPACK_CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
set(CPACK_ENABLE_SYM_FILE ${ENABLE_SYM_FILE})
set(CPACK_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
set(CPACK_PYTHON_EXE ${Python3_EXECUTABLE})
set(CPACK_PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(ENABLE_GPU)
    set(CPACK_MS_BACKEND "ms")
elseif(ENABLE_D)
    set(CPACK_MS_BACKEND "ms")
elseif(ENABLE_CPU)
    set(CPACK_MS_BACKEND "ms")
else()
    set(CPACK_MS_BACKEND "debug")
endif()
set(CPACK_MS_PACKAGE_NAME "mindspore")
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "utils/bin")
set(INSTALL_CFG_DIR "config")
set(INSTALL_LIB_DIR "lib")
set(INSTALL_PLUGIN_DIR "${INSTALL_LIB_DIR}/plugin")
set(INSTALL_ASCEND_DIR "${INSTALL_PLUGIN_DIR}/ascend")
set(CUSTOM_ASCENDC_PREBUILD_DIR "${CMAKE_SOURCE_DIR}/mindspore/ops/kernel/ascend/ascendc/prebuild")
# set package files
install(
    TARGETS _c_expression
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindspore
)

install(
    TARGETS mindspore_core mindspore_ops mindspore_common mindspore_ms_backend mindspore_pyboost mindspore_pynative
        mindspore_backend_manager mindspore_hardware_abstract mindspore_frontend mindspore_profiler
        mindspore_memory_pool mindspore_runtime_pipeline mindspore_backend_common
        mindspore_runtime_utils mindspore_cluster mindspore_tools mindspore_pynative_utils mindspore_ops_host
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
)

if(ENABLE_TESTCASES)
    install(
        TARGETS proto_input
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore)
endif()

if(ENABLE_D)
    install(
        TARGETS mindspore_ge_backend
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_ascend_res_manager
        DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_ascend mindspore_ops_ascend LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
        NAMELINK_SKIP
    )
endif()

if(ENABLE_GPU)
    install(
            TARGETS mindspore_gpu LIBRARY
            DESTINATION ${INSTALL_PLUGIN_DIR}
            COMPONENT mindspore
            NAMELINK_SKIP
    )
endif()

if(USE_GLOG)
    install(FILES ${glog_LIBPATH}/libglog.so.0.7.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libglog.so.2 COMPONENT mindspore)
endif()

if(ENABLE_MINDDATA)
    install(
        TARGETS _c_dataengine _c_mindrecord
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore
    )
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        install(
            TARGETS dataset-cache dataset-cache-server
            OPTIONAL
            DESTINATION ${INSTALL_BIN_DIR}
            COMPONENT mindspore
        )
    endif()

    if(ENABLE_FFMPEG)
        install(FILES ${ffmpeg_LIBPATH}/libavcodec.so.59.37.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libavcodec.so.59 COMPONENT mindspore)
        install(FILES ${ffmpeg_LIBPATH}/libavdevice.so.59.7.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libavdevice.so.59 COMPONENT mindspore)
        install(FILES ${ffmpeg_LIBPATH}/libavfilter.so.8.44.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libavfilter.so.8 COMPONENT mindspore)
        install(FILES ${ffmpeg_LIBPATH}/libavformat.so.59.27.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libavformat.so.59 COMPONENT mindspore)
        install(FILES ${ffmpeg_LIBPATH}/libavutil.so.57.28.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libavutil.so.57 COMPONENT mindspore)
        install(FILES ${ffmpeg_LIBPATH}/libswresample.so.4.7.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libswresample.so.4 COMPONENT mindspore)
        install(FILES ${ffmpeg_LIBPATH}/libswscale.so.6.7.100
          DESTINATION ${INSTALL_LIB_DIR} RENAME libswscale.so.6 COMPONENT mindspore)
    endif()

    install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.11.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.so.411 COMPONENT mindspore)
    install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.11.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.so.411 COMPONENT mindspore)
    install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.11.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.so.411 COMPONENT mindspore)
    install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.so.10.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libtinyxml2.so.10 COMPONENT mindspore)

    install(FILES ${icu4c_LIBPATH}/libicuuc.so.74.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicuuc.so.74 COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicudata.so.74.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicudata.so.74 COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicui18n.so.74.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicui18n.so.74 COMPONENT mindspore)
    install(FILES ${jemalloc_LIBPATH}/libjemalloc.so.2 DESTINATION ${INSTALL_LIB_DIR} COMPONENT mindspore)
endif()

if(ENABLE_CPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        install(FILES ${onednn_LIBPATH}/libdnnl.so.2.2
          DESTINATION ${INSTALL_LIB_DIR} RENAME libdnnl.so.2 COMPONENT mindspore)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl*${CMAKE_SHARED_LIBRARY_SUFFIX}*)
        install(
            FILES ${DNNL_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
        )
    endif()
    install(
        TARGETS nnacl
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_cpu
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_ops_cpu LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
        NAMELINK_SKIP
    )
endif()

if(ENABLE_MPI)
    if(ENABLE_CPU)
        install(
          TARGETS mpi_collective
          DESTINATION ${INSTALL_LIB_DIR}
          COMPONENT mindspore
        )
    endif()
endif()

if(ENABLE_GPU)
    if(ENABLE_MPI)
        install(
          TARGETS nvidia_collective
          DESTINATION ${INSTALL_PLUGIN_DIR}/gpu${CUDA_VERSION}
          COMPONENT mindspore
        )
        if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND GPU_BACKEND_CUDA)
            install(FILES ${nccl_LIBPATH}/libnccl.so.2.16.5 DESTINATION ${INSTALL_PLUGIN_DIR}/gpu${CUDA_VERSION}
                    RENAME libnccl.so.2 COMPONENT mindspore)
        endif()
    endif()
    install(
            TARGETS cuda_ops LIBRARY
            DESTINATION ${INSTALL_PLUGIN_DIR}/gpu${CUDA_VERSION}
            COMPONENT mindspore
            NAMELINK_SKIP
    )
endif()

if(ENABLE_D)
    install(
        TARGETS mindspore_graph_ir LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
        COMPONENT mindspore
        NAMELINK_SKIP
    )
    if(EXISTS ${ASCEND_NNAL_ATB_PATH})
        install(
                TARGETS mindspore_atb_kernels mindspore_pyboost_atb_kernels LIBRARY
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
                NAMELINK_SKIP
        )
        install(
                TARGETS mindspore_extension_ascend_atb ARCHIVE
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
    endif()
    if(EXISTS ${ASCEND_NNAL_ASDSIP_PATH})
        install(
                TARGETS mindspore_extension_ascend_asdsip ARCHIVE
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
    endif()
    if(ENABLE_MPI)
        install(
                TARGETS d_collective
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
        if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
            install(
                    TARGETS mindspore_internal_kernels LIBRARY
                    DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                    COMPONENT mindspore
                    NAMELINK_SKIP
            )
            install(
                    TARGETS lowlatency_collective
                    DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                    COMPONENT mindspore
            )
        endif()
        if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
            install(
                    TARGETS mindspore_symmetric_memory_kernels LIBRARY
                    DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                    COMPONENT mindspore
                    NAMELINK_SKIP
            )
        endif()
    endif()
endif()

if(ENABLE_D OR ENABLE_ACL)
    if(DEFINED ENV{ASCEND_CUSTOM_PATH})
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH /usr/local/Ascend)
    endif()
    set(ASCEND_DRIVER_PATH ${ASCEND_PATH}/driver/lib64/common)

    if(ENABLE_D)
        install(
          TARGETS hccl_plugin
          DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
          COMPONENT mindspore
        )
    endif()
    if(ENABLE_ACL)
        install(
                TARGETS dvpp_utils
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
    endif()
endif()

if(MS_BUILD_GRPC)
    install(FILES ${grpc_LIBPATH}/libmindspore_grpc++.so.1.36.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_grpc++.so.1 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_grpc.so.15.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_grpc.so.15 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_gpr.so.15.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_gpr.so.15 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_upb.so.15.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_upb.so.15 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_address_sorting.so.15.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_address_sorting.so.15 COMPONENT mindspore)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB CXX_LIB_LIST ${CXX_DIR}/*.dll)

    string(REPLACE "\\" "/" SystemRoot $ENV{SystemRoot})
    file(GLOB VC_LIB_LIST ${SystemRoot}/System32/msvcp140.dll ${SystemRoot}/System32/vcomp140.dll)

    file(GLOB JPEG_LIB_LIST ${jpeg_turbo_LIBPATH}/*.dll)
    file(GLOB SQLITE_LIB_LIST ${sqlite_LIBPATH}/*.dll)
    install(
        FILES ${CXX_LIB_LIST} ${JPEG_LIB_LIST} ${SQLITE_LIB_LIST} ${VC_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
endif()

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/*.py)
install(
    FILES ${MS_PY_LIST}
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

file(GLOB NOTICE ${CMAKE_SOURCE_DIR}/Third_Party_Open_Source_Software_Notice)
install(
    FILES ${NOTICE}
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)
install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/nn
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/_extends
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/parallel
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/mindrecord
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/numpy
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/scipy
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/train
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/boost
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/common
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/ops_generate
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/communication
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/profiler
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/rewrite
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/safeguard
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/run_check
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/experimental
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/graph
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/mint
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/multiprocessing
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/hal
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/utils
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/tools
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/device_context
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/runtime
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/device_context
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/onnx
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

if(ENABLE_AKG AND CMAKE_SYSTEM_NAME MATCHES "Linux" AND NOT ENABLE_D)
    set (AKG_PATH ${BUILD_PATH}/mindspore/akg)
    file(REMOVE_RECURSE ${AKG_PATH}/_akg)
    file(MAKE_DIRECTORY ${AKG_PATH}/_akg)
    file(TOUCH ${AKG_PATH}/_akg/__init__.py)
    install(DIRECTORY "${akg_LIBPATH}/akg" DESTINATION "${AKG_PATH}/_akg")
    install(
        DIRECTORY
            ${AKG_PATH}/_akg
        DESTINATION ${INSTALL_PY_DIR}/
        COMPONENT mindspore
    )
    if(ENABLE_CPU AND NOT ENABLE_GPU)
        install(
                FILES ${akg_LIBPATH}/libakg.so
                DESTINATION ${INSTALL_PLUGIN_DIR}/cpu
                COMPONENT mindspore
        )
    endif()

    if(ENABLE_GPU)
        install(
                FILES ${akg_LIBPATH}/libakg.so
                DESTINATION ${INSTALL_PLUGIN_DIR}/gpu${CUDA_VERSION}
                COMPONENT mindspore
        )
    endif()
endif()

if(ENABLE_D)
    if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
        set(_MS_INTERNAL_KERNEL_HOME $ENV{MS_INTERNAL_KERNEL_HOME})
        install(
            DIRECTORY ${_MS_INTERNAL_KERNEL_HOME}
            DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
            COMPONENT mindspore
        )
    endif()
    if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
        set(_MS_SYMMETRIC_MEMORY_KERNEL_HOME $ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
        install(
            DIRECTORY ${_MS_SYMMETRIC_MEMORY_KERNEL_HOME}
            DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
            COMPONENT mindspore
        )
    endif()
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/dataset)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/dataset
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore
    )
endif()

## Public header files for minddata
install(
    FILES ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/config.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/constants.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/text.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/transforms.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_lite.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
    DESTINATION ${INSTALL_BASE_DIR}/include/dataset
    COMPONENT mindspore
)

install(
    FILES
        ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/base/format.h
        ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/base/type_id.h
        ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/base/types.h
        ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/base/shape_vector.h
        ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/base/macros.h
    DESTINATION ${INSTALL_BASE_DIR}/include/mindapi/base
    COMPONENT mindspore)

## ms header files
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/core
    DESTINATION ${INSTALL_BASE_DIR}/include/mindspore
    COMPONENT mindspore
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
)
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/ops
    DESTINATION ${INSTALL_BASE_DIR}/include/mindspore
    COMPONENT mindspore
    FILES_MATCHING PATTERN "*.h"
)
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/ccsrc
    DESTINATION ${INSTALL_BASE_DIR}/include/mindspore
    COMPONENT mindspore
    FILES_MATCHING PATTERN "*.h"
)
install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/include
        DESTINATION ${INSTALL_BASE_DIR}/include/mindspore
        COMPONENT mindspore
        FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${securec_INC}
    DESTINATION ${INSTALL_BASE_DIR}/include/third_party
    COMPONENT mindspore
    FILES_MATCHING PATTERN "*.h")

## msextension for custom ops
install(FILES ${CMAKE_SOURCE_DIR}/mindspore/include/custom_op_api.h
    DESTINATION ${INSTALL_BASE_DIR}/include/mindspore/ccsrc/ms_extension
    RENAME all.h
    COMPONENT mindspore)

install(FILES ${CMAKE_SOURCE_DIR}/mindspore/include/custom_op_api.h
        DESTINATION ${INSTALL_BASE_DIR}/include/mindspore/ccsrc/ms_extension
        RENAME api.h
        COMPONENT mindspore)

install(FILES ${CMAKE_SOURCE_DIR}/mindspore/include/custom_op_api.h
        DESTINATION ${INSTALL_BASE_DIR}/include/mindspore/ccsrc/include
        RENAME ms_extension.h
        COMPONENT mindspore)
## third-party header files
install(DIRECTORY ${pybind11_INC}/pybind11
    DESTINATION ${INSTALL_BASE_DIR}/include/third_party
    COMPONENT mindspore)
install(DIRECTORY ${robin_hood_hashing_INC}/include
    DESTINATION ${INSTALL_BASE_DIR}/include/third_party/robin_hood_hashing
    COMPONENT mindspore)
if (NOT ENABLE_NATIVE_JSON)
    install(DIRECTORY ${nlohmann_json3101_INC}/nlohmann
        DESTINATION ${INSTALL_BASE_DIR}/include/third_party
        COMPONENT mindspore)
endif()

## config files
install(
    FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
    DESTINATION ${INSTALL_CFG_DIR}
    COMPONENT mindspore
)

if(ENABLE_AIO)
    install(
        TARGETS aio_plugin
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
endif()

if(ENABLE_D)
    install(
        DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/custom_compiler
        ${CUSTOM_ASCENDC_PREBUILD_DIR}/${CMAKE_SYSTEM_PROCESSOR}/custom_ascendc_ops/custom_ascendc_910b
        DESTINATION ${INSTALL_ASCEND_DIR}
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_extension_ascend_aclnn ARCHIVE
        DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
        COMPONENT mindspore
    )
endif()
