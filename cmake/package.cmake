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
if(BUILD_DEV_MODE)
    # providing cuda11 version of dev package only
    set(CPACK_MS_PACKAGE_NAME "mindspore-dev")
else()
    set(CPACK_MS_PACKAGE_NAME "mindspore")
endif()
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "bin")
set(INSTALL_CFG_DIR "config")
set(INSTALL_LIB_DIR "lib")
set(INSTALL_PLUGIN_DIR "${INSTALL_LIB_DIR}/plugin")
# set package files
install(
    TARGETS _c_expression
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindspore
)

if(ENABLE_DEBUGGER)
    install(
        TARGETS _mindspore_offline_debug
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore
    )
endif()

install(
    TARGETS mindspore_shared_lib
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
)

if(ENABLE_D OR ENABLE_GPU)
    install(
            TARGETS api_lib
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
endif()

install(
    TARGETS mindspore_core mindspore_common mindspore_backend
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
)

if(ENABLE_D)
    install(
        TARGETS mindspore_ascend
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
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
    install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_glog.so.0 COMPONENT mindspore)
endif()

if(ENABLE_MINDDATA)
    install(
        TARGETS _c_dataengine _c_mindrecord
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore
    )
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        install(
            TARGETS cache_admin cache_server
            OPTIONAL
            DESTINATION ${INSTALL_BIN_DIR}
            COMPONENT mindspore
        )
    endif()
    install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
      DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.so.4.5 COMPONENT mindspore)
    install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
      DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5 COMPONENT mindspore)
    install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
      DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.so.4.5 COMPONENT mindspore)
    install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.so.8.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libtinyxml2.so.8 COMPONENT mindspore)

    install(FILES ${icu4c_LIBPATH}/libicuuc.so.69.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicuuc.so.69 COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicudata.so.69.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicudata.so.69 COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicui18n.so.69.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicui18n.so.69 COMPONENT mindspore)
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
endif()

if(ENABLE_MPI)
    if(ENABLE_CPU)
        install(
            TARGETS mpi_adapter
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
        )
        install(
          TARGETS mpi_collective
          DESTINATION ${INSTALL_LIB_DIR}
          COMPONENT mindspore
        )
    endif()
    if(ENABLE_D)
        install(
                TARGETS _ascend_mpi
                DESTINATION ${INSTALL_BASE_DIR}
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
            install(FILES ${nccl_LIBPATH}/libnccl.so.2.7.6 DESTINATION ${INSTALL_PLUGIN_DIR}/gpu${CUDA_VERSION}
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
    if(ENABLE_MPI)
        install(
                TARGETS ascend_collective
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
    endif()
endif()

if(ENABLE_CPU AND NOT WIN32)
    install(
        TARGETS ps_cache
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
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
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/communication
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/profiler
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/rewrite
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/run_check
        ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/experimental
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

if(ENABLE_AKG AND CMAKE_SYSTEM_NAME MATCHES "Linux")
    set (AKG_PATH ${BUILD_PATH}/mindspore/akg)
    file(REMOVE_RECURSE ${AKG_PATH}/_akg)
    file(MAKE_DIRECTORY ${AKG_PATH}/_akg)
    file(TOUCH ${AKG_PATH}/_akg/__init__.py)
    install(DIRECTORY "${AKG_PATH}/akg" DESTINATION "${AKG_PATH}/_akg")
    install(
        DIRECTORY
            ${AKG_PATH}/_akg
        DESTINATION ${INSTALL_PY_DIR}/
        COMPONENT mindspore
    )
    if(ENABLE_CPU AND NOT ENABLE_GPU AND NOT ENABLE_D)
        install(
                TARGETS akg
                DESTINATION ${INSTALL_PLUGIN_DIR}/cpu
                COMPONENT mindspore
        )
    endif()

    if(ENABLE_GPU)
        install(
                TARGETS akg
                DESTINATION ${INSTALL_PLUGIN_DIR}/gpu${CUDA_VERSION}
                COMPONENT mindspore
        )
    endif()

    if(ENABLE_D)
        install(
                TARGETS akg
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

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("offline debugger does not support windows system temporarily")
else()
    if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/offline_debug)
        install(
            DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/offline_debug
            DESTINATION ${INSTALL_PY_DIR}
            COMPONENT mindspore
        )
    endif()
endif()

## Public header files
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindspore
)

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
        ${CMAKE_SOURCE_DIR}/mindspore/core/mindapi/base/format.h
        ${CMAKE_SOURCE_DIR}/mindspore/core/mindapi/base/type_id.h
        ${CMAKE_SOURCE_DIR}/mindspore/core/mindapi/base/types.h
    DESTINATION ${INSTALL_BASE_DIR}/include/mindapi/base
    COMPONENT mindspore)

## config files
install(
    FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
          ${CMAKE_SOURCE_DIR}/config/super_bar_config.json
    DESTINATION ${INSTALL_CFG_DIR}
    COMPONENT mindspore
)

