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
if(ENABLE_GE)
    set(CPACK_MS_BACKEND "ge")
    set(CPACK_MS_TARGET "ascend or cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore")
elseif(ENABLE_GPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "gpu or cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore-gpu")
elseif(ENABLE_D)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "ascend or cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore-ascend")
elseif(ENABLE_CPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore")
elseif(ENABLE_ACL)
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend or gpu or cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore-ascend")
else()
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend or gpu or cpu")
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

install(
    TARGETS mindspore_gvar
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
)

if(USE_GLOG)
    install(FILES ${glog_LIBPATH}/libmindspore_glog.so.0.4.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_glog.so.0 COMPONENT mindspore)
endif()

install(FILES ${libevent_LIBPATH}/libevent-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent-2.1.so.7 COMPONENT mindspore)
install(FILES ${libevent_LIBPATH}/libevent_core-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_core-2.1.so.7 COMPONENT mindspore)
install(FILES ${libevent_LIBPATH}/libevent_extra-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_extra-2.1.so.7 COMPONENT mindspore)
install(FILES ${libevent_LIBPATH}/libevent_openssl-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_openssl-2.1.so.7 COMPONENT mindspore)
install(FILES ${libevent_LIBPATH}/libevent_pthreads-2.1.so.7.0.1
  DESTINATION ${INSTALL_LIB_DIR} RENAME libevent_pthreads-2.1.so.7 COMPONENT mindspore)

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
    if(PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7")
      install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.2.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.so.4.2 COMPONENT mindspore)
      install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.2.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.so.4.2 COMPONENT mindspore)
      install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.2.0
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.so.4.2 COMPONENT mindspore)
    elseif(PYTHON_VERSION MATCHES "3.9")
      install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.so.4.5 COMPONENT mindspore)
      install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5 COMPONENT mindspore)
      install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.1
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.so.4.5 COMPONENT mindspore)
    endif()
    install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.so.8.0.0
      DESTINATION ${INSTALL_LIB_DIR} RENAME libtinyxml2.so.8 COMPONENT mindspore)

    install(FILES ${icu4c_LIBPATH}/libicuuc.so.67.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicuuc.so.67 COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicudata.so.67.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicudata.so.67 COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicui18n.so.67.1
      DESTINATION ${INSTALL_LIB_DIR} RENAME libicui18n.so.67 COMPONENT mindspore)
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
    if(ENABLE_GPU)
        install(
            TARGETS _ms_mpi
            DESTINATION ${INSTALL_BASE_DIR}
            COMPONENT mindspore
        )
    endif()
    if(ENABLE_CPU)
        install(
            TARGETS mpi_adapter
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
            TARGETS gpu_collective
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
        )
    endif()
    install(
        TARGETS gpu_queue
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
endif()

if(ENABLE_D)
    if(ENABLE_MPI)
        install(
                TARGETS ascend_collective
                DESTINATION ${INSTALL_LIB_DIR}
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

if(NOT ENABLE_GE)
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
              DESTINATION ${INSTALL_LIB_DIR}
              COMPONENT mindspore
            )
        endif()
    elseif(ENABLE_TESTCASES)
        install(
            FILES
                ${CMAKE_BINARY_DIR}/graphengine/metadef/graph/libgraph.so
                ${BUILD_PATH}/graphengine/c_sec/lib/libc_sec.so
            DESTINATION ${INSTALL_LIB_DIR}
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
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore/*.py)
install(
    FILES ${MS_PY_LIST}
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

install(
    DIRECTORY
        ${CMAKE_SOURCE_DIR}/mindspore/nn
        ${CMAKE_SOURCE_DIR}/mindspore/_extends
        ${CMAKE_SOURCE_DIR}/mindspore/parallel
        ${CMAKE_SOURCE_DIR}/mindspore/mindrecord
        ${CMAKE_SOURCE_DIR}/mindspore/numpy
        ${CMAKE_SOURCE_DIR}/mindspore/train
        ${CMAKE_SOURCE_DIR}/mindspore/boost
        ${CMAKE_SOURCE_DIR}/mindspore/common
        ${CMAKE_SOURCE_DIR}/mindspore/ops
        ${CMAKE_SOURCE_DIR}/mindspore/communication
        ${CMAKE_SOURCE_DIR}/mindspore/profiler
        ${CMAKE_SOURCE_DIR}/mindspore/explainer
        ${CMAKE_SOURCE_DIR}/mindspore/compression
        ${CMAKE_SOURCE_DIR}/mindspore/run_check
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

if((ENABLE_D OR ENABLE_GPU) AND ENABLE_AKG)
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
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/dataset)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/dataset
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore
    )
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("offline debugger does not support windows system temporarily")
else()
    if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/offline_debug)
        install(
            DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/offline_debug
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

## config files
install(
    FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
    DESTINATION ${INSTALL_CFG_DIR}
    COMPONENT mindspore
)

