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
set(CPACK_OSX_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})


if(ENABLE_CPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "cpu")
    if(BUILD_DEV_MODE)
        set(CPACK_MS_PACKAGE_NAME "mindspore-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "mindspore")
    endif()
else()
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend or gpu or cpu")
    if(BUILD_DEV_MODE)
        set(CPACK_MS_PACKAGE_NAME "mindspore-dev")
    else()
        set(CPACK_MS_PACKAGE_NAME "mindspore")
    endif()
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
    TARGETS mindspore_core mindspore_common mindspore_backend
    DESTINATION ${INSTALL_LIB_DIR}
    COMPONENT mindspore
)

if(USE_GLOG)
    install(FILES ${glog_LIBPATH}/libmindspore_glog.0.4.0.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_glog.0.dylib COMPONENT mindspore)
endif()

if(ENABLE_CPU AND NOT WIN32)
    install(
        TARGETS ps_cache
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
endif()

if(MS_BUILD_GRPC)
    install(FILES ${grpc_LIBPATH}/libmindspore_grpc++.1.36.1.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_grpc++.1.dylib COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_grpc.15.0.0.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_grpc.15.dylib COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_gpr.15.0.0.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_gpr.15.dylib COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_upb.15.0.0.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_upb.15.dylib COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_address_sorting.15.0.0.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_address_sorting.15.dylib COMPONENT mindspore)
endif()

if(ENABLE_MINDDATA)
    install(
        TARGETS _c_dataengine _c_mindrecord
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore
    )

    install(FILES ${opencv_LIBPATH}/libopencv_core.4.5.2.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_core.4.5.dylib COMPONENT mindspore)
    install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.4.5.2.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgcodecs.4.5.dylib COMPONENT mindspore)
    install(FILES ${opencv_LIBPATH}/libopencv_imgproc.4.5.2.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libopencv_imgproc.4.5.dylib COMPONENT mindspore)
    install(FILES ${tinyxml2_LIBPATH}/libtinyxml2.8.0.0.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libtinyxml2.8.dylib COMPONENT mindspore)

    install(FILES ${icu4c_LIBPATH}/libicuuc.69.1.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libicuuc.69.dylib COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicudata.69.1.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libicudata.69.dylib COMPONENT mindspore)
    install(FILES ${icu4c_LIBPATH}/libicui18n.69.1.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libicui18n.69.dylib COMPONENT mindspore)
endif()

if(ENABLE_CPU)
    install(FILES ${onednn_LIBPATH}/libdnnl.2.2.dylib
        DESTINATION ${INSTALL_LIB_DIR} RENAME libdnnl.2.dylib COMPONENT mindspore)
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

# set python files
file(GLOB MS_PY_LIST ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/*.py)
install(
    FILES ${MS_PY_LIST}
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
    ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/compression
    ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/rewrite
    ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/run_check
    ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/experimental
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/dataset)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/dataset
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore
    )
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/offline_debug)
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/offline_debug
        DESTINATION ${INSTALL_PY_DIR}
        COMPONENT mindspore
    )
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
