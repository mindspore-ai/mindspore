# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# set package information
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_GENERATOR "External")
set(CPACK_CMAKE_GENERATOR "Ninja")
set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${CMAKE_SOURCE_DIR}/cmake/package_script.cmake)
set(CPACK_EXTERNAL_ENABLE_STAGING true)
set(CPACK_TEMPORARY_PACKAGE_FILE_NAME ${CMAKE_SOURCE_DIR}/build/package/mindspore)
set(CPACK_TEMPORARY_INSTALL_DIRECTORY ${CMAKE_SOURCE_DIR}/build/package/mindspore)
if(ENABLE_GE)
    set(CPACK_MS_BACKEND "ge")
    set(CPACK_MS_TARGET "ascend-cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore")
elseif(ENABLE_GPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "gpu-cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore-gpu")
elseif(ENABLE_D)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "ascend-cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore-ascend")
elseif(ENABLE_CPU)
    set(CPACK_MS_BACKEND "ms")
    set(CPACK_MS_TARGET "cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore")
elseif(ENABLE_ACL)
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend-gpu-cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore-ascend")
else()
    set(CPACK_MS_BACKEND "debug")
    set(CPACK_MS_TARGET "ascend-gpu-cpu")
    set(CPACK_MS_PACKAGE_NAME "mindspore")
endif()
include(CPack)

# set install path
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_PY_DIR ".")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "bin")
set(INSTALL_CFG_DIR "config")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(INSTALL_LIB_DIR ".")
    set(onednn_LIBPATH ${onednn_LIBPATH}/../bin/)
    set(glog_LIBPATH ${glog_LIBPATH}/../bin/)
    set(opencv_LIBPATH ${opencv_LIBPATH}/../bin/)
    set(jpeg_turbo_LIBPATH ${jpeg_turbo_LIBPATH}/../bin/)
    set(sqlite_LIBPATH ${sqlite_LIBPATH}/../bin/)
    set(tinyxml2_LIBPATH ${tinyxml2_LIBPATH}/../bin/)
    set(sentencepiece_LIBPATH ${sentencepiece_LIBPATH}/../bin/)
else()
    set(INSTALL_LIB_DIR "lib")
endif()

# set package files
install(
    TARGETS _c_expression
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindspore
)

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
    file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libmindspore_glog*)
    install(
        FILES ${GLOG_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
endif()

file(GLOB_RECURSE LIBEVENT_LIB_LIST
        ${libevent_LIBPATH}/libevent*${CMAKE_SHARED_LIBRARY_SUFFIX}*
        ${libevent_LIBPATH}/libevent_pthreads*${CMAKE_SHARED_LIBRARY_SUFFIX}*
        )

install(
        FILES ${LIBEVENT_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
)

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
    file(GLOB_RECURSE OPENCV_LIB_LIST
            ${opencv_LIBPATH}/libopencv_core*
            ${opencv_LIBPATH}/libopencv_imgcodecs*
            ${opencv_LIBPATH}/libopencv_imgproc*
    )
    install(
        FILES ${OPENCV_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    file(GLOB_RECURSE TINYXML2_LIB_LIST ${tinyxml2_LIBPATH}/libtinyxml2*)
    install(
        FILES ${TINYXML2_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    file(GLOB_RECURSE SENTENCEPIECE_LIB_LIST
        ${sentencepiece_LIBPATH}/libsentencepiece*
    )
    install(
    FILES ${SENTENCEPIECE_LIB_LIST}
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        message("icu4c does not support windows system temporarily")
    else()
        file(GLOB_RECURSE ICU4C_LIB_LIST
            ${icu4c_LIBPATH}/libicuuc*
            ${icu4c_LIBPATH}/libicudata*
            ${icu4c_LIBPATH}/libicui18n*
        )
        install(
            FILES ${ICU4C_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
        )
    endif()
endif()

if(ENABLE_CPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}*)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl*${CMAKE_SHARED_LIBRARY_SUFFIX}*)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/dnnl.dll)
    endif()
    install(
        FILES ${DNNL_LIB_LIST}
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

if(ENABLE_CPU AND (ENABLE_D OR ENABLE_GPU))
    install(
        TARGETS ps_cache
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
endif()

if(ENABLE_TESTCASES)
    file(GLOB_RECURSE LIBEVENT_LIB_LIST
            ${libevent_LIBPATH}/libevent*
            ${libevent_LIBPATH}/libevent_pthreads*
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

        install(
            FILES ${CMAKE_SOURCE_DIR}/build/graphengine/c_sec/lib/libc_sec.so
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
        )

        if(ENABLE_D)
            install(
                TARGETS ms_profile
                DESTINATION ${INSTALL_LIB_DIR}
                COMPONENT mindspore
            )
            install(
                FILES
                    ${CMAKE_BINARY_DIR}/graphengine/metadef/graph/libgraph.so
                    ${CMAKE_BINARY_DIR}/graphengine/ge/common/libge_common.so
                    ${CMAKE_BINARY_DIR}/graphengine/ge/ge_runtime/libge_runtime.so
                DESTINATION ${INSTALL_LIB_DIR}
                COMPONENT mindspore
            )
        endif()
    elseif(ENABLE_TESTCASES)
        install(
            FILES
                ${CMAKE_BINARY_DIR}/graphengine/metadef/graph/libgraph.so
                ${CMAKE_SOURCE_DIR}/build/graphengine/c_sec/lib/libc_sec.so
                ${LIBEVENT_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
        )
    endif()
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
        ${CMAKE_SOURCE_DIR}/mindspore/common
        ${CMAKE_SOURCE_DIR}/mindspore/ops
        ${CMAKE_SOURCE_DIR}/mindspore/communication
        ${CMAKE_SOURCE_DIR}/mindspore/profiler
        ${CMAKE_SOURCE_DIR}/mindspore/explainer
        ${CMAKE_SOURCE_DIR}/mindspore/compression
    DESTINATION ${INSTALL_PY_DIR}
    COMPONENT mindspore
)

if((ENABLE_D OR ENABLE_GPU) AND ENABLE_AKG)
    set (AKG_PATH ${CMAKE_SOURCE_DIR}/build/mindspore/akg)
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

## Public header files
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/include
    DESTINATION ${INSTALL_BASE_DIR}
    COMPONENT mindspore
)

## Public header files for minddata
install(
    FILES ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/constants.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/transforms.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/vision.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/vision_lite.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/vision_ascend.h
          ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/execute.h
    DESTINATION ${INSTALL_BASE_DIR}/include/minddata/dataset/include
    COMPONENT mindspore
)

## config files
install(
    FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
    DESTINATION ${INSTALL_CFG_DIR}
    COMPONENT mindspore
)

