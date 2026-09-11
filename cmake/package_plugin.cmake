# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# prepare output directory
file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# cpack variables
string(TOLOWER linux_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_NAME)
if(PYTHON_VERSION MATCHES "3.11")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py311)
elseif(PYTHON_VERSION MATCHES "3.10")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py310)
elseif(PYTHON_VERSION MATCHES "3.9")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py39)
elseif(PYTHON_VERSION MATCHES "3.8")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py38)
elseif(PYTHON_VERSION MATCHES "3.7")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py37)
elseif(PYTHON_VERSION MATCHES "3.12")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py312)
else()
    message("Could not find Python versions 3.7 - 3.12")
    return()
endif()

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BASE_DIR ".")
set(INSTALL_LIB_DIR "lib")
set(INSTALL_PLUGIN_DIR "${INSTALL_LIB_DIR}/plugin")
set(CUSTOM_ASCENDC_PREBUILD_DIR "${CMAKE_SOURCE_DIR}/mindspore/ops/kernel/ascend/ascendc/prebuild")

if(ENABLE_CPU)
    install(
        TARGETS mindspore_ops_cpu LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
        NAMELINK_SKIP
    )
endif()

if(ENABLE_D)
    install(
            TARGETS mindspore_ascend mindspore_ops_ascend LIBRARY
            DESTINATION ${INSTALL_PLUGIN_DIR}
            COMPONENT mindspore
            NAMELINK_SKIP
    )
    install(
        TARGETS mindspore_ascend_res_manager mindspore_graph_ir LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
        COMPONENT mindspore
        NAMELINK_SKIP
    )
    install(
        TARGETS mindspore_ge_backend LIBRARY
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
        NAMELINK_SKIP
    )
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
    endif()
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
    install(
            TARGETS hccl_plugin
            DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
            COMPONENT mindspore
    )
    install(
            TARGETS _c_dataengine
            DESTINATION ${INSTALL_BASE_DIR}
            COMPONENT mindspore
    )
    install(
            DIRECTORY
            ${CMAKE_SOURCE_DIR}/mindspore/python/mindspore/custom_compiler
            ${CUSTOM_ASCENDC_PREBUILD_DIR}/${CMAKE_SYSTEM_PROCESSOR}/custom_ascendc_ops/custom_ascendc_910b
            DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
            COMPONENT mindspore
    )
    install(
            TARGETS mindspore_extension_ascend_aclnn ARCHIVE
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

if(ENABLE_GPU)
    install(
            TARGETS mindspore_gpu LIBRARY
            DESTINATION ${INSTALL_PLUGIN_DIR}
            COMPONENT mindspore
            NAMELINK_SKIP
    )
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

if(ENABLE_AKG AND CMAKE_SYSTEM_NAME MATCHES "Linux")
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
endif()

if(ENABLE_SYM_FILE)
    install(CODE "\
      execute_process(COMMAND ${CMAKE_COMMAND} -DMS_PACK_ROOT_DIR=${CPACK_PACKAGE_DIRECTORY} \
        -DMS_INSTALL_DIR=${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/${CMAKE_HOST_SYSTEM_NAME}/${CPACK_GENERATOR} \
        -DMS_PACKAGE_FILE_NAME=${CPACK_PACKAGE_FILE_NAME} -P ${CMAKE_SOURCE_DIR}/cmake/plugin_debuginfo_script.cmake)"
    )
endif()

include(CPack)
