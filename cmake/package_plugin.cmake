# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# prepare output directory
file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# cpack variables
string(TOLOWER linux_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_NAME)
if(PYTHON_VERSION MATCHES "3.9")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py39)
elseif(PYTHON_VERSION MATCHES "3.8")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py38)
elseif(PYTHON_VERSION MATCHES "3.7")
    set(CPACK_PACKAGE_FILE_NAME mindspore.py37)
else()
    message("Could not find 'Python 3.9' OR 'Python 3.8' or 'Python 3.7'")
    return()
endif()

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BASE_DIR ".")
set(INSTALL_LIB_DIR "lib")
set(INSTALL_PLUGIN_DIR "${INSTALL_LIB_DIR}/plugin")

# set package files
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

if(ENABLE_D)
    install(
            TARGETS mindspore_ascend
            DESTINATION ${INSTALL_PLUGIN_DIR}
            COMPONENT mindspore
    )
    if(ENABLE_MPI)
        install(
                TARGETS ascend_collective
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
    endif()
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

if(ENABLE_AKG AND CMAKE_SYSTEM_NAME MATCHES "Linux")
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

if(ENABLE_SYM_FILE)
    install(CODE "\
      execute_process(COMMAND ${CMAKE_COMMAND} -DMS_PACK_ROOT_DIR=${CPACK_PACKAGE_DIRECTORY} \
        -DMS_INSTALL_DIR=${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/${CMAKE_HOST_SYSTEM_NAME}/${CPACK_GENERATOR} \
        -DMS_PACKAGE_FILE_NAME=${CPACK_PACKAGE_FILE_NAME} -P ${CMAKE_SOURCE_DIR}/cmake/plugin_debuginfo_script.cmake)"
    )
endif()

include(CPack)
