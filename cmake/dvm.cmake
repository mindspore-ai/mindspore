set(DVM_DIR ${CMAKE_SOURCE_DIR}/dvm)
message(STATUS "DVM_DIR: ${DVM_DIR}")
include_directories(${DVM_DIR}/include)

if(ENABLE_D)
    set(DVM_BUILD_DIR ${CMAKE_BINARY_DIR}/dvm)
    set(DVM_LIB_NAME "libdvm.a")
    set(DVM_LIB ${DVM_BUILD_DIR}/${DVM_LIB_NAME})

    set(DVM_MAKE_ARGS "")
    if(DEBUG_MODE)
        set(DVM_MAKE_ARGS "${DVM_MAKE_ARGS} dbg=1")
    endif()
    if(ENABLE_ASAN)
        set(DVM_MAKE_ARGS "${DVM_MAKE_ARGS} asan=1")
    endif()

    set(DVM_MAKE_ENV "ASCEND_PATH=${ASCEND_PATH}")
    if("${ASCEND_PATH}" MATCHES "/cann$")
        list(APPEND DVM_MAKE_ENV "CANN_VER_85=1")
    endif()

    message(STATUS "DVM_LIB: ${DVM_LIB}")
    message(STATUS "DVM_BUILD_DIR: ${DVM_BUILD_DIR}")
    message(STATUS "DVM_MAKE_ARGS: ${DVM_MAKE_ARGS}")
    message(STATUS "DVM_MAKE_ENV: ${DVM_MAKE_ENV}")

    file(GLOB_RECURSE DVM_DEPENDS
            ${DVM_DIR}/Makefile
            ${DVM_DIR}/src/*.cce
            ${DVM_DIR}/src/*.cc
            ${DVM_DIR}/src/*.h
            ${DVM_DIR}/include/*.h
    )

    add_custom_command(
            OUTPUT ${DVM_LIB}
            # create dvm build dir
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${DVM_BUILD_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${DVM_BUILD_DIR}
            # copy dvm dir to dvm build dir
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${DVM_DIR} ${DVM_BUILD_DIR}
            COMMAND find ${DVM_BUILD_DIR} -name "*.a" -delete
            COMMAND find ${DVM_BUILD_DIR} -name "*.o" -delete
            COMMAND find ${DVM_BUILD_DIR} -name "*.so" -delete
            # compile
            COMMAND cd ${DVM_BUILD_DIR} && env ${DVM_MAKE_ENV} make ${DVM_MAKE_ARGS} ${DVM_LIB_NAME}
            # compile dependency, re-compile if any dependency changes
            DEPENDS ${DVM_DEPENDS}
            COMMENT "Compiling DVM library"
    )
    add_custom_target(dvm ALL DEPENDS ${DVM_LIB})
    add_compile_definitions(ENABLE_DVM)
endif()