message(STATUS "Compiling GraphEngine")
set(GE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/graphengine)

message(STATUS "[ME] build_path: ${BUILD_PATH}")

function(find_submodule_lib module name path)
    find_library(${module}_LIBRARY_DIR NAMES ${name} NAMES_PER_DIR PATHS ${path}
            PATH_SUFFIXES lib
            )
    if("${${module}_LIBRARY_DIR}" STREQUAL "${module}_LIBRARY_DIR-NOTFOUND")
        message(FATAL_ERROR "${name} not found in any of following paths: ${path}")
    endif()
    add_library(${module} SHARED IMPORTED)
    set_target_properties(${module} PROPERTIES
            IMPORTED_LOCATION ${${module}_LIBRARY_DIR}
            )
endfunction()

if(ENABLE_D OR ENABLE_ACL OR ENABLE_TESTCASES)
    set(_ge_tmp_CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    set(_ge_tmp_ENABLE_GITEE ${ENABLE_GITEE})
    set(_ge_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(ENABLE_GITEE ON)
    set(CMAKE_INSTALL_PREFIX ${BUILD_PATH}/graphengine)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FILE__='\"$(subst $(realpath ${CMAKE_SOURCE_DIR})/,,$(abspath $<))\"' \
    -Wno-builtin-macro-redefined")

    if(ENABLE_TESTCASES)
        # use slog, error manager, mmpa in non ascend mode, e.g. tests
        set(GE_PREBUILD_PATH ${GE_SOURCE_DIR}/third_party/prebuild/${CMAKE_HOST_SYSTEM_PROCESSOR})
        set(ENABLE_MS_TESTCASES TRUE)
        find_submodule_lib(slog libalog.so ${GE_PREBUILD_PATH})
        find_submodule_lib(static_mmpa libmmpa.a ${GE_PREBUILD_PATH})
    endif()

    string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_subdirectory(${GE_SOURCE_DIR})
    set(CMAKE_INSTALL_PREFIX ${_ge_tmp_CMAKE_INSTALL_PREFIX})
    set(ENABLE_GITEE ${_ge_tmp_ENABLE_GITEE})
    set(CMAKE_CXX_FLAGS ${_ge_tmp_CMAKE_CXX_FLAGS})
else()
    message(FATAL_ERROR "No compile option defined for GraphEngine, exiting")
endif()
