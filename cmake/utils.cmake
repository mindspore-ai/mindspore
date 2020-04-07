include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

function(mindspore_add_submodule_obj des_submodule_objs sub_dir submodule_name_obj)

    add_subdirectory(${sub_dir})

    if(NOT TARGET ${submodule_name_obj})
        message(FATAL_ERROR "Can not find submodule '${submodule_name_obj}'. in ${CMAKE_CURRENT_LIST_FILE}")
    endif()
    if("$<TARGET_OBJECTS:${submodule_name_obj}>" IN_LIST ${des_submodule_objs})
        message(FATAL_ERROR "submodule '${submodule_name_obj}' added more than once. in ${CMAKE_CURRENT_LIST_FILE}")
    endif()

    set(${des_submodule_objs} ${${des_submodule_objs}} $<TARGET_OBJECTS:${submodule_name_obj}> PARENT_SCOPE)

endfunction()

if (DEFINED ENV{MSLIBS_CACHE_PATH})
    set(_MS_LIB_CACHE  $ENV{MSLIBS_CACHE_PATH})
else()
    set(_MS_LIB_CACHE ${CMAKE_BINARY_DIR}/.mslib)
endif ()
message("MS LIBS CACHE PATH:  ${_MS_LIB_CACHE}")

if (NOT EXISTS ${_MS_LIB_CACHE})
    file(MAKE_DIRECTORY ${_MS_LIB_CACHE})
endif ()

if (DEFINED ENV{MSLIBS_SERVER})
    set(LOCAL_LIBS_SERVER  $ENV{MSLIBS_SERVER})
    message("LOCAL_LIBS_SERVER:  ${LOCAL_LIBS_SERVER}")
endif ()

include(ProcessorCount)
ProcessorCount(N)
if (JOBS)
    set(THNUM ${JOBS})
else()
    set(JOBS 8)
    if (${JOBS} GREATER ${N})
        set(THNUM ${N})
    else()
        set(THNUM ${JOBS})
    endif()
endif ()
message("set make thread num: ${THNUM}")

if(LOCAL_LIBS_SERVER)
    if (NOT ENV{no_proxy})
        set(ENV{no_proxy} "${LOCAL_LIBS_SERVER}")
    else()
        string(FIND $ENV{no_proxy} ${LOCAL_LIBS_SERVER} IP_POS)
        if (${IP_POS} EQUAL -1)
            set(ENV{no_proxy} "$ENV{no_proxy},${LOCAL_LIBS_SERVER}")
        endif ()
    endif ()
endif()

function(__download_pkg pkg_name pkg_url pkg_md5)

    if(LOCAL_LIBS_SERVER)
        get_filename_component(_URL_FILE_NAME ${pkg_url} NAME)
        set(pkg_url "http://${LOCAL_LIBS_SERVER}:8081/libs/${pkg_name}/${_URL_FILE_NAME}" ${pkg_url})
    endif()

    FetchContent_Declare(
            ${pkg_name}
            URL      ${pkg_url}
            URL_HASH MD5=${pkg_md5}
    )
    FetchContent_GetProperties(${pkg_name})
    message("download: ${${pkg_name}_SOURCE_DIR} , ${pkg_name} , ${pkg_url}")
    if(NOT ${pkg_name}_POPULATED)
        FetchContent_Populate(${pkg_name})
        set(${pkg_name}_SOURCE_DIR ${${pkg_name}_SOURCE_DIR} PARENT_SCOPE)
    endif()

endfunction()

function(__download_pkg_with_git pkg_name pkg_url pkg_git_commit pkg_md5)

    if(LOCAL_LIBS_SERVER)
        set(pkg_url "http://${LOCAL_LIBS_SERVER}:8081/libs/${pkg_name}/${pkg_git_commit}")
        FetchContent_Declare(
                ${pkg_name}
                URL      ${pkg_url}
                URL_HASH MD5=${pkg_md5}
    )
    else()
	FetchContent_Declare(
            ${pkg_name}
	    GIT_REPOSITORY      ${pkg_url}
	    GIT_TAG             ${pkg_git_commit})
    endif()
    FetchContent_GetProperties(${pkg_name})
    message("download: ${${pkg_name}_SOURCE_DIR} , ${pkg_name} , ${pkg_url}")
    if(NOT ${pkg_name}_POPULATED)
        FetchContent_Populate(${pkg_name})
        set(${pkg_name}_SOURCE_DIR ${${pkg_name}_SOURCE_DIR} PARENT_SCOPE)
    endif()

endfunction()


function(__find_pkg_then_add_target pkg_name pkg_exe)

    unset(${pkg_name}_LIBS)

    message("_FIND:${${pkg_name}_BASE_DIR}")

    if(pkg_exe)
        find_program(${pkg_exe}_EXE ${pkg_exe} PATHS ${${pkg_name}_BASE_DIR}/bin NO_DEFAULT_PATH)
        if(NOT ${pkg_exe}_EXE)
            return()
        endif()
        add_executable(${pkg_name}::${pkg_exe} IMPORTED GLOBAL)
        set_target_properties(${pkg_name}::${pkg_exe} PROPERTIES
                IMPORTED_LOCATION ${${pkg_exe}_EXE}
                )
        message("found ${${pkg_exe}_EXE}")
    endif()

    foreach(_LIB_NAME ${ARGN})
        set(_LIB_SEARCH_NAME ${_LIB_NAME})
        set(_LIB_TYPE SHARED)
        if (${pkg_name}_USE_STATIC_LIBS)
            set(_LIB_SEARCH_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
            set(_LIB_TYPE STATIC)
        endif ()
        set(${_LIB_NAME}_LIB ${_LIB_NAME}_LIB-NOTFOUND)
        find_library(${_LIB_NAME}_LIB ${_LIB_SEARCH_NAME} PATHS ${${pkg_name}_BASE_DIR}/lib NO_DEFAULT_PATH)
        if(NOT ${_LIB_NAME}_LIB)
            return()
        endif()
        add_library(${pkg_name}::${_LIB_NAME} ${_LIB_TYPE} IMPORTED GLOBAL)
        set_target_properties(${pkg_name}::${_LIB_NAME} PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${${pkg_name}_BASE_DIR}/include"
                IMPORTED_LOCATION ${${_LIB_NAME}_LIB}
                )
        list(APPEND ${pkg_name}_LIBS ${pkg_name}::${_LIB_NAME})
        message("found ${${_LIB_NAME}_LIB}")
        STRING( REGEX REPLACE "(.+)/(.+)" "\\1" LIBPATH ${${_LIB_NAME}_LIB})
        set(${pkg_name}_LIBPATH ${LIBPATH} CACHE STRING INTERNAL)
    endforeach(_LIB_NAME)

    set(${pkg_name}_LIBS ${${pkg_name}_LIBS} PARENT_SCOPE)
endfunction()

function(__exec_cmd)
    set(options )
    set(oneValueArgs WORKING_DIRECTORY)
    set(multiValueArgs COMMAND)

    cmake_parse_arguments(EXEC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    execute_process(COMMAND ${EXEC_COMMAND}
            WORKING_DIRECTORY ${EXEC_WORKING_DIRECTORY}
            RESULT_VARIABLE RESULT)
    if(NOT RESULT EQUAL "0")
        message(FATAL_ERROR "error! when ${EXEC_COMMAND} in ${EXEC_WORKING_DIRECTORY}")
    endif()
endfunction()

function(__check_patches pkg_patches)
    # check patches
    if (PKG_PATCHES)
        file(TOUCH ${_MS_LIB_CACHE}/${pkg_name}_patch.md5)
        file(READ ${_MS_LIB_CACHE}/${pkg_name}_patch.md5 ${pkg_name}_PATCHES_MD5)

        message("patches md5:${${pkg_name}_PATCHES_MD5}")

        set(${pkg_name}_PATCHES_NEW_MD5 )
        foreach(_PATCH ${PKG_PATCHES})
            file(MD5 ${_PATCH} _PF_MD5)
            set(${pkg_name}_PATCHES_NEW_MD5 "${${pkg_name}_PATCHES_NEW_MD5},${_PF_MD5}")
        endforeach(_PATCH)

        if (NOT ${pkg_name}_PATCHES_MD5 STREQUAL ${pkg_name}_PATCHES_NEW_MD5)
            set(${pkg_name}_PATCHES ${PKG_PATCHES})
            file(REMOVE_RECURSE "${_MS_LIB_CACHE}/${pkg_name}-subbuild")
            file(WRITE ${_MS_LIB_CACHE}/${pkg_name}_patch.md5 ${${pkg_name}_PATCHES_NEW_MD5})
            message("patches changed : ${${pkg_name}_PATCHES_NEW_MD5}")
        endif ()
    endif ()
endfunction()

set(MS_FIND_NO_DEFAULT_PATH NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH NO_SYSTEM_ENVIRONMENT_PATH
                            NO_CMAKE_BUILDS_PATH NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PATH
                            NO_CMAKE_SYSTEM_PACKAGE_REGISTRY)
set(MS_FIND_NO_DEFAULT_PATH ${MS_FIND_NO_DEFAULT_PATH} PARENT_SCOPE)
function(mindspore_add_pkg pkg_name )

    set(options )
    set(oneValueArgs URL MD5 GIT_REPOSITORY GIT_TAG VER EXE DIR HEAD_ONLY)
    set(multiValueArgs CMAKE_OPTION LIBS PRE_CONFIGURE_COMMAND CONFIGURE_COMMAND BUILD_OPTION INSTALL_INCS INSTALL_LIBS PATCHES)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    set(__FIND_PKG_NAME ${pkg_name})
    string(TOLOWER ${pkg_name} pkg_name)
    message("pkg name:${__FIND_PKG_NAME},${pkg_name}")

    set(${pkg_name}_PATCHES_HASH )
    foreach(_PATCH ${PKG_PATCHES})
        file(MD5 ${_PATCH} _PF_MD5)
        set(${pkg_name}_PATCHES_HASH "${${pkg_name}_PATCHES_HASH},${_PF_MD5}")
    endforeach(_PATCH)

    # check options
    set(${pkg_name}_CONFIG_TXT
            "${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_C_COMPILER_VERSION}
            ${ARGN} - ${${pkg_name}_USE_STATIC_LIBS}- ${${pkg_name}_PATCHES_HASH}
            ${${pkg_name}_CXXFLAGS}--${${pkg_name}_CFLAGS}--${${pkg_name}_LDFLAGS}")
    string(REPLACE ";" "-" ${pkg_name}_CONFIG_TXT ${${pkg_name}_CONFIG_TXT})
    string(MD5 ${pkg_name}_CONFIG_HASH ${${pkg_name}_CONFIG_TXT})

    message("${pkg_name} config hash: ${${pkg_name}_CONFIG_HASH}")

    set(${pkg_name}_BASE_DIR ${_MS_LIB_CACHE}/${pkg_name}_${${pkg_name}_CONFIG_HASH})
    set(${pkg_name}_DIRPATH ${${pkg_name}_BASE_DIR} CACHE STRING INTERNAL)

    if(EXISTS ${${pkg_name}_BASE_DIR}/options.txt AND PKG_HEAD_ONLY)
        set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/${PKG_HEAD_ONLY} PARENT_SCOPE)
        add_library(${pkg_name} INTERFACE)
        target_include_directories(${pkg_name} INTERFACE ${${pkg_name}_INC})
        return()
    endif ()

    if(NOT PKG_EXE)
        set(PKG_EXE 0)
    endif()

    set(${__FIND_PKG_NAME}_ROOT ${${pkg_name}_BASE_DIR})
    set(${__FIND_PKG_NAME}_ROOT ${${pkg_name}_BASE_DIR} PARENT_SCOPE)

    if (PKG_LIBS)
        __find_pkg_then_add_target(${pkg_name} ${PKG_EXE} ${PKG_LIBS})
        if(${pkg_name}_LIBS)
            set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
            message("Found libs: ${${pkg_name}_LIBS}")
            return()
        endif()
    elseif(NOT PKG_HEAD_ONLY)
        find_package(${__FIND_PKG_NAME} ${PKG_VER} ${MS_FIND_NO_DEFAULT_PATH})
        if (${__FIND_PKG_NAME}_FOUND)
            set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
            message("Found pkg: ${__FIND_PKG_NAME}")
            return()
        endif ()
    endif ()

    if (NOT PKG_DIR)
	if (PKG_GIT_REPOSITORY)
	    __download_pkg_with_git(${pkg_name} ${PKG_GIT_REPOSITORY} ${PKG_GIT_TAG} ${PKG_MD5})
	else()
            __download_pkg(${pkg_name} ${PKG_URL} ${PKG_MD5})
	endif()
    else()
        set(${pkg_name}_SOURCE_DIR ${PKG_DIR})
    endif ()
    file(WRITE ${${pkg_name}_BASE_DIR}/options.txt ${${pkg_name}_CONFIG_TXT})
    message("${pkg_name}_SOURCE_DIR : ${${pkg_name}_SOURCE_DIR}")

    foreach(_PATCH_FILE ${PKG_PATCHES})
        message("patching ${${pkg_name}_SOURCE_DIR} -p1 < ${_PATCH_FILE}")
        execute_process(COMMAND patch -p1 INPUT_FILE ${_PATCH_FILE}
                WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR}
                RESULT_VARIABLE Result)
        if(NOT Result EQUAL "0")
            message(FATAL_ERROR "Failed patch: ${_PATCH_FILE}")
        endif()
    endforeach(_PATCH_FILE)
        
    file(LOCK ${${pkg_name}_BASE_DIR} DIRECTORY GUARD FUNCTION RESULT_VARIABLE ${pkg_name}_LOCK_RET TIMEOUT 600)
    if(NOT ${pkg_name}_LOCK_RET EQUAL "0")
        message(FATAL_ERROR "error! when try lock ${${pkg_name}_BASE_DIR} : ${${pkg_name}_LOCK_RET}")
    endif()

    if(${pkg_name}_SOURCE_DIR)
        if (PKG_HEAD_ONLY)
            file(GLOB ${pkg_name}_SOURCE_SUBDIRS ${${pkg_name}_SOURCE_DIR}/*)
            file(COPY ${${pkg_name}_SOURCE_SUBDIRS} DESTINATION ${${pkg_name}_BASE_DIR})
            set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/${PKG_HEAD_ONLY} PARENT_SCOPE)
            add_library(${pkg_name} INTERFACE)
            target_include_directories(${pkg_name} INTERFACE ${${pkg_name}_INC})

        elseif (PKG_CMAKE_OPTION)
            # in cmake
            file(MAKE_DIRECTORY ${${pkg_name}_SOURCE_DIR}/_build)
            if (${pkg_name}_CFLAGS)
                set(${pkg_name}_CMAKE_CFLAGS "-DCMAKE_C_FLAGS=${${pkg_name}_CFLAGS}")
            endif ()
            if (${pkg_name}_CXXFLAGS)
                set(${pkg_name}_CMAKE_CXXFLAGS "-DCMAKE_CXX_FLAGS=${${pkg_name}_CXXFLAGS}")
            endif ()

            if (${pkg_name}_LDFLAGS)
                if (${pkg_name}_USE_STATIC_LIBS)
                    #set(${pkg_name}_CMAKE_LDFLAGS "-DCMAKE_STATIC_LINKER_FLAGS=${${pkg_name}_LDFLAGS}")
                else()
                    set(${pkg_name}_CMAKE_LDFLAGS "-DCMAKE_SHARED_LINKER_FLAGS=${${pkg_name}_LDFLAGS}")
                endif ()
            endif ()

            __exec_cmd(COMMAND ${CMAKE_COMMAND} ${PKG_CMAKE_OPTION} -G ${CMAKE_GENERATOR}
                    ${${pkg_name}_CMAKE_CFLAGS} ${${pkg_name}_CMAKE_CXXFLAGS} ${${pkg_name}_CMAKE_LDFLAGS}
                    -DCMAKE_INSTALL_PREFIX=${${pkg_name}_BASE_DIR} ..
                    WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR}/_build)

            __exec_cmd(COMMAND ${CMAKE_COMMAND} --build . --target install -- -j${THNUM}
                    WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR}/_build)

        else()
            if (${pkg_name}_CFLAGS)
                set(${pkg_name}_MAKE_CFLAGS "CFLAGS=${${pkg_name}_CFLAGS}")
            endif ()
            if (${pkg_name}_CXXFLAGS)
                set(${pkg_name}_MAKE_CXXFLAGS "CXXFLAGS=${${pkg_name}_CXXFLAGS}")
            endif ()
            if (${pkg_name}_LDFLAGS)
                set(${pkg_name}_MAKE_LDFLAGS "LDFLAGS=${${pkg_name}_LDFLAGS}")
            endif ()
            # in configure && make
            if (PKG_PRE_CONFIGURE_COMMAND)
                __exec_cmd(COMMAND ${PKG_PRE_CONFIGURE_COMMAND}
                        WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
            endif ()

            if (PKG_CONFIGURE_COMMAND)
                __exec_cmd(COMMAND ${PKG_CONFIGURE_COMMAND}
                        ${${pkg_name}_MAKE_CFLAGS} ${${pkg_name}_MAKE_CXXFLAGS} ${${pkg_name}_MAKE_LDFLAGS}
                        --prefix=${${pkg_name}_BASE_DIR}
                        WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
            endif ()
            set(${pkg_name}_BUILD_OPTION ${PKG_BUILD_OPTION})
            if (NOT PKG_CONFIGURE_COMMAND)
                set(${pkg_name}_BUILD_OPTION ${${pkg_name}_BUILD_OPTION}
                        ${${pkg_name}_MAKE_CFLAGS} ${${pkg_name}_MAKE_CXXFLAGS} ${${pkg_name}_MAKE_LDFLAGS})
            endif ()
            # build
            __exec_cmd(COMMAND ${CMAKE_MAKE_PROGRAM} ${${pkg_name}_BUILD_OPTION} -j${THNUM}
                    WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})

            if (PKG_INSTALL_INCS OR PKG_INSTALL_LIBS)
                file(GLOB ${pkg_name}_INSTALL_INCS ${${pkg_name}_SOURCE_DIR}/${PKG_INSTALL_INCS})
                file(GLOB ${pkg_name}_INSTALL_LIBS ${${pkg_name}_SOURCE_DIR}/${PKG_INSTALL_LIBS})
                file(COPY ${${pkg_name}_INSTALL_INCS} DESTINATION ${${pkg_name}_BASE_DIR}/include)
                file(COPY ${${pkg_name}_INSTALL_LIBS} DESTINATION ${${pkg_name}_BASE_DIR}/lib)
            else()
                __exec_cmd(COMMAND ${CMAKE_MAKE_PROGRAM} install WORKING_DIRECTORY ${${pkg_name}_SOURCE_DIR})
            endif ()
        endif ()
    endif()

    if (PKG_LIBS)
        __find_pkg_then_add_target(${pkg_name} ${PKG_EXE} ${PKG_LIBS})
        set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
        if(NOT ${pkg_name}_LIBS)
            message(FATAL_ERROR "Can not find pkg: ${pkg_name}")
        endif()
    else()
        find_package(${__FIND_PKG_NAME} ${PKG_VER} QUIET ${MS_FIND_NO_DEFAULT_PATH})
        if (${__FIND_PKG_NAME}_FOUND)
            set(${pkg_name}_INC ${${pkg_name}_BASE_DIR}/include PARENT_SCOPE)
            message("Found pkg: ${${__FIND_PKG_NAME}_LIBRARIES}")
            return()
        endif ()
    endif ()
endfunction()
