set(ENABLE_NATIVE_FLATBUFFER "off")
if(EXISTS ${TOP_DIR}/mindspore/lite/providers/flatbuffer/native_flatbuffer.cfg)
    set(ENABLE_NATIVE_FLATBUFFER on)
endif()
if(ENABLE_NATIVE_FLATBUFFER)
    file(STRINGS ${TOP_DIR}/mindspore/lite/providers/flatbuffer/native_flatbuffer.cfg native_flatbuffer_path)
    set(FLATC "${native_flatbuffer_path}/bin")
    set(FLAT_BUFFERS "")
    set(flatbuffers_INC "${native_flatbuffer_path}/common")
    if(EXISTS ${native_flatbuffer_path}/template)
        set(FLATBUFFER_TEMPALTE "${native_flatbuffer_path}/template")
    endif()
    include_directories(${flatbuffers_INC})

else()
    if(MSVC)
        set(flatbuffers_CXXFLAGS "/DWIN32 /D_WINDOWS /W3 /GR /EHsc")
        set(flatbuffers_CFLAGS "${CMAKE_C_FLAGS}")
        set(flatbuffers_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        # flatbuffers.lib cimplied by msvc
        set(CMAKE_STATIC_LIBRARY_PREFIX "")
    else()
        set(flatbuffers_CXXFLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-strong")
        set(flatbuffers_CFLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -fstack-protector-strong")
    endif()

    if(WIN32)
        set(flatbuffers_USE_STATIC_LIBS ON)
    endif()
    if(ENABLE_GITEE_EULER)
        set(GIT_REPOSITORY "https://gitee.com/src-openeuler/flatbuffers.git")
        set(GIT_TAG "openEuler-22.03-LTS")
        set(SHA256 "d94ef2fb0c22198c7ffe2a6044e864bd467ca70b8cfdc52720dc94313321777b")
        set(FLATBUFFER_SRC "${CMAKE_BINARY_DIR}/_deps/flatbuffers-src")
        set(FLATBUFFER_DIR "${FLATBUFFER_SRC}/flatbuffers-2.0.0")
        __download_pkg_with_git(flatbuffers ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
        execute_process(COMMAND tar -xf ${FLATBUFFER_SRC}/v2.0.0.tar.gz WORKING_DIRECTORY ${FLATBUFFER_SRC})

        foreach(_SUBMODULE_FILE ${PKG_SUBMODULES})
            STRING(REGEX REPLACE "(.+)_(.+)" "\\1" _SUBMODEPATH ${_SUBMODULE_FILE})
            STRING(REGEX REPLACE "(.+)/(.+)" "\\2" _SUBMODENAME ${_SUBMODEPATH})
            file(GLOB ${pkg_name}_INSTALL_SUBMODULE ${_SUBMODULE_FILE}/*)
            file(COPY ${${pkg_name}_INSTALL_SUBMODULE} DESTINATION ${${pkg_name}_SOURCE_DIR}/3rdparty/${_SUBMODENAME})
        endforeach()
    else()
    if(ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/flatbuffers/repository/archive/v2.0.0.tar.gz")
        set(SHA256 "3d1eabe298ddac718de34d334aefc22486064dcd8e7a367a809d87393d59ac5a")
    else()
        set(REQ_URL "https://github.com/google/flatbuffers/archive/v2.0.0.tar.gz")
        set(SHA256 "9ddb9031798f4f8754d00fca2f1a68ecf9d0f83dfac7239af1311e4fd9a565c4")
    endif()
    endif()

    if(APPLE)
        set(flatbuffers_CXXFLAGS "${flatbuffers_CXXFLAGS} -Wno-deprecated")
    endif()
    if(APPLE)
        mindspore_add_pkg(flatbuffers
                VER 2.0.0
                LIBS flatbuffers
                EXE flatc
                URL ${REQ_URL}
                SHA256 ${SHA256}
                CMAKE_OPTION -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib)
    else()
        if(TARGET_AOS_ARM)
            mindspore_add_pkg(flatbuffers
                    VER 2.0.0
                    LIBS flatbuffers
                    EXE flatc
                    URL ${REQ_URL}
                    SHA256 ${SHA256}
                    CMAKE_OPTION -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
                    -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=Release)
        else()
            mindspore_add_pkg(flatbuffers
                    VER 2.0.0
                    LIBS flatbuffers
                    EXE flatc
                    URL ${REQ_URL}
                    SHA256 ${SHA256}
                    DIR ${FLATBUFFER_DIR}
                    CMAKE_OPTION -DCMAKE_C_COMPILER=${FLATC_GCC_COMPILER} -DCMAKE_CXX_COMPILER=${FLATC_GXX_COMPILER}
                    -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=Release)
        endif()
    endif()

    include_directories(${flatbuffers_INC})
    add_library(mindspore::flatbuffers ALIAS flatbuffers::flatbuffers)
    add_executable(mindspore::flatc ALIAS flatbuffers::flatc)
endif()

# recover original value
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()

function(ms_build_flatbuffers source_schema_files
                              source_schema_dirs
                              custom_target_name
                              generated_output_dir)

    set(total_schema_dirs "")
    set(total_generated_files "")
    set(FLATC mindspore::flatc)
    foreach(schema_dir ${source_schema_dirs})
        set(total_schema_dirs -I ${schema_dir} ${total_schema_dirs})
    endforeach()

    foreach(schema IN LISTS ${source_schema_files})
        get_filename_component(filename ${schema} NAME_WE)
        if(NOT ${generated_output_dir} STREQUAL "")
            set(generated_file ${generated_output_dir}/${filename}_generated.h)
            add_custom_command(
                    OUTPUT ${generated_file}
                    COMMAND ${FLATC} --gen-mutable
                    --reflect-names --gen-object-api -o ${generated_output_dir}
                    ${total_schema_dirs}
                    -c -b --reflect-types ${schema}
                    DEPENDS ${FLATC} ${schema}
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Running C++ flatbuffers compiler on ${schema}" VERBATIM)
            list(APPEND total_generated_files ${generated_file})
        endif()
    endforeach()

    add_custom_target(${custom_target_name} ALL
            DEPENDS ${total_generated_files})

    if(NOT ${generated_output_dir} STREQUAL "")
        include_directories(${generated_output_dir})
        set_property(TARGET ${custom_target_name}
                PROPERTY GENERATED_OUTPUT_DIR
                ${generated_output_dir})
    endif()
endfunction()

if(ENABLE_NATIVE_FLATBUFFER)
    function(ms_build_flatbuffers_lite
            source_schema_files source_schema_dirs custom_target_name generated_output_dir if_inner)
        set(total_schema_dirs "")
        set(total_generated_files "")
        foreach(schema_dir ${source_schema_dirs})
            set(total_schema_dirs -I ${schema_dir} ${total_schema_dirs})
        endforeach()
        file(MAKE_DIRECTORY ${generated_output_dir})
        foreach(schema IN LISTS ${source_schema_files})
            get_filename_component(filename ${schema} NAME_WE)
            if(NOT ${generated_output_dir} STREQUAL "")
                set(generated_file ${generated_output_dir}/${filename}_generated.h)
                if(if_inner MATCHES "inner")
                    add_custom_command(
                            OUTPUT ${generated_file}
                            COMMAND ${FLATC} --template ${FLATBUFFER_TEMPALTE} --cpp
                            -o ${generated_output_dir}
                            ${schema}
                            DEPENDS ${FLATC} ${schema}
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                            COMMENT "Running C++ flatbuffers compiler on ${schema}" VERBATIM)
                else()
                    add_custom_command(
                            OUTPUT ${generated_file}
                            COMMAND ${FLATC} --template ${FLATBUFFER_TEMPALTE} --cpp
                            -o ${generated_output_dir}
                            ${schema}
                            DEPENDS ${FLATC} ${schema}
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                            COMMENT "Running C++ flatbuffers compiler on ${schema}" VERBATIM)
                endif()
                list(APPEND total_generated_files ${generated_file})
            endif()
        endforeach()

        add_custom_target(${custom_target_name} ALL
                DEPENDS ${total_generated_files})

        if(NOT ${generated_output_dir} STREQUAL "")
            include_directories(${generated_output_dir})
            set_property(TARGET ${custom_target_name}
                    PROPERTY GENERATED_OUTPUT_DIR
                    ${generated_output_dir})
        endif()
    endfunction()
else()
    function(ms_build_flatbuffers_lite
      source_schema_files source_schema_dirs custom_target_name generated_output_dir if_inner)

        set(total_schema_dirs "")
        set(total_generated_files "")
        set(FLATC mindspore::flatc)
        foreach(schema_dir ${source_schema_dirs})
            set(total_schema_dirs -I ${schema_dir} ${total_schema_dirs})
        endforeach()

        foreach(schema IN LISTS ${source_schema_files})
            get_filename_component(filename ${schema} NAME_WE)
            if(NOT ${generated_output_dir} STREQUAL "")
                set(generated_file ${generated_output_dir}/${filename}_generated.h)
                if(if_inner MATCHES "inner")
                    add_custom_command(
                            OUTPUT ${generated_file}
                            COMMAND ${FLATC} --gen-mutable
                            --reflect-names --gen-object-api -o ${generated_output_dir}
                            ${total_schema_dirs}
                            -c -b --reflect-types ${schema}
                            DEPENDS ${FLATC} ${schema}
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                            COMMENT "Running C++ flatbuffers compiler on ${schema}" VERBATIM)
                else()
                    add_custom_command(
                            OUTPUT ${generated_file}
                            COMMAND ${FLATC} -o ${generated_output_dir}
                            ${total_schema_dirs}
                            -c -b  ${schema}
                            DEPENDS ${FLATC} ${schema}
                            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                            COMMENT "Running C++ flatbuffers compiler on ${schema}" VERBATIM)
                endif()
                list(APPEND total_generated_files ${generated_file})
            endif()
        endforeach()

        add_custom_target(${custom_target_name} ALL
                DEPENDS ${total_generated_files})

        if(NOT ${generated_output_dir} STREQUAL "")
            include_directories(${generated_output_dir})
            set_property(TARGET ${custom_target_name}
                    PROPERTY GENERATED_OUTPUT_DIR
                    ${generated_output_dir})
        endif()
    endfunction()
endif()