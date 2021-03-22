set(flatbuffers_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(flatbuffers_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(WIN32)
    set(flatbuffers_USE_STATIC_LIBS ON)
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/flatbuffers/repository/archive/v1.11.0.tar.gz")
    set(MD5 "4051dc865063ffa724c4264dea8dbbe9")
else()
    set(REQ_URL "https://github.com/google/flatbuffers/archive/v1.11.0.tar.gz")
    set(MD5 "02c64880acb89dbd57eebacfd67200d8")
endif()

mindspore_add_pkg(flatbuffers
        VER 1.11.0
        LIBS flatbuffers
        EXE flatc
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib)

include_directories(${flatbuffers_INC})
add_library(mindspore::flatbuffers ALIAS flatbuffers::flatbuffers)
add_executable(mindspore::flatc ALIAS flatbuffers::flatc)
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

    foreach(schema ${source_schema_files})
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
