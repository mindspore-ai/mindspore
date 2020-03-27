set(flatbuffers_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(flatbuffers_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(flatbuffers
        VER 1.11.0
        LIBS flatbuffers
        EXE flatc
        URL https://github.com/google/flatbuffers/archive/v1.11.0.tar.gz
        MD5 02c64880acb89dbd57eebacfd67200d8
        CMAKE_OPTION -DFLATBUFFERS_BUILD_TESTS=OFF )

include_directories(${flatbuffers_INC})
add_library(mindspore::flatbuffers ALIAS flatbuffers::flatbuffers)
add_executable(mindspore::flatc ALIAS flatbuffers::flatc)
include_directories(${flatbuffers_INC})
function(ms_build_flatbuffers source_schema_files
                              source_schema_dirs
                              custom_target_name
                              generated_output_dir)

    set(total_schema_dirs "")
    set(total_generated_files "")
    set(FLATC mindspore::flatc)
    foreach (schema_dir ${source_schema_dirs})
        set(total_schema_dirs -I ${schema_dir} ${total_schema_dirs})
    endforeach()

    foreach(schema ${source_schema_files})
        get_filename_component(filename ${schema} NAME_WE)
        if (NOT ${generated_output_dir} STREQUAL "")
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

    if (NOT ${generated_output_dir} STREQUAL "")
        include_directories(${generated_output_dir})
        set_property(TARGET ${custom_target_name}
                PROPERTY GENERATED_OUTPUT_DIR
                ${generated_output_dir})
    endif()
endfunction()
