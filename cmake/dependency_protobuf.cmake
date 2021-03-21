# Protobuf
#
#
# PROTOBUF_LIBRARY - Link this to use protobuf
#
if(NOT TARGET protobuf::libprotobuf)
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable protobuf test")
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "Gen shared library")
    set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

    string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/protobuf/cmake ${CMAKE_BINARY_DIR}/protobuf)

    set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/protobuf/src)

set(PROTOBUF_LIBRARY protobuf::libprotobuf)
set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)

function(ms_protobuf_generate c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_protobuf_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})

    foreach(file ${ARGN})
        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${file_dir})


        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.cc"
                "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/${rel_path}"
                COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/${rel_path} ${abs_file}
                DEPENDS protobuf::protoc ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)

endfunction()

function(ms_protobuf_generate_py c_var h_var py_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_protobuf_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})
    set(${py_var})

    foreach(file ${ARGN})
        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${file_dir})


        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.h")
        list(APPEND ${py_var} "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}_pb2.py")

        add_custom_command(
                OUTPUT "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.cc"
                "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}.pb.h"
                "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}_pb2.py"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/${rel_path}"
                COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/${rel_path} ${abs_file}
                COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/${rel_path} ${abs_file}
                COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/${rel_path} ${abs_file}
                COMMAND perl -pi -e "s/import (.+_pb2.*)/from . import \\1/"
                        "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}_pb2.py"
                COMMAND cp "${CMAKE_BINARY_DIR}/${rel_path}/${file_name}_pb2.py"
                        "${PROJECT_SOURCE_DIR}/mindspore/train/"
                DEPENDS protobuf::protoc ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} ${${py_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
    set(${py_var} ${${py_var}} PARENT_SCOPE)

endfunction()
