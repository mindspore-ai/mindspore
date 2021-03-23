set(protobuf_USE_STATIC_LIBS ON)
if(BUILD_LITE)
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
        -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    else()
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
    endif()
endif()

set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.8.0.tar.gz")
    set(MD5 "eba86ae9f07ba5cfbaf8af3bc4e84236")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.8.0.tar.gz")
    set(MD5 "3d9e32700639618a4d2d342c99d4507a")
endif()

mindspore_add_pkg(protobuf
        VER 3.8.0
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_PATH cmake/
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF)

include_directories(${protobuf_INC})
add_library(mindspore::protobuf ALIAS protobuf::protobuf)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})

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

        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.cc"
                "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto"
                COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
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

        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.h")
        list(APPEND ${py_var} "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py")
        if(WIN32)
            add_custom_command(
                    OUTPUT "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.cc"
                    "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.h"
                    "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
                    COMMAND perl -pi.bak -e "s/import (.+_pb2.*)/from . import \\1/"
                            "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py"
                    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py"
                            "${PROJECT_SOURCE_DIR}/mindspore/train/"
                    DEPENDS protobuf::protoc ${abs_file}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        else()
            add_custom_command(
                    OUTPUT "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.cc"
                    "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.h"
                    "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto ${abs_file}
                    COMMAND perl -pi -e "s/import (.+_pb2.*)/from . import \\1/"
                            "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py"
                    COMMAND cp "${CMAKE_BINARY_DIR}/proto/${file_name}_pb2.py" "${PROJECT_SOURCE_DIR}/mindspore/train/"
                    DEPENDS protobuf::protoc ${abs_file}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        endif()
    endforeach()
    set_source_files_properties(${${c_var}} ${${h_var}} ${${py_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
    set(${py_var} ${${py_var}} PARENT_SCOPE)
endfunction()
