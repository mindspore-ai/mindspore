set(protobuf_USE_STATIC_LIBS ON)
set(ENABLE_NATIVE_PROTOBUF "off")
if(EXISTS ${TOP_DIR}/mindspore/lite/providers/protobuf/native_protobuf.cfg)
    set(ENABLE_NATIVE_PROTOBUF "on")
    file(STRINGS ${TOP_DIR}/mindspore/lite/providers/protobuffer/native_protobuffer.cfg native_protobuffer_path)
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
        -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    if(MSVC)
        set(protobuf_CXXFLAGS "/DWIN32 /D_WINDOWS /W3 /GR /EHsc")
        set(protobuf_CFLAGS "${CMAKE_C_FLAGS}")
        set(protobuf_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
        if(DEBUG_MODE)
            set(protobuf_Debug ON)
        endif()
    else()
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    endif()
else()
    set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
        -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    if(NOT ENABLE_GLIBCXX)
        set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()
set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v34.1.tar.gz")
    set(SHA256 "25a88b23229a878989336faeedaab4cf75dd124ad2de608bc925892ab58797cf")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v34.1.tar.gz")
    set(SHA256 "a83103b7ed3afaeedee9a212c8f65825444f58144f5e075b73c83f2b4ff27b62")
endif()

set(PROTOBUF_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/protobuf)
set(PROTOBUF_PATCHES)
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(PROTOBUF_PATCHES ${PROTOBUF_PATCH_ROOT}/protobuf_v34_gcc7_constexpr.patch)
endif()
set(PROTOBUF_CMAKE_POLICY_OPTION)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    set(PROTOBUF_CMAKE_POLICY_OPTION -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
endif()

if(EXISTS ${absl_ROOT}/lib64)
    set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib64/cmake/absl")
else()
    set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib/cmake/absl")
endif()
message("protobuf using absl_DIR : " ${_FINDPACKAGE_ABSL_CONFIG_DIR})

if(MSVC)
mindspore_add_pkg(protobuf
        VER 34.1
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${PROTOBUF_PATCHES}
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
            ${PROTOBUF_CMAKE_POLICY_OPTION}
            -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
            -Dprotobuf_ABSL_PROVIDER:STRING=package
            -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR})
else()
mindspore_add_pkg(protobuf
        VER 34.1
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${PROTOBUF_PATCHES}
        CMAKE_OPTION -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
            ${PROTOBUF_CMAKE_POLICY_OPTION}
            -Dprotobuf_ABSL_PROVIDER:STRING=package
            -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR})
endif()
include_directories(${protobuf_INC})
include_directories(${CMAKE_BINARY_DIR}/proto_py)

# protobuf 34.1 statically depends on absl and utf8_range; add them as transitive link deps
if(MSVC)
    # Windows absl libs are named absl_*.lib (no leading "lib")
    file(GLOB _PROTOBUF_ABSL_LIBS "${absl_ROOT}/lib/absl_*.lib")
    set(_PROTOBUF_UTF8_EXT ".lib")
else()
    file(GLOB _PROTOBUF_ABSL_LIBS "${absl_ROOT}/lib/libabsl_*.a")
    set(_PROTOBUF_UTF8_EXT ".a")
endif()
if(EXISTS "${protobuf_DIRPATH}/lib64/libutf8_range${_PROTOBUF_UTF8_EXT}")
    set(_PROTOBUF_UTF8_DIR "${protobuf_DIRPATH}/lib64")
else()
    set(_PROTOBUF_UTF8_DIR "${protobuf_DIRPATH}/lib")
endif()
set(_PROTOBUF_UTF8_LIBS
    "${_PROTOBUF_UTF8_DIR}/libutf8_range${_PROTOBUF_UTF8_EXT}"
    "${_PROTOBUF_UTF8_DIR}/libutf8_validity${_PROTOBUF_UTF8_EXT}")
if(NOT MSVC AND NOT APPLE)
    set_target_properties(protobuf::protobuf PROPERTIES
        INTERFACE_LINK_LIBRARIES "-Wl,--start-group;${_PROTOBUF_ABSL_LIBS};${_PROTOBUF_UTF8_LIBS};-Wl,--end-group")
else()
    set_target_properties(protobuf::protobuf PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_PROTOBUF_ABSL_LIBS};${_PROTOBUF_UTF8_LIBS}")
endif()

add_library(mindspore::protobuf ALIAS protobuf::protobuf)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
# recover original value
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()

if(ENABLE_NATIVE_PROTOBUF)
    set(PROTOC ${native_protobuffer_path}/bin/protoc)
    set(PROTOBUF_LIB ${native_protobuffer_path}/lib/libprotobuf.so.34.1.0)
    set(protobuf_LIBPATH ${native_protobuffer_path}/lib)
    set(protobuf_INC ${native_protobuffer_path}/include)

    include_directories(${protobuf_INC})
    message("protobuf_INC : ${protobuf_INC}")
    set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
endif()
function(common_protobuf_generate path c_var h_var)
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

        list(APPEND ${c_var} "${path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${path}/${file_name}.pb.h")
        if(ENABLE_NATIVE_PROTOBUF)
            add_custom_command(
            OUTPUT "${path}/${file_name}.pb.cc" "${path}/${file_name}.pb.h"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${path}"
            COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${protobuf_LIBPATH}" ${PROTOC} -I${file_dir}
            --cpp_out=${path} ${abs_file}
            DEPENDS ${PROTOC} ${abs_file}
            COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        else()
        add_custom_command(
                OUTPUT "${path}/${file_name}.pb.cc" "${path}/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${path}"
                COMMAND protobuf::protoc -I${file_dir} --cpp_out=${path} ${abs_file}
                DEPENDS protobuf::protoc ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        endif()
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()

function(ms_protobuf_generate c_var h_var)
        common_protobuf_generate(${CMAKE_BINARY_DIR}/proto ${c_var} ${h_var} ${ARGN})
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

        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.h")
        list(APPEND ${py_var} "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py")
        if(WIN32)
            add_custom_command(
                    OUTPUT "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.cc"
                    "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.h"
                    "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto_py/proto"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto_py/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto_py/proto ${abs_file}
                    COMMAND perl -pi.bak -e "s/import (.+_pb2.*)/from . import \\1/"
                            "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                            "${TOP_DIR}/mindspore/python/mindspore/train/"
                    DEPENDS protobuf::protoc ${abs_file}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        else()
            add_custom_command(
                    OUTPUT "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.cc"
                    "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.h"
                    "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto_py/proto"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto_py/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto_py/proto ${abs_file}
                    COMMAND perl -pi -e "s/import (.+_pb2.*)/from . import \\1/"
                            "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                    COMMAND cp "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                            "${TOP_DIR}/mindspore/python/mindspore/train/"
                    DEPENDS protobuf::protoc ${abs_file}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        endif()
    endforeach()
    set_source_files_properties(${${c_var}} ${${h_var}} ${${py_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
    set(${py_var} ${${py_var}} PARENT_SCOPE)
endfunction()
