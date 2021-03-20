set(grpc_USE_STATIC_LIBS ON)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
        -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
        -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
else()
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
        -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    if(NOT ENABLE_GLIBCXX)
        set(grpc_CXXFLAGS "${grpc_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

set(grpc_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")


if(EXISTS ${protobuf_ROOT}/lib64)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib64/cmake/protobuf")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${protobuf_ROOT}/lib/cmake/protobuf")
endif()
message("grpc using Protobuf_DIR : " ${_FINDPACKAGE_PROTOBUF_CONFIG_DIR})

if(EXISTS ${absl_ROOT}/lib64)
  set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib64/cmake/absl")
else()
  set(_FINDPACKAGE_ABSL_CONFIG_DIR "${absl_ROOT}/lib/cmake/absl")
endif()
message("grpc using absl_DIR : " ${_FINDPACKAGE_ABSL_CONFIG_DIR})

set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if(OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/grpc/repository/archive/v1.27.3.tar.gz")
    set(MD5 "b8b6d8defeda0355105e3b64b4201786")
else()
    set(REQ_URL "https://github.com/grpc/grpc/archive/v1.27.3.tar.gz")
    set(MD5 "0c6c3fc8682d4262dd0e5e6fabe1a7e2")
endif()

mindspore_add_pkg(grpc
        VER 1.27.3
        LIBS grpc++ grpc gpr upb address_sorting
        EXE grpc_cpp_plugin
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:PATH=${zlib_ROOT}
        -DgRPC_ABSL_PROVIDER:STRING=package
        -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR}
        -DgRPC_CARES_PROVIDER:STRING=package
        -Dc-ares_DIR:PATH=${c-ares_ROOT}/lib/cmake/c-ares
        -DgRPC_SSL_PROVIDER:STRING=package
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        )

include_directories(${grpc_INC})

add_library(mindspore::grpc++ ALIAS grpc::grpc++)

# link other grpc libs
target_link_libraries(grpc::grpc++ INTERFACE grpc::grpc grpc::gpr grpc::upb grpc::address_sorting)

# link built dependencies
target_link_libraries(grpc::grpc++ INTERFACE mindspore::z)
target_link_libraries(grpc::grpc++ INTERFACE mindspore::cares)
target_link_libraries(grpc::grpc++ INTERFACE mindspore::absl_strings mindspore::absl_throw_delegate
                      mindspore::absl_raw_logging_internal mindspore::absl_int128 mindspore::absl_bad_optional_access)

# link system openssl
find_package(OpenSSL REQUIRED)
target_link_libraries(grpc::grpc++ INTERFACE OpenSSL::SSL OpenSSL::Crypto)


function(ms_grpc_generate c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_grpc_generate() called without any proto files")
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
        list(APPEND ${c_var} "${CMAKE_BINARY_DIR}/proto/${file_name}.grpc.pb.cc")
        list(APPEND ${h_var} "${CMAKE_BINARY_DIR}/proto/${file_name}.grpc.pb.h")

        add_custom_command(
                OUTPUT "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.cc"
                "${CMAKE_BINARY_DIR}/proto/${file_name}.pb.h"
                "${CMAKE_BINARY_DIR}/proto/${file_name}.grpc.pb.cc"
                "${CMAKE_BINARY_DIR}/proto/${file_name}.grpc.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto"
                COMMAND protobuf::protoc --version
                COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto
                        --grpc_out=${CMAKE_BINARY_DIR}/proto
                        --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc::grpc_cpp_plugin> ${abs_file}
                DEPENDS protobuf::protoc grpc::grpc_cpp_plugin ${abs_file}
                COMMENT "Running C++ gRPC compiler on ${file}" VERBATIM)
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()
