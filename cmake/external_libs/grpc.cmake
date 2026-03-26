set(grpc_USE_STATIC_LIBS OFF)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2 \
        -Dgrpc=mindspore_grpc -Dgrpc_impl=mindspore_grpc_impl -Dgrpc_core=mindspore_grpc_core")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
else()
    set(grpc_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2 \
        -Dgrpc=mindspore_grpc -Dgrpc_impl=mindspore_grpc_impl -Dgrpc_core=mindspore_grpc_core")
    set(grpc_CFLAGS "-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
    if(NOT ENABLE_GLIBCXX)
        set(grpc_CXXFLAGS "${grpc_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(grpc_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

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

if(EXISTS ${c-ares_ROOT}/lib64)
    set(_FINDPACKAGE_CARES_CONFIG_DIR "${c-ares_ROOT}/lib64/cmake/c-ares")
else()
    set(_FINDPACKAGE_CARES_CONFIG_DIR "${c-ares_ROOT}/lib/cmake/c-ares")
endif()
message("grpc using c-ares_DIR : " ${_FINDPACKAGE_CARES_CONFIG_DIR})

if(EXISTS ${re2_ROOT}/lib64)
    set(_FINDPACKAGE_RE2_CONFIG_DIR "${re2_ROOT}/lib64/cmake/re2")
else()
    set(_FINDPACKAGE_RE2_CONFIG_DIR "${re2_ROOT}/lib/cmake/re2")
endif()
message("grpc using re2_DIR : " ${_FINDPACKAGE_RE2_CONFIG_DIR})

if(EXISTS ${protobuf_ROOT}/lib64)
    set(_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR "${protobuf_ROOT}/lib64/cmake/utf8_range")
else()
    set(_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR "${protobuf_ROOT}/lib/cmake/utf8_range")
endif()
message("grpc using utf8_range_DIR : " ${_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR})

if(EXISTS ${openssl_ROOT})
    set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${openssl_ROOT}")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/grpc/repository/archive/v1.73.0.tar.gz")
    set(SHA256 "fe91d435bc7e0d9430abd88897306dc03b345346db64b57585e6084f047ed6d9")
else()
    set(REQ_URL "https://github.com/grpc/grpc/archive/v1.73.0.tar.gz")
    set(SHA256 "00865dbc3d7689fd9c5c7d0a10822a4eae175ca692870433b64106abcb1294a0")
endif()

mindspore_add_pkg(grpc
        VER 1.73.0
        LIBS mindspore_grpc++ mindspore_grpc mindspore_gpr mindspore_upb_base_lib mindspore_address_sorting
        EXE grpc_cpp_plugin
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/grpc/grpc.patch001
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_CXX_STANDARD=17
        -DBUILD_SHARED_LIBS=ON
        -DgRPC_INSTALL:BOOL=ON
        -DgRPC_BUILD_TESTS:BOOL=OFF
        -DgRPC_DOWNLOAD_ARCHIVES=OFF
        -DgRPC_BUILD_GRPC_CSHARP_PLUGIN:BOOL=OFF
        -DgRPC_BUILD_GRPC_NODE_PLUGIN:BOOL=OFF
        -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN:BOOL=OFF
        -DgRPC_BUILD_GRPC_PHP_PLUGIN:BOOL=OFF
        -DgRPC_BUILD_GRPC_PYTHON_PLUGIN:BOOL=OFF
        -DgRPC_BUILD_GRPC_RUBY_PLUGIN:BOOL=OFF
        -DgRPC_PROTOBUF_PROVIDER:STRING=package
        -DgRPC_PROTOBUF_PACKAGE_TYPE:STRING=CONFIG
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_ZLIB_PROVIDER:STRING=package
        -DZLIB_ROOT:PATH=${zlib_ROOT}
        -DgRPC_ABSL_PROVIDER:STRING=package
        -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR}
        -DgRPC_CARES_PROVIDER:STRING=package
        -Dc-ares_DIR:PATH=${_FINDPACKAGE_CARES_CONFIG_DIR}
        -DgRPC_SSL_PROVIDER:STRING=package
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        -DgRPC_RE2_PROVIDER:STRING=package
        -Dre2_DIR:PATH=${_FINDPACKAGE_RE2_CONFIG_DIR}
        -Dutf8_range_DIR:PATH=${_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR}
        )

include_directories(${grpc_INC})

add_library(mindspore::grpc++ ALIAS grpc::mindspore_grpc++)

# link other grpc libs
target_link_libraries(grpc::mindspore_grpc++ INTERFACE grpc::mindspore_grpc grpc::mindspore_gpr
    grpc::mindspore_upb_base_lib grpc::mindspore_address_sorting)

# modify mindspore macro define
add_compile_definitions(grpc=mindspore_grpc)
add_compile_definitions(grpc_impl=mindspore_grpc_impl)
add_compile_definitions(grpc_core=mindspore_grpc_core)

function(ms_grpc_generate c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_grpc_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})
    set(_GRPC_PROTOC_ENV)
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(_GRPC_PROTOC_ENV ${CMAKE_COMMAND} -E env "DYLD_LIBRARY_PATH=${zlib_ROOT}/lib:$ENV{DYLD_LIBRARY_PATH}")
    endif()

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
                COMMAND ${_GRPC_PROTOC_ENV} $<TARGET_FILE:protobuf::protoc>
                        -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto
                        --grpc_out=${CMAKE_BINARY_DIR}/proto
                        --plugin=protoc-gen-grpc=$<TARGET_FILE:grpc::grpc_cpp_plugin> ${abs_file}
                DEPENDS protobuf::protoc grpc::grpc_cpp_plugin ${abs_file}
                COMMENT "Running C++ gRPC compiler on ${file}" VERBATIM)
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()
