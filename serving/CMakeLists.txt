find_package(Threads REQUIRED)

# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.

#set(protobuf_MODULE_COMPATIBLE TRUE)
#find_package(Protobuf CONFIG REQUIRED)
#message(STATUS "Using protobuf ${protobuf_VERSION}")
add_library(protobuf::libprotobuf ALIAS protobuf::protobuf)
add_executable(protobuf::libprotoc ALIAS protobuf::protoc)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
if(CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
else()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(hw_proto "ms_service.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/ms_service.grpc.pb.h")
add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/core"
        "${PROJECT_SOURCE_DIR}/mindspore/ccsrc")
file(GLOB_RECURSE CORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "core/*.cc" "core/util/*.cc" "core/version_control/*.cc")

list(APPEND SERVING_SRC "main.cc" ${hw_proto_srcs} ${hw_grpc_srcs} ${CORE_SRC_LIST})

include_directories(${CMAKE_BINARY_DIR})
add_executable(ms_serving ${SERVING_SRC})
target_link_libraries(ms_serving inference mindspore_gvar)
target_link_libraries(ms_serving ${_REFLECTION} ${_GRPC_GRPCPP} ${_PROTOBUF_LIBPROTOBUF} pthread)
if (ENABLE_D)
    add_compile_definitions(ENABLE_D)
    target_link_libraries(ms_serving ${RUNTIME_LIB})
endif()
