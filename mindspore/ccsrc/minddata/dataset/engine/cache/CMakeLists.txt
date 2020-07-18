include_directories("${CMAKE_BINARY_DIR}/minddata/dataset/engine/cache")
set(MD_FLATBUFFER_OU "${CMAKE_BINARY_DIR}/minddata/dataset/engine/cache")
ms_build_flatbuffers("de_tensor.fbs" ${CMAKE_CURRENT_SOURCE_DIR} generated_engine_files ${MD_FLATBUFFER_OU})

file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

add_library(engine-cache-client OBJECT
    cache_client.cc
    cache_fbb.cc
    cache_request.cc)

if (ENABLE_CACHE)
  ms_grpc_generate(CACHE_GRPC_SRCS CACHE_GRPC_HDRS cache_grpc.proto)
  target_sources(engine-cache-client PUBLIC ${CACHE_GRPC_SRCS} cache_grpc_client.cc)

  add_library(engine-cache-server OBJECT
      ${CACHE_GRPC_SRCS}
      cache_grpc_server.cc
      cache_arena.cc
      cache_service.cc
      cache_server.cc)

  add_executable(cache_server cache_main.cc)
  target_link_libraries(cache_server
      engine-cache-server
      $<TARGET_OBJECTS:utils>
      mindspore
      mindspore::glog
      mindspore::protobuf
      mindspore::grpc++
      mindspore_gvar
      ${PYTHON_LIBRARIES}
      ${SECUREC_LIBRARY}
      pthread)

  add_executable(cache_admin cache_admin.cc cache_admin_arg.cc)
  target_link_libraries(cache_admin _c_dataengine _c_mindrecord ${PYTHON_LIBRARIES} mindspore::glog)

  add_dependencies(engine-cache-server generated_engine_files)

else ()
  ms_protobuf_generate(CACHE_PROTO_SRCS CACHE_PRTO_HDRS cache_grpc.proto)
  target_sources(engine-cache-client PUBLIC ${CACHE_PROTO_SRCS})
endif ()

add_dependencies(engine-cache-client generated_engine_files)
