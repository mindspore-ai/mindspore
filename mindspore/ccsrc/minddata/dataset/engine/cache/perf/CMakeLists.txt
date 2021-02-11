file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

if(ENABLE_CACHE)
  ms_protobuf_generate(CACHE_PERF_PROTO_SRCS CACHE_PERF_PROTO_HDRS cache_perf.proto)

  add_executable(cache_perf cache_perf.cc cache_msg.cc cache_perf_run.cc ${CACHE_PERF_PROTO_SRCS})
  target_link_libraries(cache_perf
      _c_dataengine
      _c_mindrecord
      mindspore
      mindspore::protobuf
      mindspore_gvar
      ${PYTHON_LIBRARIES}
      pthread)

  if(USE_GLOG)
    target_link_libraries(cache_perf mindspore::glog)
  endif()

  add_executable(cache_pipeline cache_pipeline.cc cache_msg.cc cache_pipeline_run.cc ${CACHE_PERF_PROTO_SRCS})
  target_link_libraries(cache_pipeline
      _c_dataengine
      _c_mindrecord
      mindspore
      mindspore::protobuf
      mindspore_gvar
      ${PYTHON_LIBRARIES}
      pthread)

  if(USE_GLOG)
    target_link_libraries(cache_pipeline mindspore::glog)
  endif()
endif()
