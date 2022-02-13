include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/src)

file(GLOB_RECURSE GRAPH_SCHEDULER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set_property(SOURCE ${GRAPH_SCHEDULER_SRC_LIST}
  PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
add_library(_mindspore_runtime_graph_scheduler_obj OBJECT ${GRAPH_SCHEDULER_SRC_LIST})