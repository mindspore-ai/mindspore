add_definitions(-DPRIMITIVE_WRITEABLE)

add_library(kernel_executor SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_executor_impl.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op_converter.cc
        ${TOP_DIR}/mindspore/lite/src/common/ops/ops_utils.cc
        ${TOP_DIR}/mindspore/lite/src/common/primitive_t_utils.cc
        ${TOP_DIR}/mindspore/lite/src/common/ops/ops_def.cc)

add_dependencies(kernel_executor fbs_inner_src fbs_src mindspore_core)

target_link_libraries(kernel_executor
        mindspore-lite
        mindspore_core
        mindspore::json
        mindspore::flatbuffers
        mindspore::glog)
