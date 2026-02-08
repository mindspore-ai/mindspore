include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)

set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPS_DIR ${OPS_DIR} PARENT_SCOPE)

include(cmake/op_def.cmake)

# mindspore_ops
if(ONLY_BUILD_DEVICE_PLUGINS)
    add_library(mindspore_ops INTERFACE) # dummy target to link
else()
    include(cmake/ops.cmake)
endif()

if(NOT BUILD_LITE)
    # mindspore_ops_fallback
    include(cmake/ops_fallback.cmake)

    add_subdirectory(kernel)

    include(cmake/ops_cpu.cmake)
endif()
