include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/tools/)
include_directories(${CMAKE_BINARY_DIR})
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(_DATADUMP_SRC_LIST)

if(ENABLE_DEBUGGER)
    list(APPEND _DATADUMP_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/debugger_proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/tensor_summary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debug_services.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/debugger_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/tensor_stat_dump.cc"
        )
else()
    list(APPEND _DATADUMP_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/proto_exporter_stub.cc"
        )
endif()

list(APPEND _DATADUMP_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/cpu_e2e_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_json_parser.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/npy_header.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/data_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/common/csv_writer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_control.cc"
    )

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND _DATADUMP_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/e2e_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/tensor_statistic.cc"
        )
    file(GLOB_RECURSE _DEVICE_STATISTIC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/device_statistic/*.cc")
endif()
list(APPEND _DATADUMP_SRC_LIST ${_DEVICE_STATISTIC_LIST})

file(GLOB_RECURSE _TOOLS_SRC_INDEPENDENT_LIST "${CMAKE_CURRENT_SOURCE_DIR}/summary/*.cc")
list(APPEND _TOOLS_SRC_INDEPENDENT_LIST "${CMAKE_CURRENT_SOURCE_DIR}/callback.cc"
                                        "${CMAKE_CURRENT_SOURCE_DIR}/debug_func.cc"
                                        )

list(APPEND _TOOLS_SRC_INDEPENDENT_LIST ${_DATADUMP_SRC_LIST})

file(GLOB_RECURSE _TENSOR_DUMP_LIST "${CMAKE_CURRENT_SOURCE_DIR}/tensor_dump/*.cc")
list(APPEND _TOOLS_SRC_INDEPENDENT_LIST ${_TENSOR_DUMP_LIST})

file(GLOB_RECURSE _ALL_SILENTDETECT_LIST "${CMAKE_CURRENT_SOURCE_DIR}/silent_detect/*.cc")
set(_SILENTDETECT_LIST ${_ALL_SILENTDETECT_LIST})
list(APPEND _TOOLS_SRC_INDEPENDENT_LIST ${_SILENTDETECT_LIST})

file(GLOB_RECURSE _ERROR_HANDLER_LIST "${CMAKE_CURRENT_SOURCE_DIR}/error_handler/*.cc")
list(APPEND _TOOLS_SRC_INDEPENDENT_LIST ${_ERROR_HANDLER_LIST})

add_library(mindspore_tools SHARED ${_TOOLS_SRC_INDEPENDENT_LIST})

target_link_libraries(mindspore_tools PRIVATE mindspore_core mindspore_common proto_input
        mindspore_backend_common mindspore_hardware_abstract mindspore_memory_pool mindspore::securec)
set_target_properties(mindspore_tools PROPERTIES INSTALL_RPATH $ORIGIN)

if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_tools PRIVATE mindspore::pybind11_module)
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(mindspore_tools PRIVATE mindspore::ssl mindspore::crypto)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_tools PRIVATE TOOLS_DLL)
endif()
