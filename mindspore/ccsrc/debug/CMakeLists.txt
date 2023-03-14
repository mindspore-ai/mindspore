include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/debug/)
include_directories(${CMAKE_BINARY_DIR})
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

set(_DEBUG_SRC_LIST)

set(_OFFLINE_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/debugger/tensor_summary.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/debugger/offline_debug/dbg_services.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/debugger/offline_debug/mi_pybind_register.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc"
)

if(ENABLE_DUMP_IR)
    file(GLOB_RECURSE _RDR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "rdr/*.cc")
endif()

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_DEBUGGER)
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/grpc_client.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/tensor_summary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/tensor_stat_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/csv_writer.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc"
        )
else()
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter_stub.cc"
        )
endif()

if(NOT ENABLE_SECURITY)
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/cpu_e2e_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/overflow_dumper.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_json_parser.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/npy_header.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc"
        )
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        list(APPEND _DEBUG_SRC_LIST
            "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/e2e_dump.cc"
            )
    endif()
endif()

set_property(SOURCE ${_DEBUG_SRC_LIST} ${_RDR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
list(APPEND _DEBUG_SRC_LIST ${_RDR_SRC_LIST})
list(LENGTH _DEBUG_SRC_LIST debug_files_size)
if(${debug_files_size} GREATER 0)
  add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST})
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
      target_compile_definitions(_mindspore_debug_obj PRIVATE BACKEND_DLL)
  endif()
  add_dependencies(_mindspore_debug_obj proto_input)
endif()
if(ENABLE_DEBUGGER)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        add_compile_options(-Wall -DOFFLINE_DBG_MODE -fPIC -O2)
        set_property(SOURCE ${_OFFLINE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_OFFLINE_DEBUG)
        add_library(_mindspore_offline_debug SHARED ${_OFFLINE_SRC_LIST})
        target_link_libraries(_mindspore_offline_debug PRIVATE
            mindspore::pybind11_module
            mindspore_core
            )
        set_target_properties(_mindspore_offline_debug PROPERTIES
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
            )
        if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
            set_target_properties(_mindspore_offline_debug PROPERTIES MACOSX_RPATH ON)
            set_target_properties(_mindspore_offline_debug PROPERTIES INSTALL_RPATH @loader_path/lib)
        else()
            set_target_properties(_mindspore_offline_debug PROPERTIES INSTALL_RPATH $ORIGIN/lib)
            target_link_options(_mindspore_offline_debug PRIVATE -Wl,-init,common_log_init)
        endif()
    endif()
endif()
