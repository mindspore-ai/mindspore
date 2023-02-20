include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(NOT "${CMAKE_C_FLAGS}" STREQUAL "")
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-exceptions" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()
if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()
add_subdirectory(mindrt)

if(NOT ENABLE_COVERAGE AND (CMAKE_SIZEOF_VOID_P EQUAL 4 OR NOT BUILD_LITE))
    file(GLOB_RECURSE CORE_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ops/*.cc")
    if(ENABLE_SECURITY)
        file(GLOB_RECURSE _INFER_SUMMARY_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ops/*_summary.cc")
        list(REMOVE_ITEM CORE_OPS_LIST ${_INFER_SUMMARY_FILES})
    endif()
else()
    include(${TOP_DIR}/mindspore/lite/cmake/merge.cmake)
    if(ENABLE_SECURITY)
        merge_files(${CMAKE_CURRENT_SOURCE_DIR}/ops/ ${CMAKE_BINARY_DIR}/merge/mindspore/core ops_merge
                    "_summary.cc$")
    else()
        merge_files(${CMAKE_CURRENT_SOURCE_DIR}/ops/ ${CMAKE_BINARY_DIR}/merge/mindspore/core ops_merge "")
    endif()
    file(GLOB_RECURSE CORE_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_BINARY_DIR}/merge/mindspore/core/ops_merge*.cc")
endif()

file(GLOB_RECURSE CORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "abstract/*.cc"
  "base/*.cc"
  "ir/*.cc"
  "utils/*.cc"
  "load_mindir/*.cc"
  "mindapi/src/*.cc"
  "expander/*.cc"
  )

if(NOT ONLY_BUILD_DEVICE_PLUGINS)
    set(CORE_SRC_LIST ${CORE_SRC_LIST} ${CORE_OPS_LIST})
endif()

if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    LIST(REMOVE_ITEM CORE_SRC_LIST utils/numa_interface.cc)
endif()

if(ENABLE_SECURITY)
    file(GLOB_RECURSE _INFER_SUMMARY_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ops/*_summary.cc")
    list(REMOVE_ITEM CORE_SRC_LIST ${_INFER_SUMMARY_FILES})
endif()

file(GLOB_RECURSE PROTO_FILE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "proto/*.proto")

ms_protobuf_generate_py(PROTO_SRCS PY_HDRS PY_PYS ${PROTO_FILE})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
  endif()
  add_compile_definitions(BUILDING_CORE_DLL)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wno-overloaded-virtual -Wno-user-defined-warnings -Winconsistent-missing-override -Wno-delete-non-virtual-dtor \
    -Wno-unused-private-field")
endif()

set(CORE_OBJECT_COUNT 1)

set_property(SOURCE ${CORE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_CORE)
add_library(core_proto_obj OBJECT ${PROTO_SRCS})
list(APPEND CORE_OBJECT_LIST $<TARGET_OBJECTS:core_proto_obj>)

src_separate_compile(
    OBJECT_NAME core_obj
    OBJECT_SIZE CORE_OBJECT_COUNT
    SRC_LIST ${CORE_SRC_LIST})
foreach(number RANGE 1 ${CORE_OBJECT_COUNT})
    list(APPEND CORE_OBJECT_LIST $<TARGET_OBJECTS:core_obj_${number}>)
    add_dependencies(core_obj_${number} core_proto_obj)
endforeach()
set(CORE_OBJECT_COUNT "${CORE_OBJECT_COUNT}" PARENT_SCOPE)

add_library(mindspore_core SHARED ${CORE_OBJECT_LIST})
target_link_libraries(mindspore_core PRIVATE mindspore::protobuf securec)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(mindspore_core PRIVATE -Wl,-init,mindspore_log_init)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_core PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_core PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_core PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

target_link_libraries(mindspore_core PRIVATE mindrt_mid)

if(USE_GLOG)
    target_link_libraries(mindspore_core PRIVATE mindspore::glog)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
    target_link_libraries(mindspore_core PRIVATE mindspore::crypto -pthread)
endif()

