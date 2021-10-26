include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
add_subdirectory(gvar)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(NOT(BUILD_LITE))
    add_subdirectory(mindrt)
endif()

message("************ build core ***************")

file(GLOB_RECURSE CORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "abstract/*.cc"
  "base/*.cc"
  "ops/*.cc"
  "ir/*.cc"
  "utils/*.cc"
  "load_mindir/*.cc"
  )

if(ENABLE_SECURITY)
    file(GLOB_RECURSE _INFER_SUMMARY_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ops/*_summary.cc")
    list(REMOVE_ITEM CORE_SRC_LIST ${_INFER_SUMMARY_FILES})
endif()

file(GLOB_RECURSE PROTO_FILE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "proto/*.proto")
if(NOT(BUILD_LITE))
    ms_protobuf_generate_py(PROTO_SRCS PY_HDRS PY_PYS ${PROTO_FILE})
else()
    ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
  add_compile_definitions(BUILDING_DLL)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wuser-defined-warnings -Winconsistent-missing-override -Wno-delete-non-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${CORE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_CORE)
set_property(SOURCE ${PROTO_SRCS} PROPERTY COMPILE_OPTIONS -Wno-array-bounds)
add_library(mindspore_core STATIC ${CORE_SRC_LIST} ${PROTO_SRCS})
target_link_libraries(mindspore_core PRIVATE mindspore_gvar)
if(NOT(BUILD_LITE))
    target_link_libraries(mindspore_core PRIVATE mindrt_mid)
endif()

if(USE_GLOG)
  target_link_libraries(mindspore_core PRIVATE mindspore::glog)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if((NOT BUILD_LITE) OR (BUILD_LITE AND MSLITE_ENABLE_MODEL_ENCRYPTION))
        target_link_libraries(mindspore_core PRIVATE mindspore::crypto -pthread)
        add_compile_definitions(ENABLE_OPENSSL)
    endif()
endif()
