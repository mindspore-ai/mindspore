include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
add_subdirectory(gvar)

if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(NOT(COMPILE_LITE))
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
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
  add_compile_definitions(BUILDING_DLL)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wuser-defined-warnings -Winconsistent-missing-override -Wno-delete-non-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${CORE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_CORE)
add_library(mindspore_core STATIC ${CORE_SRC_LIST})
target_link_libraries(mindspore_core PRIVATE mindspore_gvar)

if(USE_GLOG)
  target_link_libraries(mindspore_core PRIVATE mindspore::glog)
endif()
