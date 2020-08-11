include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
add_subdirectory(gvar)
message("************build core***************")

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
    add_compile_definitions(BUILDING_DLL)
endif()

file(GLOB_RECURSE CORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "abstract/*.cc"
        "base/*.cc"
        "c_ops/*.cc"
        "ir/*.cc"
        "utils/*.cc"
)
set_property(SOURCE ${CORE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_CORE)
add_library(mindspore_core STATIC ${CORE_SRC_LIST})
target_link_libraries(mindspore_core PRIVATE mindspore_gvar)
if (USE_GLOG)
    target_link_libraries(mindspore_core PRIVATE mindspore::glog)
endif ()
