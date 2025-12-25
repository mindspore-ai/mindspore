include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

file(GLOB_RECURSE PYBOOST_META_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "pyboost/*.cc"
    "pyboost/auto_generate/*.cc"
    )

set_property(SOURCE ${PYBOOST_META_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)

add_library(mindspore_ops_meta OBJECT ${PYBOOST_META_SRC_LIST})
target_link_libraries(mindspore_ops_meta PRIVATE mindspore_core mindspore_ops mindspore_common
    mindspore_hardware_abstract mindspore_profiler proto_input mindspore_runtime_utils mindspore::pybind11_module
    mindspore::securec mindspore_pyboost mindspore_pynative_utils)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_ops_meta PRIVATE OPS_HOST_DLL)
endif()