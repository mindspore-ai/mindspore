file(GLOB_RECURSE HARDWARE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "collective/*.cc"
  "data_queue/*.cc"
  "device_context/*.cc"
  "event/*.cc"
  "kernel_base/oplib/*.cc"
  "memory_manager/*.cc"
  "stream/*.cc")

file(GLOB KERNEL_BASE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "kernel_base/*.cc"
)

list(APPEND HARDWARE_SRC_LIST ${KERNEL_BASE_SRC_LIST})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  list(REMOVE_ITEM HARDWARE_SRC_LIST "kernel_base/duplex_pipe.cc")
else()
  list(REMOVE_ITEM HARDWARE_SRC_LIST "kernel_base/duplex_pipe_win.cc")
endif()

file(GLOB_RECURSE GRAPH_KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_base/graph_fusion/framework_utils.cc"
    "kernel_base/graph_fusion/graph_kernel_flags.cc"
    "kernel_base/graph_fusion/graph_kernel_info.cc"
    "kernel_base/graph_fusion/kash/*.cc"
    "kernel_base/graph_fusion/graph_kernel/graph_kernel_comm_info_manager.cc"
)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel_base/graph_fusion/graph_kernel/node.cc"
        "kernel_base/graph_fusion/graph_kernel/graph_kernel_json_generator.cc"
        "kernel_base/graph_fusion/graph_kernel/graph_kernel_builder.cc"
        "kernel_base/graph_fusion/graph_kernel/graph_kernel_builder_manager.cc"
        "kernel_base/graph_fusion/graph_kernel/set_infershape_functor.cc"
        "kernel_base/graph_fusion/graph_kernel/fake_abstract_shape.cc"
        "kernel_base/graph_fusion/graph_kernel/akg/akg_kernel_build.cc"
        "kernel_base/graph_fusion/graph_kernel/akg/akg_kernel_json_decoder.cc"
        "kernel_base/graph_fusion/graph_kernel/dynamic_akg/dynamic_akg_kernel_build.cc"
        "kernel_base/graph_fusion/graph_kernel/kernel_packet/*.cc"
        "kernel_base/graph_fusion/graph_kernel/symbol_engine/*.cc"
        "kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/*.cc"
    )
    list(APPEND GRAPH_KERNEL_SRC_LIST ${AKG_SRC_LIST})
else()
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel_base/graph_fusion/graph_kernel/node.cc"
        "kernel_base/graph_fusion/graph_kernel/graph_kernel_json_generator.cc"
        "kernel_base/graph_fusion/graph_kernel/akg/akg_kernel_json_decoder.cc"
        "kernel_base/graph_fusion/graph_kernel/kernel_packet/*.cc"
        "kernel_base/graph_fusion/graph_kernel/set_infershape_functor.cc"
        "kernel_base/graph_fusion/graph_kernel/fake_abstract_shape.cc"
        "kernel_base/graph_fusion/graph_kernel/symbol_engine/*.cc"
        "kernel_base/graph_fusion/graph_kernel/symbol_engine/jit/*.cc"
    )
    list(APPEND GRAPH_KERNEL_SRC_LIST ${AKG_SRC_LIST})
endif()



if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
  string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(gsm)
list(APPEND HARDWARE_SRC_LIST ${RES_SRC_LIST} ${GRAPH_KERNEL_SRC_LIST})
set_property(SOURCE ${HARDWARE_SRC_LIST} PROPERTY SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(mindspore_hardware_abstract SHARED ${HARDWARE_SRC_LIST})
target_link_libraries(mindspore_hardware_abstract PRIVATE mindspore_core mindspore_common
    mindspore_memory_pool mindspore_runtime_pipeline mindspore_ops mindspore::pybind11_module)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_hardware_abstract PRIVATE RUNTIME_HARDWARE_DLL)
endif()
set_target_properties(mindspore_hardware_abstract PROPERTIES INSTALL_RPATH $ORIGIN)
