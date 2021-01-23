file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "common/*.cc"
    "mem_reuse/*.cc"
    "pass/*.cc"
    "somas/*.cc"
)

if(ENABLE_D)
    file(GLOB_RECURSE _D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend/*.cc"
        "graph_kernel/*.cc"
    )
    list(APPEND _PREACTIVATE_SRC_LIST ${_D_SRC_LIST})
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE _GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu/*.cc"
        "graph_kernel/*.cc"
    )
    list(APPEND _PREACTIVATE_SRC_LIST ${_GPU_SRC_LIST})
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

set_property(SOURCE ${_PREACTIVATE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PRE_ACT)
add_library(_mindspore_backend_optimizer_obj OBJECT ${_PREACTIVATE_SRC_LIST})
