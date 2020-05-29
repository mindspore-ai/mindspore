file(GLOB_RECURSE _SESSION_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_graph.cc"
    "session_basic.cc"
    "session_factory.cc"
    "anf_runtime_algorithm.cc"
)

if (ENABLE_GPU)
    file(GLOB_RECURSE _GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu_session.cc"
        )
    list(APPEND _SESSION_SRC_LIST ${_GPU_SRC_LIST})
endif ()

if (ENABLE_CPU)
    file(GLOB_RECURSE _CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "cpu_session.cc"
        )
    list(APPEND _SESSION_SRC_LIST ${_CPU_SRC_LIST})
endif ()

if (ENABLE_D)
    file(GLOB_RECURSE _D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend_session.cc"
        "ascend_control_parser.cc"
        )
    list(APPEND _SESSION_SRC_LIST ${_D_SRC_LIST})
endif ()

set_property(SOURCE ${_SESSION_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_SESSION)
add_library(_mindspore_session_obj OBJECT ${_SESSION_SRC_LIST})
