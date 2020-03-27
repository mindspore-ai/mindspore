file(GLOB_RECURSE _SESSION_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel_graph.cc"
        "session_basic.cc"
        "session_factory.cc"
        "anf_runtime_algorithm.cc"
        )
#TODO :  Not include session_context.cc
add_library(_mindspore_session_obj OBJECT ${_SESSION_ALL_SRC_FILES})

if(ENABLE_D)
        file(GLOB_RECURSE _D_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "ascend_session.cc"
            )
        add_library(_mindspore_session_obj  OBJECT ${_D_SRC_LIST})
endif()

if(ENABLE_GPU)
        file(GLOB_RECURSE _C_EXPRESSION_GPU_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "gpu_session.cc"
            )
        add_library(_c_expression_gpu_session_obj OBJECT ${_C_EXPRESSION_GPU_SRC_LIST})
endif()