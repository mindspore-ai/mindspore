add_subdirectory(datasetops)
add_subdirectory(opt)
add_subdirectory(gnn)
add_subdirectory(ir)
add_subdirectory(perf)
add_subdirectory(cache)

if(ENABLE_TDTQUE)
    add_subdirectory(tdt)
endif()

file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
set(SRC_FILES_LIST
        execution_tree.cc
        data_buffer.cc
        data_schema.cc
        dataset_iterator.cc
        tree_adapter.cc
        tree_adapter_lite.cc
        runtime_context.cc
        python_runtime_context.cc
        consumers/pull_based_tree_consumer.cc
        consumers/tree_consumer.cc
        serdes.cc
        )
if(ENABLE_PYTHON)
    set(SRC_FILES_LIST
            ${SRC_FILES_LIST}
            python_runtime_context.cc
            consumers/python_tree_consumer.cc
            )
endif()

add_library(engine OBJECT ${SRC_FILES_LIST})

if(ENABLE_PYTHON)
    target_include_directories(engine PRIVATE ${pybind11_INCLUDE_DIRS})
endif()

add_dependencies(engine engine-datasetops
                    engine-datasetops-source
                    engine-opt
                    engine-gnn
                    engine-perf
                    engine-cache-client
                    engine-datasetops-mapop
                    )

if(ENABLE_TDTQUE)
    add_dependencies(engine engine-tdt)
endif()
