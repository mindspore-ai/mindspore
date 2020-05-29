add_subdirectory(datasetops)
add_subdirectory(opt)
add_subdirectory(gnn)
if (ENABLE_TDTQUE)
  add_subdirectory(tdt)
endif ()

file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(engine OBJECT
    execution_tree.cc
    data_buffer.cc
    data_schema.cc
    dataset_iterator.cc
    )
target_include_directories(engine PRIVATE ${pybind11_INCLUDE_DIRS})

if (ENABLE_TDTQUE)
  add_dependencies(engine engine-datasetops engine-datasetops-source engine-tdt engine-opt engine-gnn)
else()
  add_dependencies(engine engine-datasetops engine-datasetops-source engine-opt engine-gnn)
endif ()
