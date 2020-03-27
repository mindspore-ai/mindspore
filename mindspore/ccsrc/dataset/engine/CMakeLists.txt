add_subdirectory(datasetops)
if (ENABLE_TDTQUE)
  add_subdirectory(tdt)
endif ()

add_library(engine OBJECT
    execution_tree.cc
    data_buffer.cc
    data_schema.cc
    dataset_iterator.cc
    )
target_include_directories(engine PRIVATE ${pybind11_INCLUDE_DIRS})

if (ENABLE_TDTQUE)
  add_dependencies(engine engine-datasetops engine-datasetops-source engine-tdt)
else()
  add_dependencies(engine engine-datasetops engine-datasetops-source)
endif ()
