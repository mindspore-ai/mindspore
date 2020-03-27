add_library(APItoPython OBJECT
  de_pipeline.cc
  python_bindings.cc
  )
target_include_directories(APItoPython PRIVATE ${pybind11_INCLUDE_DIRS})
