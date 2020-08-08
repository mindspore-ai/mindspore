file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
if (ENABLE_PYTHON)
    add_library(APItoPython OBJECT
            python/de_pipeline.cc
            python/pybind_register.cc
            python/bindings.cc
            python/bindings/dataset/engine/cache/bindings.cc
            python/bindings/dataset/core/bindings.cc
            python/bindings/dataset/kernels/data/bindings.cc
            python/bindings/dataset/kernels/bindings.cc
            python/bindings/dataset/engine/datasetops/bindings.cc
            python/bindings/dataset/engine/datasetops/source/bindings.cc
            python/bindings/dataset/engine/gnn/bindings.cc
            python/bindings/dataset/kernels/image/bindings.cc
            python/bindings/dataset/engine/datasetops/source/sampler/bindings.cc
            python/bindings/dataset/text/bindings.cc
            python/bindings/dataset/text/kernels/bindings.cc
            python/bindings/mindrecord/include/bindings.cc
            )
    target_include_directories(APItoPython PRIVATE ${pybind11_INCLUDE_DIRS})
endif ()

add_library(cpp-API OBJECT
        datasets.cc
        iterator.cc
        transforms.cc
        samplers.cc
        )
