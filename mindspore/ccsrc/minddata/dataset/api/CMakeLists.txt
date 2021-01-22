file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
if(ENABLE_PYTHON)
    add_library(APItoPython OBJECT
            python/bindings/dataset/callback/bindings.cc
            python/bindings/dataset/core/bindings.cc
            python/bindings/dataset/engine/cache/bindings.cc
            python/bindings/dataset/engine/datasetops/bindings.cc
            python/bindings/dataset/engine/datasetops/source/bindings.cc
            python/bindings/dataset/engine/datasetops/source/sampler/bindings.cc
            python/bindings/dataset/engine/gnn/bindings.cc
            python/bindings/dataset/include/datasets_bindings.cc
            python/bindings/dataset/include/iterator_bindings.cc
            python/bindings/dataset/include/execute_binding.cc
            python/bindings/dataset/include/schema_bindings.cc
            python/bindings/dataset/kernels/bindings.cc
            python/bindings/dataset/kernels/data/bindings.cc
            python/bindings/dataset/kernels/ir/bindings.cc
            python/bindings/dataset/kernels/ir/image/bindings.cc
            python/bindings/dataset/text/bindings.cc
            python/bindings/dataset/text/kernels/ir/bindings.cc
            python/bindings/mindrecord/include/bindings.cc
            python/pybind_conversion.cc
            python/pybind_register.cc
            )
    target_include_directories(APItoPython PRIVATE ${pybind11_INCLUDE_DIRS})
endif()


if(ENABLE_ACL)
    add_library(cpp-API OBJECT
            config.cc
            datasets.cc
            execute.cc
            iterator.cc
            minddata_eager.cc
            transforms.cc
            samplers.cc
            text.cc
            vision.cc
            )
else()
    add_library(cpp-API OBJECT
            config.cc
            datasets.cc
            execute.cc
            iterator.cc
            transforms.cc
            samplers.cc
            text.cc
            vision.cc
            )
endif()
