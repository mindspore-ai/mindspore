file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
if(ENABLE_PYTHON)
    add_library(APItoPython OBJECT
            python/bindings/dataset/audio/bindings.cc
            python/bindings/dataset/audio/transform/bindings.cc
            python/bindings/dataset/callback/bindings.cc
            python/bindings/dataset/core/bindings.cc
            python/bindings/dataset/data_source/bindings.cc
            python/bindings/dataset/data_source/samplers/bindings.cc
            python/bindings/dataset/engine/cache/bindings.cc
            python/bindings/dataset/engine/datasetops/bindings.cc
            python/bindings/dataset/engine/consumer/bindings.cc
            python/bindings/dataset/engine/execute/bindings.cc
            python/bindings/dataset/engine/schema/bindings.cc
            python/bindings/dataset/engine/perf/bindings.cc
            python/bindings/dataset/kernels/bindings.cc
            python/bindings/dataset/vision/bindings.cc
            python/bindings/dataset/general/bindings.cc
            python/bindings/dataset/util/bindings.cc
            python/bindings/mindrecord/include/bindings.cc
            python/pybind_conversion.cc
            python/pybind_register.cc
            )
    target_include_directories(APItoPython PRIVATE ${pybind11_INCLUDE_DIRS})
endif()

add_library(cpp-API OBJECT
        audio.cc
        config.cc
        data_helper.cc
        datasets.cc
        execute.cc
        iterator.cc
        transforms.cc
        samplers.cc
        vision.cc
        )
