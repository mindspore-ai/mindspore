file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

set(DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES
        distributed_sampler.cc
        pk_sampler.cc
        random_sampler.cc
        sampler.cc
        sequential_sampler.cc
        subset_random_sampler.cc
        subset_sampler.cc
        weighted_random_sampler.cc
        )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES
            ${DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES}
            python_sampler.cc
            )
endif()

add_library(engine-datasetops-source-sampler OBJECT ${DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES})
