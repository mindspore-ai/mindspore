file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_subdirectory(source)

set(DATASET_ENGINE_IR_DATASETOPS_SRC_FILES
        batch_node.cc
        concat_node.cc
        project_node.cc
        rename_node.cc
        repeat_node.cc
        shuffle_node.cc
        take_node.cc
        )

if (NOT ENABLE_ANDROID)
    set(DATASET_ENGINE_IR_DATASETOPS_SRC_FILES
            ${DATASET_ENGINE_IR_DATASETOPS_SRC_FILES}
            bucket_batch_by_length_node.cc
            build_vocab_node.cc)
endif ()

add_library(engine-ir-datasetops OBJECT ${DATASET_ENGINE_IR_DATASETOPS_SRC_FILES})
