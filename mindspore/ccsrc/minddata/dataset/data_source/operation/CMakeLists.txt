file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_subdirectory(samplers)

set(DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES
        album_node.cc
        csv_node.cc
        image_folder_node.cc
        manifest_node.cc
        minddata_node.cc
        random_node.cc
        text_file_node.cc
        tf_record_node.cc
        )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES
            ${DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES}
            generator_node.cc
            )
endif()

add_library(data_source-operation OBJECT ${DATASET_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES})
