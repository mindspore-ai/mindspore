add_subdirectory(sampler)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
    io_block.cc
    image_folder_op.cc
    mnist_op.cc
    coco_op.cc
    cifar_op.cc
    random_data_op.cc
    celeba_op.cc
    text_file_op.cc
    clue_op.cc
    csv_op.cc
    album_op.cc
    )

set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
    ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
    mindrecord_op.cc
    tf_reader_op.cc
    )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES
        ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
        generator_op.cc
        voc_op.cc
        manifest_op.cc
        )
endif()

add_library(engine-datasetops-source OBJECT ${DATASET_ENGINE_DATASETOPS_SOURCE_SRC_FILES})
