add_subdirectory(sampler)
file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(engine-datasetops-source OBJECT
    generator_op.cc
    io_block.cc
    mindrecord_op.cc
    tf_reader_op.cc
    image_folder_op.cc
    mnist_op.cc
    voc_op.cc
    coco_op.cc
    manifest_op.cc
    cifar_op.cc
    random_data_op.cc
    celeba_op.cc
    text_file_op.cc
    clue_op.cc
    )