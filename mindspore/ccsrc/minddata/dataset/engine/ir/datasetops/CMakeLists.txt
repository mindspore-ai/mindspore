file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_subdirectory(source)

set(DATASET_ENGINE_IR_DATASETOPS_SRC_FILES
        dataset_node.cc
        batch_node.cc
        bucket_batch_by_length_node.cc
        build_sentence_piece_vocab_node.cc
        build_vocab_node.cc
        cache_lookup_node.cc
        cache_merge_node.cc
        cache_node.cc
        concat_node.cc
        epoch_ctrl_node.cc
        filter_node.cc
        map_node.cc
        project_node.cc
        rename_node.cc
        repeat_node.cc
        root_node.cc
        shuffle_node.cc
        skip_node.cc
        sync_wait_node.cc
        take_node.cc
        transfer_node.cc
        zip_node.cc
        )

add_library(engine-ir-datasetops OBJECT ${DATASET_ENGINE_IR_DATASETOPS_SRC_FILES})
