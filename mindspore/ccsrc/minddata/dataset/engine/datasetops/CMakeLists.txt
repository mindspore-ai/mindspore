add_subdirectory(source)
add_subdirectory(map_op)

file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)

set(DATASET_ENGINE_DATASETOPS_SRC_FILES
    dataset_op.cc
    parallel_op.cc
    pipeline_op.cc
    batch_op.cc
    device_queue_op.cc
    project_op.cc
    rename_op.cc
    repeat_op.cc
    skip_op.cc
    take_op.cc
    shuffle_op.cc
    zip_op.cc
    concat_op.cc
    epoch_ctrl_op.cc
    cache_base_op.cc
    cache_lookup_op.cc
    cache_op.cc
    cache_merge_op.cc
    )

if(ENABLE_PYTHON)
    set(DATASET_ENGINE_DATASETOPS_SRC_FILES
        ${DATASET_ENGINE_DATASETOPS_SRC_FILES}
        bucket_batch_by_length_op.cc
        barrier_op.cc
        filter_op.cc
        build_vocab_op.cc
        build_sentence_piece_vocab_op.cc
        )
endif()

add_library(engine-datasetops OBJECT ${DATASET_ENGINE_DATASETOPS_SRC_FILES})
