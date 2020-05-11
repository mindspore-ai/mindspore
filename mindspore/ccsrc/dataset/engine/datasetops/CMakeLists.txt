add_subdirectory(source)

file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(engine-datasetops OBJECT
    dataset_op.cc
    parallel_op.cc
    pipeline_op.cc
    barrier_op.cc
    batch_op.cc
    device_queue_op.cc
    map_op.cc
    project_op.cc
    rename_op.cc
    repeat_op.cc
    skip_op.cc
    take_op.cc
    shuffle_op.cc
    zip_op.cc
    concat_op.cc
    filter_op.cc
    )

