add_subdirectory(source)

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
    filter_op.cc
    )

