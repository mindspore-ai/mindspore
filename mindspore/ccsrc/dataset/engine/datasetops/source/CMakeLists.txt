ms_protobuf_generate(EXAMPLE_SRCS EXAMPLE_HDRS example.proto)
ms_protobuf_generate(FEATURE_SRCS FEATURE_HDRS feature.proto)
add_subdirectory(sampler)
add_library(engine-datasetops-source OBJECT
    generator_op.cc
    io_block.cc
    mindrecord_op.cc
    storage_client.cc
    storage_op.cc
    tf_buffer.cc
    tf_client.cc
    tf_reader_op.cc
    image_folder_op.cc
    mnist_op.cc
    voc_op.cc
    ${EXAMPLE_SRCS}
    ${FEATURE_SRCS}
    manifest_op.cc
    cifar_op.cc
    celeba_op.cc
    )

add_dependencies(engine-datasetops-source mindspore::protobuf)
