add_library(engine-tdt OBJECT
    tdt_plugin.cc
    ${EXAMPLE_SRCS}
    ${FEATURE_SRCS}
    )

add_dependencies(engine-tdt mindspore::protobuf)
