set(CPU_PROTO_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_attr.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_node_def.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor_shape.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto/proto/cpu_tensor.proto
)

ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${CPU_PROTO_SRC})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/common COMMON_LISTS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/cpu_proto CPU_PROTO_LISTS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/utils UTILS_LISTS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/format_transfer FORMAT_TRANSFER_LISTS)
set(CPU_SRC
    ${COMMON_LISTS}
    ${CPU_PROTO_LISTS}
    ${UTILS_LISTS}
    ${FORMAT_TRANSFER_LISTS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../aicpu_sharder/aicpu_context.cc
)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ms_kernel MS_KERNELS)
set(CPU_OPS_SRC
    ${MS_KERNELS}
)

add_library(mindspore_cpu_kernels SHARED
    ${PROTO_SRCS}
    ${CPU_SRC}
    ${CPU_OPS_SRC}
)

target_compile_options(mindspore_cpu_kernels PRIVATE
    -march=armv8-a
    -O2
    -fvisibility-inlines-hidden
    -fvisibility=hidden
    -fno-strict-aliasing
    -fno-common
)

target_link_libraries(mindspore_cpu_kernels PRIVATE
    -ldl
    -shared
    PUBLIC
    ${SECUREC_ARM_LIBRARY}
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -Wl,-Bsymbolic
    -rdynamic
    mindspore::protobuf_arm
    -pthread
    )

set(INSTALL_LIBRARY_DIR lib/plugin)
install(TARGETS mindspore_cpu_kernels OPTIONAL
    EXPORT mindspore_cpu_kernels-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/ascend
)
