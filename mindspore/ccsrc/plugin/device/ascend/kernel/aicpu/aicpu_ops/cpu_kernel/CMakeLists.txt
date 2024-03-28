set(CANN_CONTEXT "${ASCEND_PATH}/latest/tools/msopgen/template/custom_operator_sample/AICPU/Tensorflow/cpukernel/")
if(NOT EXISTS ${CANN_CONTEXT})
    message(FATAL_ERROR "${CANN_CONTEXT} not exists. Please check if the cann package is installed properly.")
endif()

include_directories(${CANN_CONTEXT})
include_directories(${ASCEND_CANN_AICPU_KERNEL_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ms_kernel MS_KERNELS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ms_kernel/environ ENVIRON_KERNELS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/utils UTILS_LISTS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/format_transfer FORMAT_TRANSFER_LISTS)
set(CPU_OPS_SRC
    ${MS_KERNELS}
    ${ENVIRON_KERNELS}
    ${CMAKE_CURRENT_SOURCE_DIR}/ms_kernel/random/utils.cc
    ${UTILS_LISTS}
)

add_library(mindspore_cpu_kernels SHARED ${CPU_OPS_SRC})

remove_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
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

set(ASCEND_SOC_VERSION $ENV{AICPU_SOC_VERSION})
if(EXISTS "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/${ASCEND_SOC_VERSION}/libcpu_kernels_context.a")
    set(CONTEXT_LIB "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/${ASCEND_SOC_VERSION}/libcpu_kernels_context.a")
elseif(EXISTS "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/libcpu_kernels_context.a")
    set(CONTEXT_LIB "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/libcpu_kernels_context.a")
else()
    message(FATAL_ERROR "Can not find libcpu_kernels_context.a in the environment."
                        "Please check if the cann package is installed properly.")
endif()

if(EXISTS "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/${ASCEND_SOC_VERSION}/libascend_protobuf.a")
    set(PROTOBUF_LIB "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/libascend_protobuf.a")
elseif(EXISTS "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/Ascend/libascend_protobuf.a")
    set(PROTOBUF_LIB "${ASCEND_CANN_AICPU_KERNEL_PATH}/lib/Ascend/libascend_protobuf.a")
else()
    message(FATAL_ERROR "Can not find libascend_protobuf.a in the environment."
                        "Please check if the cann package is installed properly.")
endif()


target_link_libraries(mindspore_cpu_kernels PRIVATE
    -ldl
    -Wl,--whole-archive
    ${CONTEXT_LIB}
    ${PROTOBUF_LIB}
    -Wl,--no-whole-archive
    -pthread
    )

set(INSTALL_LIBRARY_DIR lib/plugin)
install(TARGETS mindspore_cpu_kernels OPTIONAL
    EXPORT mindspore_cpu_kernels-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/ascend
)
