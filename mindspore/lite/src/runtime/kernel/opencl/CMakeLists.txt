if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB_RECURSE OPENCL_KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/int8/*.cc)
    add_library(opencl_kernel_mid OBJECT ${OPENCL_KERNEL_SRC})
    add_dependencies(opencl_kernel_mid fbs_src)
endif()
