string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB_RECURSE OPENCL_KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../opencl/*.cc)
    add_library(opencl_kernel_mid OBJECT ${OPENCL_KERNEL_SRC})
    add_dependencies(opencl_kernel_mid fbs_src)
endif()
