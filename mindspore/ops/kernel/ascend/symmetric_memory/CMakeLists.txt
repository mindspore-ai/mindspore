file(GLOB_RECURSE SYMMETRIC_MEMORY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)
list(REMOVE_ITEM SYMMETRIC_MEMORY_SRC_LIST
    "symmetric_memory_kernel_build.cc"
    "symmetric_memory_ascend_adapter.cc"
)

set(ME_KERNELS_SYMMETRIC_MEMORY_LIB_PATH
    $ORIGIN/ms_kernels_symmetric_memory/symmetric_memory_kernel/lib)
set(ME_KERNELS_SYMMETRIC_MEMORY_LIB_PATH
    $ORIGIN/ms_kernels_symmetric_memory/shmem/memfabric_hybrid/lib:${ME_KERNELS_SYMMETRIC_MEMORY_LIB_PATH})
set(ME_KERNELS_SYMMETRIC_MEMORY_LIB_PATH
    $ORIGIN/ms_kernels_symmetric_memory/shmem/shmem/lib:${ME_KERNELS_SYMMETRIC_MEMORY_LIB_PATH})

if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    include_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    include_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/symmetric_memory_kernel)
    include_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/symmetric_memory_kernel/include)

    link_directories($ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME}/symmetric_memory_kernel/lib)

    add_compile_definitions(ENABLE_SYMMETRIC_MEMORY_KERNELS)
    add_library(mindspore_symmetric_memory_kernels SHARED ${SYMMETRIC_MEMORY_SRC_LIST})

    target_link_libraries(mindspore_symmetric_memory_kernels PRIVATE mindspore_ascend ms_kernels_symmetric_memory
        mindspore_internal_kernels)
    set_target_properties(mindspore_symmetric_memory_kernels PROPERTIES INSTALL_RPATH
        $ORIGIN:$ORIGIN/ascend/:$ORIGIN/../:${ME_KERNELS_SYMMETRIC_MEMORY_LIB_PATH})
endif()
