file(GLOB_RECURSE KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	"kernel_build_info.cc"
	"kash/*.cc"
	"common_utils.cc"
	"oplib/*.cc"
)

if (ENABLE_D)
	file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
		"kernel_query.cc"
		"kernel_fusion.cc"
		"tbe/*.cc"
		"aicpu/*.cc"
		"rts/*.cc"
		"hccl/*.cc"
	)
	add_compile_definitions(ENABLE_D)
endif ()

if (ENABLE_CPU)
    file(GLOB_RECURSE CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "cpu/*.cc"
    )
    
    if (NOT ENABLE_MPI)
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/allgather_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/reduce_scatter_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/embedding_look_up_comm_grad_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/embedding_look_up_cpu_kernel.cc")
        list(REMOVE_ITEM CPU_SRC_LIST "cpu/sub_cpu_kernel.cc")
    endif ()
endif ()

if (ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu/*.cu"
        "akg/gpu/*.cc"
        "akg/akgkernelbuild.cc"
        "akg/akg_kernel_attrs_process.cc"
	)

    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/*.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "gpu/nccl/nccl_gpu_kernel.cc")

    if (ENABLE_MPI)
        include(ExternalProject)
        file(GLOB_RECURSE GPU_NCCL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gpu/nccl/*.cc")
        list(APPEND GPU_SRC_LIST ${GPU_NCCL_LIST})
	endif ()

	# add_library(_mindspore_kernel_cuda_obj OBJECT ${CUDA_SRC_LIST})
endif()

set_property(SOURCE ${KERNEL_SRC_LIST} ${CPU_SRC_LIST} ${GPU_SRC_LIST} ${D_SRC_LIST}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(_mindspore_kernel_obj OBJECT ${KERNEL_SRC_LIST} ${CPU_SRC_LIST} ${GPU_SRC_LIST} ${D_SRC_LIST})
