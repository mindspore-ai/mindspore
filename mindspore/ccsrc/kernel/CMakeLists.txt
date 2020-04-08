file(GLOB_RECURSE _SESSION_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel_query.cc"
        "kernel_fusion.cc"
        "kernel_build_info.cc"
        "kash/*.cc"
        "common_utils.cc"
        "oplib/*.cc"
        )

add_library(_mindspore_kernel_obj OBJECT ${_SESSION_ALL_SRC_FILES})

if(ENABLE_GPU)
		file(GLOB_RECURSE _CUDA_GPU_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
						"gpu/*.cu"
						"akg/gpu/*.cc"
						)
		add_library(_cuda_gpu_kernel_obj  OBJECT ${_CUDA_GPU_SRC_LIST})

		file(GLOB_RECURSE _C_EXPRESSION_GPU_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
						"gpu/*.cc"
						)
		list(REMOVE_ITEM _C_EXPRESSION_GPU_SRC_LIST "gpu/nccl/nccl_gpu_kernel.cc")
		add_library(_c_expression_gpu_device_obj OBJECT ${_C_EXPRESSION_GPU_SRC_LIST})				
endif()

if(ENABLE_D)
    file(GLOB_RECURSE _D_SRC_LIST  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
						"tbe/*.cc"
						"aicpu/*.cc"
						"mng/*.cc"
						"hccl/*.cc"
            )
    target_sources(_mindspore_kernel_obj  PRIVATE ${_D_SRC_LIST})
endif()
