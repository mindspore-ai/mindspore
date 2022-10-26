include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})
add_subdirectory(kernel/cuda_impl)

########### mindspore_gpu.so #####
set(GPU_SUB_COMP
        hal/device
        hal/hardware
        hal/profiler
        kernel
        optimizer
        )

if(ENABLE_GPU)
    foreach(g_comp ${GPU_SUB_COMP})
        add_subdirectory(${g_comp})
        string(REPLACE "/" "_" sub ${g_comp})
        if(TARGET _mindspore_plugin_device_gpu_${sub}_obj)
            list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_gpu_${sub}_obj>)
            add_dependencies(_mindspore_plugin_device_gpu_${sub}_obj proto_input)
            if(CMAKE_SYSTEM_NAME MATCHES "Windows")
                target_compile_definitions(_mindspore_plugin_device_gpu_${sub}_obj PRIVATE BUILDING_GPU_DLL)
            endif()
        endif()
    endforeach()
endif()

### cxx api need file ###
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
file(GLOB_RECURSE API_INFER_GPU
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/graph/gpu/*.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/cxx_api/model/ms/*.cc"
        )
add_library(mindspore_axx_api_gpu_obj OBJECT ${API_INFER_GPU})
list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:mindspore_axx_api_gpu_obj>)

add_library(mindspore_gpu SHARED ${GPU_SUB_OBJECTS_SRC})
if(MSVC)
target_link_options(mindspore_gpu PRIVATE /FORCE:UNRESOLVED)
endif()
target_link_libraries(mindspore_gpu PRIVATE mindspore_backend mindspore_core mindspore_common)
target_link_libraries(mindspore_gpu PRIVATE mindspore_shared_lib)
target_link_libraries(mindspore_gpu PRIVATE proto_input mindspore::protobuf)
target_link_libraries(mindspore_gpu PRIVATE securec)
set_target_properties(mindspore_gpu PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
target_link_libraries(mindspore_gpu PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
if(NOT WIN32)
target_link_libraries(mindspore_gpu PRIVATE mindspore::ssl mindspore::crypto)
target_link_libraries(mindspore_gpu PRIVATE mindspore::event mindspore::event_pthreads
        mindspore::event_openssl -Wl,--no-as-needed mindspore::event_core ps_cache)
endif()

if(ENABLE_GPU)
    message("add gpu lib to mindspore_gpu")
    if(WIN32)
    target_link_libraries(mindspore_gpu PRIVATE cuda_ops
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/curand.lib
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/cudart.lib
            ${CUDA_PATH}/lib/x64/cuda.lib
            ${CUDA_PATH}/lib/x64/cusolver.lib
            ${CUDA_PATH}/lib/x64/cufft.lib
            ${CUDA_PATH}/lib/x64/cusparse.lib)
    else()
    target_link_libraries(mindspore_gpu PRIVATE cuda_ops
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcurand.so
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcusolver.so
            ${CUDA_PATH}/lib64/libcufft.so
            ${CUDA_PATH}/lib64/libcusparse.so)
    endif()

    if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
        target_link_libraries(mindspore_gpu PRIVATE gpu_hash_table)
    endif()
endif()

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(ENABLE_GPU)
        target_link_libraries(mindspore_gpu PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(ENABLE_GPU)
    if(ENABLE_MPI)
        set_target_properties(_ms_mpi PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
        set_target_properties(nvidia_collective PROPERTIES INSTALL_RPATH ${ORIGIN_PATH})
        set_target_properties(gpu_collective PROPERTIES INSTALL_RPATH ${ORIGIN_PATH})
    endif()
endif()
