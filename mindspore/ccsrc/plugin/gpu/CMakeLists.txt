include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})

########### mindspore_gpu.so #####
set(GPU_SUB_COMP
        graph_optimizer
        kernel_executor
        res_manager
        profiler)

if(ENABLE_GPU)
    file(GLOB GPU_DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(APPEND GPU_SUB_OBJECTS_SRC ${GPU_DEVICE_SRC_LIST})
    foreach(g_comp ${GPU_SUB_COMP})
        add_subdirectory(${g_comp})
        string(REPLACE "/" "_" sub ${g_comp})
        if(TARGET _mindspore_plugin_gpu_${sub}_obj)
            list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_gpu_${sub}_obj>)
            if(CMAKE_SYSTEM_NAME MATCHES "Windows")
                target_compile_definitions(_mindspore_plugin_gpu_${sub}_obj PRIVATE BUILDING_GPU_DLL)
            endif()
        endif()
    endforeach()
endif()

if(ENABLE_GPU)
    if(NOT ${TENSORRT_HOME} STREQUAL "")
        find_path(TENSORRT_HOME_INCLUDE NvInfer.h HINTS ${TENSORRT_HOME}/include)
        if(TENSORRT_HOME_INCLUDE STREQUAL TENSORRT_HOME_INCLUDE-NOTFOUND)
            message(FATAL_ERROR "Tensor-RT dir not exist ${TENSORRT_HOME}")
        endif()
        message("Enable GPU inference. Tensor-RT include dir: ${TENSORRT_HOME_INCLUDE}")
        set(ENABLE_GPU_INFER TRUE)
        add_compile_definitions(ENABLE_GPU_INFER)
        include_directories(${TENSORRT_HOME_INCLUDE})
        list(APPEND GPU_PLUGIN_SRC_LIST graph_optimizer/pass/inference/trt_pass/trt_loader.cc)
    endif()
    list(APPEND GPU_SUB_OBJECTS_SRC ${GPU_PLUGIN_SRC_LIST})
endif()

### cxx api need file ###
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
set_property(SOURCE ${GPU_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(mindspore_gpu SHARED ${GPU_SUB_OBJECTS_SRC})
set_target_properties(mindspore_gpu PROPERTIES SOVERSION ${CUDA_VERSION})
if(MSVC)
target_link_options(mindspore_gpu PRIVATE /FORCE:UNRESOLVED)
endif()
target_link_libraries(mindspore_gpu PRIVATE mindspore_ms_backend mindspore_core mindspore_ops mindspore_common
    mindspore_hardware_abstract mindspore_profiler mindspore_backend_common mindspore_runtime_utils
    mindspore_ops_cpu)
target_link_libraries(mindspore_gpu PRIVATE -Wl,--no-as-needed mindspore_pyboost -Wl,--as-needed)
target_link_libraries(mindspore_gpu PRIVATE -Wl,--whole-archive mindspore_ops_gpu -Wl,--no-whole-archive)
set_target_properties(mindspore_gpu PROPERTIES INSTALL_RPATH
    ${ORIGIN_PATH}:${ORIGIN_PATH}/gpu:${ORIGIN_PATH}/gpu${CUDA_VERSION}:${ORIGIN_PATH}/../:${CUDA_PATH}/lib64)
if(ENABLE_CPU)
    target_link_libraries(mindspore_gpu PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()
if(ENABLE_GPU)
    message("add gpu lib to mindspore_gpu")
    if(WIN32)
    target_link_libraries(mindspore_gpu PRIVATE
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/curand.lib
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/cudart.lib
            ${CUDA_PATH}/lib/x64/cuda.lib
            ${CUDA_PATH}/lib/x64/cusolver.lib
            ${CUDA_PATH}/lib/x64/cufft.lib
            ${CUDA_PATH}/lib/x64/cusparse.lib
            ${CUDA_PATH}/lib/x64/nvrtc.lib
            )
    else()
    target_link_libraries(mindspore_gpu PRIVATE
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcurand.so
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcusolver.so
            ${CUDA_PATH}/lib64/libcufft.so
            ${CUDA_PATH}/lib64/libcusparse.so
            ${CUDA_PATH}/lib64/libnvrtc.so
            )
    endif()
endif()

if(ENABLE_GPU)
    if(ENABLE_MPI)
        set_target_properties(nvidia_collective PROPERTIES INSTALL_RPATH ${ORIGIN_PATH})
    endif()
endif()
