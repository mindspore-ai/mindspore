file(GLOB_RECURSE CUDA_OPS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cu")

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(${CUDA_VERSION} VERSION_LESS 11.0)
    string(REPLACE "-std=c++17" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
else()
    string(REPLACE "-std=c++17" "-std=c++14" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set_property(SOURCE ${CUDA_OPS_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
if(ENABLE_GPU)
    add_library(cuda_common_obj OBJECT cuda_common.cc)
    target_compile_options(cuda_common_obj PRIVATE "-std=c++17")
    cuda_add_library(cuda_ops SHARED ${CUDA_OPS_SRC_LIST} $<TARGET_OBJECTS:cuda_common_obj>)
    message("add gpu lib to cuda_ops")
    target_link_libraries(cuda_ops mindspore_core cublas
            ${CUDA_PATH}/lib64/libcurand.so
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcusolver.so
            ${CUDA_PATH}/lib64/libcufft.so)
endif()