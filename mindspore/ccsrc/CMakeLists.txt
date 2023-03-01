## common setting
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/src)

# graphengine include directories
if(ENABLE_D OR ENABLE_ACL OR ENABLE_TESTCASES)
    include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)
endif()

set(SERVER_FLATBUFFER_OUTPUT "${CMAKE_BINARY_DIR}/schema")

## Function for setting NVCC flag
function(set_nvcc_flag CUDA_NVCC_FLAGS)
    # Detect gpu archs by cudaGetDeviceProperties.
    message("Detect gpu arch on this device.")
    set(cu_file "${CMAKE_SOURCE_DIR}/build/mindspore/ccsrc/get_device_compute_capabilities.cu")
    file(WRITE ${cu_file} ""
        "#include <cuda_runtime.h>\n"
        "#include <cstdio>\n"
        "int main () {\n"
        " int dev_num = 0;\n"
        " if (cudaGetDeviceCount(&dev_num) != cudaSuccess) return -1;\n"
        " if (dev_num < 1) return -1;\n"
        " for (int dev_id = 0; dev_id < dev_num; ++dev_id) {\n"
        "    cudaDeviceProp prop;"
        "    if (cudaGetDeviceProperties(&prop, dev_id) == cudaSuccess) {\n"
        "      printf(\"%d.%d \", prop.major, prop.minor);\n"
        "    }\n"
        "  }\n"
        "  return 0;\n"
        "}\n")
    # Build and run cu_file, get the result from properties.
    if(NOT MSVC)
        set(CUDA_LIB_PATH ${CUDA_PATH}/lib64/libcudart.so)
    else()
        set(CUDA_LIB_PATH ${CUDA_PATH}/lib/x64/cudart.lib)
    endif()
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR ${CMAKE_SOURCE_DIR}/build/mindspore/ccsrc/ ${cu_file}
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
            LINK_LIBRARIES ${CUDA_LIB_PATH}
            RUN_OUTPUT_VARIABLE compute_cap)
    set(cuda_archs_bin)
    if(RUN_RESULT_VAR EQUAL 0)
        string(REGEX REPLACE "[ \t]+" ";" compute_cap "${compute_cap}")
        list(REMOVE_DUPLICATES compute_cap)
        foreach(arch ${compute_cap})
            set(arch_bin)
            if(arch MATCHES "^([0-9]\\.[0-9](\\[0-9]\\.[0-9]\\))?)$")
                set(arch_bin ${CMAKE_MATCH_1})
            else()
                message(FATAL_ERROR "Unknown CUDA arch Name ${arch} !")
            endif()
            if(NOT arch_bin)
                message(FATAL_ERROR "arch_bin was not set !")
            endif()
            list(APPEND cuda_archs_bin ${arch_bin})
        endforeach()
    else()
        message("Failed to detect gpu arch automatically, build a base arch 6.0.")
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_60)
    endif()
    # Get build flag from env to choose common/auto build.
    set(NVCC_ARCH_FLAG_FROM_ENV $ENV{CUDA_ARCH})
    if(NVCC_ARCH_FLAG_FROM_ENV STREQUAL "common")
        message("Build common archs for release.")
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60
                                    -gencode=arch=compute_61,code=sm_61
                                    -gencode=arch=compute_70,code=sm_70)
        if(${CUDA_VERSION} VERSION_GREATER "9.5")
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
            if(${CUDA_VERSION} VERSION_LESS "11.0")
                list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=compute_75)
            endif()
        endif()
        if(${CUDA_VERSION} VERSION_GREATER "10.5")
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=sm_80)
            if(${CUDA_VERSION} VERSION_LESS "11.1")
                list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=compute_80)
            endif()
        endif()
        if(NOT ${CUDA_VERSION} VERSION_LESS "11.1")
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_86,code=compute_86)
        endif()
    else()
        message("Auto build for arch(s) " ${cuda_archs_bin})
        string(REGEX REPLACE "\\." "" cuda_archs_bin "${cuda_archs_bin}")
        string(REGEX MATCHALL "[0-9()]+" cuda_archs_bin "${cuda_archs_bin}")
        foreach(arch ${cuda_archs_bin})
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_${arch},code=sm_${arch})
        endforeach()
        # For auto build, it only generate the code for indeed arch, so add sm_60 as a default arch
        # to avoid error in different archs. It may increase the compilation time.
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_60)
    endif()
    list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
    message("Final CUDA_NVCC_FLASG " ${CUDA_NVCC_FLAGS})

    list(APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
    if(MSVC AND ${CUDA_VERSION} VERSION_GREATER_EQUAL "11.6")
        list(APPEND CUDA_NVCC_FLAGS -t0)
    endif()
    set(${CUDA_NVCC_FLAGS} ${${CUDA_NVCC_FLAGS}} PARENT_SCOPE)
endfunction()

if(ENABLE_GPU AND GPU_BACKEND_CUDA)
    find_package(CUDA REQUIRED)
    find_package(Threads)
    if(${CUDA_VERSION} VERSION_LESS ${MS_REQUIRE_CUDA_VERSION})
        message(FATAL_ERROR "The minimum CUDA version ${MS_REQUIRE_CUDA_VERSION} is required, \
              but only CUDA ${CUDA_VERSION} found.")
    endif()
    enable_language(CUDA)
    if(NOT CUDA_PATH OR CUDA_PATH STREQUAL "")
        if(DEFINED ENV{CUDA_HOME} AND NOT $ENV{CUDA_HOME} STREQUAL "")
            set(CUDA_PATH $ENV{CUDA_HOME})
        else()
            set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
        endif()
    endif()

    if(DEFINED ENV{CUDNN_HOME} AND NOT $ENV{CUDNN_HOME} STREQUAL "")
        set(CUDNN_INCLUDE_DIR $ENV{CUDNN_HOME}/include)
        if(WIN32)
        set(CUDNN_LIBRARY_DIR $ENV{CUDNN_HOME}/lib $ENV{CUDNN_HOME}/lib/x64)
        else()
        set(CUDNN_LIBRARY_DIR $ENV{CUDNN_HOME}/lib64)
        endif()
        find_path(CUDNN_INCLUDE_PATH cudnn.h HINTS ${CUDNN_INCLUDE_DIR} NO_DEFAULT_PATH)
        find_library(CUDNN_LIBRARY_PATH "cudnn" HINTS ${CUDNN_LIBRARY_DIR} NO_DEFAULT_PATH)
        if(WIN32)
        find_library(CUBLAS_LIBRARY_PATH "cublas" HINTS ${CUDA_PATH}/lib/x64)
        else()
        find_library(CUBLAS_LIBRARY_PATH "cublas" HINTS ${CUDNN_LIBRARY_DIR})
        endif()
        if(CUDNN_INCLUDE_PATH STREQUAL CUDNN_INCLUDE_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn header file, please set environment variable CUDNN_HOME to \
                    cudnn installation position.")
        endif()
        if(CUDNN_LIBRARY_PATH STREQUAL CUDNN_LIBRARY_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn library file, please set environment variable CUDNN_HOME to \
                    cudnn installation position.")
        endif()
    else()
        list(APPEND CMAKE_PREFIX_PATH  ${CUDA_TOOLKIT_ROOT_DIR})
        find_path(CUDNN_INCLUDE_PATH cudnn.h PATH_SUFFIXES cuda/inclulde include cuda)
        find_library(CUDNN_LIBRARY_PATH "cudnn" PATH_SUFFIXES cuda/lib64 lib64 lib cuda/lib lib/x86_64-linux-gnu)
        find_library(CUBLAS_LIBRARY_PATH "cublas" PATH_SUFFIXES cuda/lib64 lib64 lib cuda/lib lib/x86_64-linux-gnu)
        if(CUDNN_INCLUDE_PATH STREQUAL CUDNN_INCLUDE_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn header file, if cudnn library is not installed, please put \
                    cudnn header file in cuda include path or user include path(eg. /usr/local/cuda/include; \
                    /usr/local/include; /usr/include), if cudnn library is installed in other position, please \
                    set environment variable CUDNN_HOME to cudnn installation position, there should be cudnn.h \
                    in {CUDNN_HOME}/include.")
        endif()
        if(CUDNN_LIBRARY_PATH STREQUAL CUDNN_LIBRARY_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn library file, if cudnn library is not installed, please put \
                    cudnn library file in cuda library path or user library path(eg. /usr/local/cuda/lib64; \
                    /usr/local/lib64; /usr/lib64; /usr/local/lib; /usr/lib), if cudnn library is installed in other \
                    position, please set environment variable CUDNN_HOME to cudnn installation position, there should \
                    be cudnn library file in {CUDNN_HOME}/lib64.")
        endif()
    endif()

    if(NOT CUPTI_INCLUDE_DIRS OR CUPTI_INCLUDE_DIRS STREQUAL "")
        set(CUPTI_INCLUDE_DIRS  ${CUDA_PATH}/extras/CUPTI/include)
    endif()
    message("CUDA_PATH: ${CUDA_PATH}")
    message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message("CUDNN_INCLUDE_PATH: ${CUDNN_INCLUDE_PATH}")
    message("CUDNN_LIBRARY_PATH: ${CUDNN_LIBRARY_PATH}")
    message("CUBLAS_LIBRARY_PATH: ${CUBLAS_LIBRARY_PATH}")
    message("CUPTI_INCLUDE_DIRS: ${CUPTI_INCLUDE_DIRS}")
    include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})
    ## set NVCC ARCH FLAG
    set(CUDA_NVCC_FLAGS)
    set_nvcc_flag(CUDA_NVCC_FLAGS)
    if(NOT MSVC)
    add_definitions(-Wno-unknown-pragmas) # Avoid compilation warnings from cuda/thrust
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        list(APPEND CUDA_NVCC_FLAGS -G)
        message("CUDA_NVCC_FLAGS" ${CUDA_NVCC_FLAGS})
    endif()
    set(NVCC_TMP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS ${NVCC_TMP_CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_GPU)
endif()

if(ENABLE_CPU)
    if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(PLATFORM_ARM64 "on")
        set(X86_64_SIMD "off")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(X86_64_SIMD "off")
    elseif("${X86_64_SIMD}" STREQUAL "off" AND NOT ${ENABLE_ASAN})
        set(X86_64_SIMD "avx512")
    endif()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/cpu/kernel)
    add_subdirectory(plugin/device/cpu/kernel/nnacl)
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
    link_directories(${CMAKE_SOURCE_DIR}/build/mindspore/graphengine)
else()
    if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 \
        -Wno-delete-non-abstract-non-virtual-dtor -Wno-unused-private-field -Wno-overloaded-virtual \
        -Wno-unused-const-variable -Wno-pessimizing-move -Wno-range-loop-analysis -Wno-mismatched-tags \
        -Wno-c++11-narrowing")
    else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
endif()

# Set compile flags to ensure float compute consistency.
if(NOT MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")
endif()

if(ENABLE_MPI)
    add_compile_definitions(ENABLE_MPI)
endif()

## make protobuf files
file(GLOB ONNX_PROTO "" ${CMAKE_SOURCE_DIR}/third_party/proto/onnx/onnx.proto)
message("onnx proto path is :" ${ONNX_PROTO})
ms_protobuf_generate(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${ONNX_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/ps/core")
file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ps/core/protos/*.proto")
ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${COMM_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/distributed/cluster/topology")
file(GLOB_RECURSE DISTRIBUTED_CLUSTER_TOPOLOGY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "distributed/cluster/topology/protocol/*.proto")
ms_protobuf_generate(DISTRIBUTED_CLUSTER_TOPOLOGY_SRCS DISTRIBUTED_CLUSTER_TOPOLOGY_HDRS
  ${DISTRIBUTED_CLUSTER_TOPOLOGY})
list(APPEND MINDSPORE_PROTO_LIST ${DISTRIBUTED_CLUSTER_TOPOLOGY_SRCS})

if(NOT ENABLE_SECURITY)
    file(GLOB_RECURSE PROFILER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "plugin/device/ascend/hal/profiler/memory_profiling.proto")
    ms_protobuf_generate_py(PROFILER_MEM_PROTO_PY PROFILER_MEM_PROTO_HDRS_PY PROFILER_MEM_PROTO_PYS
      ${PROFILER_PROTO_LIST})
    list(APPEND MINDSPORE_PROTO_LIST ${PROFILER_MEM_PROTO_PY})
endif()

if(ENABLE_DEBUGGER)
    # debugger: compile proto files
    include_directories("${CMAKE_BINARY_DIR}/debug/debugger")
    file(GLOB_RECURSE DEBUGGER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/debug_graph.proto")
    ms_protobuf_generate(DEBUGGER_PROTO_SRCS DEBUGGER_PROTO_HDRS ${DEBUGGER_PROTO_LIST})
    file(GLOB_RECURSE DEBUGGER_GRPC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/debug_grpc.proto")
    ms_grpc_generate(DEBUGGER_GRPC_SRCS DEBUGGER_GRPC_HDRS ${DEBUGGER_GRPC_LIST})
    file(GLOB_RECURSE DUMP_DATA_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/dump_data.proto")
    ms_protobuf_generate(DUMP_DATA_PROTO_SRCS DUMP_DATA_PROTO_HDRS ${DUMP_DATA_PROTO_LIST})
    list(APPEND MINDSPORE_PROTO_LIST ${DEBUGGER_PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${DEBUGGER_GRPC_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${DUMP_DATA_PROTO_SRCS})
endif()

if(ENABLE_DUMP_PROTO)
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB_RECURSE PROTO_PY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "utils/anf_ir.proto"
        "utils/summary.proto"
        "utils/lineage.proto"
        "utils/checkpoint.proto"
        "utils/print.proto"
        "utils/node_strategy.proto"
        "utils/profiling_parallel.proto"
    )
    ms_protobuf_generate_py(PY_SRCS PY_HDRS PY_PYS ${PROTO_PY})

    list(APPEND MINDSPORE_PROTO_LIST ${PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${PY_SRCS})
endif()

if(MODE_ASCEND_ALL)
    include_directories("${CMAKE_BINARY_DIR}/plugin/device/ascend/kernel/aicpu")
    file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "plugin/device/ascend/kernel/aicpu/proto/*.proto")
    ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN})

    file(GLOB_RECURSE PROTO_DUMP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "plugin/device/ascend/hal/device/dump/proto/*.proto")
    ms_protobuf_generate(DUMP_PROTOSRCS PROTOHDRS ${PROTO_DUMP})

    list(APPEND MINDSPORE_PROTO_LIST ${PROTOSRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${PREDICT_PROTOSRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${DUMP_PROTOSRCS})

    add_compile_definitions(ENABLE_D)
endif()

# core/mindir.proto
file(GLOB_RECURSE CORE_PROTO_IN ${CMAKE_SOURCE_DIR}/mindspore/core/proto/*.proto)
ms_protobuf_generate(CORE_PROTO_SRC CORE_PROTO_HDR ${CORE_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${CORE_PROTO_SRC})

include_directories("${CMAKE_BINARY_DIR}/runtime/graph_scheduler/actor/rpc")
file(GLOB_RECURSE RPC_PROTO RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "runtime/graph_scheduler/actor/rpc/*.proto")
ms_protobuf_generate(RPC_PROTO_SRCS RPC_PROTO_HDRS ${RPC_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${RPC_PROTO_SRCS})

if(MINDSPORE_PROTO_LIST)
    add_library(proto_input STATIC ${MINDSPORE_PROTO_LIST})
    if(NOT MSVC)
    set_target_properties(proto_input PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-array-bounds")
    endif()
endif()

## make sub objects
set(SUB_COMP
        transform/express_ir
        frontend/optimizer
        frontend/parallel
        frontend/operator
        pipeline/jit
        pipeline/pynative
        pybind_api
)

if(WIN32)
add_compile_definitions(BUILDING_ME_DLL)
endif()

foreach(_comp ${SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core)
    endif()
endforeach()

set_property(SOURCE ${SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore STATIC ${SUB_OBJECTS_SRC})

set(COMMON_SUB_COMP
        utils
        common
        common/debug
        common/expander
        )

foreach(_comp ${COMMON_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND COMMON_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE COMMON_DLL)
        endif()
    endif()
endforeach()

add_library(mindspore_common SHARED ${COMMON_SUB_OBJECTS_SRC})
target_link_libraries(mindspore_common PRIVATE mindspore_core proto_input mindspore::protobuf)
set_target_properties(mindspore_common PROPERTIES INSTALL_RPATH $ORIGIN)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_common PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(mindspore PUBLIC securec mindspore::flatbuffers)

if(NOT WIN32)
  target_link_libraries(mindspore PUBLIC dl)
endif()

set(BACKEND_SUB_COMP
        debug
        profiler
        ps
        distributed
        kernel
        common/mem_reuse
        backend/common/optimizer
        backend/common/pass
        backend/common/session
        backend/common/somas
        common/graph_kernel
        backend/graph_compiler
        backend/operator
        runtime/collective
        runtime/device
        runtime/graph_scheduler
        runtime/hardware
        runtime/pynative
        runtime/data_queue
        plugin/device/cpu/hal/device
        plugin/device/cpu/hal/hardware
        plugin/device/cpu/hal/profiler
        plugin/device/cpu/optimizer
        )

foreach(_comp ${BACKEND_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND BACKEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE BACKEND_DLL)
        endif()
    endif()
endforeach()

set(CPU_KERNEL_OBJECT_COUNT 0)
add_subdirectory(plugin/device/cpu/kernel)
foreach(number RANGE 1 ${CPU_KERNEL_OBJECT_COUNT})
    if(TARGET _mindspore_plugin_device_cpu_kernel_obj_${number})
        list(APPEND BACKEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_cpu_kernel_obj_${number}>)
        add_dependencies(_mindspore_plugin_device_cpu_kernel_obj_${number} proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_plugin_device_cpu_kernel_obj_${number} PRIVATE BACKEND_DLL)
        endif()
    endif()
endforeach()

if(ENABLE_TEST OR ENABLE_TESTCASES)
    include_directories(${CMAKE_BINARY_DIR})
    list(APPEND STUB_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ps/ps_core_stub.cc)
    add_library(stub_backend_obj OBJECT ${STUB_BACKEND_SOURCE})
    list(APPEND BACKEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:stub_backend_obj>)
endif()

set_property(SOURCE ${BACKEND_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    list(APPEND BACKEND_SUB_OBJECTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cxx_api/types.cc)
endif()
add_library(mindspore_backend SHARED ${BACKEND_SUB_OBJECTS_SRC})

if(MODE_ASCEND_ACL)
    add_library(mindspore_backend_static STATIC ${BACKEND_SUB_OBJECTS_SRC})
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(mindspore_backend PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(mindspore_backend PRIVATE mindspore_core mindspore_common proto_input mindspore::protobuf)
target_link_libraries(mindspore_backend PRIVATE securec)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_backend PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_backend PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_backend PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

if(ENABLE_CPU)
    target_link_libraries(mindspore_backend PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore_backend PRIVATE mindspore::ssl mindspore::crypto)
endif()

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(mindspore_backend PRIVATE mindspore::grpc++)
    else()
        target_link_libraries(mindspore_backend PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_backend PRIVATE mindspore::event mindspore::event_pthreads mindspore::event_openssl
            mindspore::event_core ps_cache)
elseif(ENABLE_CPU AND NOT WIN32)
    target_link_libraries(mindspore_backend PRIVATE mindspore::event mindspore::event_pthreads mindspore::event_openssl
            -Wl,--no-as-needed mindspore::event_core ps_cache)
endif()

if(MODE_ASCEND_ALL)
    target_link_libraries(mindspore PUBLIC -Wl,--start-group proto_input mindspore::protobuf -Wl,--end-group)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(MSVC)
        target_link_libraries(mindspore PUBLIC proto_input mindspore::protobuf mindspore::sentencepiece)
    else()
        target_link_libraries(mindspore PUBLIC -Wl,--start-group proto_input mindspore::protobuf
                mindspore::sentencepiece -Wl,--end-group)
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore PUBLIC -Wl proto_input mindspore::protobuf mindspore::sentencepiece -Wl)
else()
    target_link_libraries(mindspore PUBLIC -Wl,--start-group proto_input mindspore::protobuf -Wl,--end-group)
endif()

# set c_expression building
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set_property(SOURCE "pipeline/jit/init.cc" PROPERTY
            COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PIPELINE)
pybind11_add_module(_c_expression NO_EXTRAS "pipeline/jit/init.cc" NO_EXTRAS)

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(_c_expression PRIVATE -Wl,-init,mindspore_log_init)
    set(ORIGIN_PATH $ORIGIN)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(_c_expression PROPERTIES MACOSX_RPATH ON)
    set(ORIGIN_PATH @loader_path)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ORIGIN_PATH $ORIGIN)
else()
    MESSAGE(FATAL_ERROR "other platform: ${CMAKE_SYSTEM_NAME}")
endif()

if(ENABLE_D)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/ascend)
    add_subdirectory(plugin/device/ascend)
    enable_directory_when_only_build_plugins(plugin/device/ascend)
endif()

if(ENABLE_GPU AND GPU_BACKEND_CUDA)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/gpu)
    add_subdirectory(plugin/device/gpu)
    enable_directory_when_only_build_plugins(plugin/device/gpu)
endif()

if(ENABLE_GPU AND GPU_BACKEND_ROCM)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/amd)
    add_subdirectory(plugin/device/amd)
    enable_directory_when_only_build_plugins(plugin/device/amd)
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(mindspore PUBLIC mindspore::pybind11_module)
    if(NOT MSVC)
        target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore -Wl,--no-whole-archive mindspore_core
            mindspore_common mindspore_backend)
    else()
        target_link_libraries(_c_expression PRIVATE mindspore_core mindspore_common mindspore_backend mindspore)
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore PUBLIC proto_input mindspore::protobuf mindspore::eigen mindspore::json)
    target_link_libraries(_c_expression PRIVATE -Wl,-all_load mindspore proto_input -Wl,-noall_load mindspore_core
        mindspore_common mindspore_backend)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
else()
    if(ENABLE_CPU AND NOT WIN32)
        target_link_libraries(mindspore PUBLIC proto_input mindspore::protobuf mindspore::eigen mindspore::json)
    endif()
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore proto_input -Wl,--no-whole-archive
            mindspore_core mindspore_common mindspore_backend)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
endif()

set_target_properties(_c_expression PROPERTIES INSTALL_RPATH ${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin)
if(USE_GLOG)
    target_link_libraries(_c_expression PRIVATE mindspore::glog)
endif()

if(ENABLE_GPU)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        target_link_libraries(_c_expression PRIVATE mindspore::ssl mindspore::crypto)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
    set_target_properties(_c_expression PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
endif()

if(ENABLE_CPU)
    target_link_libraries(_c_expression PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()

if(ENABLE_MINDDATA)
    add_subdirectory(minddata/mindrecord)
    add_subdirectory(minddata/dataset)
endif()

if(ENABLE_TEST OR ENABLE_TESTCASES)
    include_directories(${CMAKE_BINARY_DIR})
    list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ge/ge_operator_stub.cc)
    list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/transform/util.cc)
    list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/pipeline/action_stub.cc)
    list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/cluster/cluster_stub.cc)
    list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/profiling/parallel_strategy_profiling_stub.cc)

    list(APPEND EXPRESSION_STUB_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ps/ps_core_stub.cc)

    add_library(stub_common STATIC ${STUB_COMMON_SOURCE})
    target_link_libraries(mindspore_common PUBLIC stub_common)

    add_library(expression_ STATIC ${EXPRESSION_STUB_SOURCE})
    target_link_libraries(_c_expression PUBLIC expression_)
endif()

if(NOT ENABLE_TESTCASES AND NOT (ENABLE_D OR ENABLE_CPU OR ENABLE_GPU))
    include_directories(${CMAKE_BINARY_DIR})
    list(APPEND EXPRESSION_STUB_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/cluster/cluster_stub.cc)
    list(APPEND EXPRESSION_STUB_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ps/ps_core_stub.cc)
    list(APPEND EXPRESSION_STUB_SOURCE
            ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/profiling/parallel_strategy_profiling_stub.cc)

    add_library(expression_ STATIC ${EXPRESSION_STUB_SOURCE})
    target_link_libraries(_c_expression PUBLIC expression_)
endif()
add_subdirectory(transform/graph_ir)
add_subdirectory(cxx_api)
enable_directory_when_only_build_plugins(cxx_api)
add_subdirectory(c_api)
