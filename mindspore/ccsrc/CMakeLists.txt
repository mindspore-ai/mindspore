## common setting
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
link_directories(${CMAKE_SOURCE_DIR}/build/mindspore/graphengine)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
    add_compile_definitions(BUILDING_DLL)
endif()

if (ENABLE_MPI)
    add_compile_definitions(ENABLE_MPI)
endif ()

if(ENABLE_GPU)
    find_package(CUDA REQUIRED)
    find_package(Threads)
    if(${CUDA_VERSION} VERSION_LESS ${MS_REQUIRE_CUDA_VERSION})
      message(FATAL_ERROR "The minimum CUDA version ${MS_REQUIRE_CUDA_VERSION} is required, but only CUDA ${CUDA_VERSION} found.")
    endif()
    enable_language(CUDA)
    if(NOT CUDA_PATH OR CUDA_PATH STREQUAL "")
        if(DEFINED ENV{CUDA_HOME})
            set(CUDA_PATH $ENV{CUDA_HOME})
        else()
            set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
        endif()
    endif()
    if(NOT CUDNN_PATH OR CUDNN_PATH STREQUAL "")
        set(CUDNN_PATH ${CUDA_PATH})
    endif()
    message("CUDA_PATH: ${CUDA_PATH}")
    message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message("CUDNN_PATH: ${CUDNN_PATH}")
    include_directories(${CUDNN_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS})

    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "device/gpu/*.cc"
            "device/gpu/*.cu"
            "kernel/gpu/*.cu"
            "kernel/akg/gpu/*.cc"
            "kernel/akg/akg_kernel_build.cc"
            "kernel/akg/akg_kernel_attrs_process.cc"
            )

    list(APPEND CUDA_NVCC_FLAGS -arch=sm_53)
    list(REMOVE_ITEM GPU_SRC_LIST "device/gpu/blocking_queue.cc" "device/gpu/gpu_buffer_mgr.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "device/gpu/mpi/mpi_initializer.cc"
                                  "device/gpu/distribution/collective_wrapper.cc"
                                  "device/gpu/distribution/mpi_wrapper.cc"
                                  "device/gpu/distribution/nccl_wrapper.cc"
                                  )

    set(NVCC_TMP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-std=c++17" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set_property(SOURCE ${GPU_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    cuda_add_library(gpu_cuda_lib STATIC ${GPU_SRC_LIST})
    set(CMAKE_CXX_FLAGS ${NVCC_TMP_CMAKE_CXX_FLAGS})
endif ()

## make flatuffer files
include_directories("${CMAKE_BINARY_DIR}/predict/schema/inner")
file(GLOB_RECURSE FLATBUFFER_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "predict/schema/*.fbs")
set(FLATBUFFER_OU "${CMAKE_BINARY_DIR}/predict/schema/inner")
ms_build_flatbuffers("${FLATBUFFER_IN}" "${FLATBUFFER_IN}" flat_input "${FLATBUFFER_OU}")

## make protobuf files
file(COPY "${ms_onnx_INC}/onnx/onnx.proto" DESTINATION ${CMAKE_BINARY_DIR}/proto)
file(GLOB ONNX_PROTO "" ${CMAKE_BINARY_DIR}/proto/onnx.proto)
message("onnx proto path is :" ${ONNX_PROTO})
ms_protobuf_generate(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${ONNX_PROTO_SRCS})

if (ENABLE_DUMP_PROTO)
    include_directories(${CMAKE_BINARY_DIR})
    file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utils/node_strategy.proto")
    ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

    file(GLOB_RECURSE PROTO_PY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "utils/anf_ir.proto"
        "utils/summary.proto"
        "utils/lineage.proto"
        "utils/checkpoint.proto"
        "utils/print.proto"
    )
    ms_protobuf_generate_py(PY_SRCS PY_HDRS PY_PYS ${PROTO_PY})

    list(APPEND MINDSPORE_PROTO_LIST ${PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${PY_SRCS})
endif ()

if (ENABLE_D)
    include_directories("${CMAKE_BINARY_DIR}/kernel/aicpu")
    include_directories("${CMAKE_BINARY_DIR}/predict/generator/ir")
    file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "kernel/aicpu/proto/*.proto")
    ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN})
    
    file(GLOB_RECURSE PROTO_INNER RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "predict/proto/*.proto")
    ms_protobuf_generate(PREDICT_PROTOSRCS PREDICT_PROTOHDRS ${PROTO_INNER})

    list(APPEND MINDSPORE_PROTO_LIST ${PROTOSRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${PREDICT_PROTOSRCS})

    add_compile_definitions(ENABLE_D)
endif ()

if (MINDSPORE_PROTO_LIST)
    add_library(proto_input STATIC ${MINDSPORE_PROTO_LIST})
    set_target_properties(proto_input PROPERTIES COMPILE_FLAGS "-Wno-unused-variable")
endif()

## make sub objects
set(SUB_COMP 
    transform pre_activate parallel pipeline device kernel common debug gvar ir onnx operator optimizer predict
    pybind_api pynative session utils vm
)

foreach (_comp ${SUB_COMP})
    add_subdirectory(${_comp})
    if (TARGET _mindspore_${_comp}_obj)
        list(APPEND SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${_comp}_obj>)
        add_dependencies(_mindspore_${_comp}_obj proto_input flat_input)
    endif ()
endforeach ()

set_property(SOURCE ${SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore STATIC ${SUB_OBJECTS_SRC})
target_link_libraries(mindspore proto_input)
if (ENABLE_CPU AND ENABLE_MPI)
    target_link_libraries(mindspore securec mindspore::flatbuffers mindspore::ompi)
else ()
    target_link_libraries(mindspore securec mindspore::flatbuffers)
endif ()
if (NOT WIN32)
  target_link_libraries(mindspore dl)
endif()

if (ENABLE_GE)
    if(ENABLE_TRAIN)
        target_link_libraries(mindspore ge_runner hccl)
    else ()
        target_link_libraries(mindspore ge_client)
    endif ()
    target_link_libraries(mindspore graph tsdclient)
endif()

if (ENABLE_D)
    if (DEFINED ENV{D_LINK_PATH})
        if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
            MESSAGE("system processor matches aarch64")
            set(D_LIB_PATH $ENV{D_LINK_PATH}/aarch64)
        elseif (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
            MESSAGE("system processor matches x86_64")
            set(D_LIB_PATH $ENV{D_LINK_PATH}/x86_64)
        else ()
            MESSAGE("system ${CMAKE_HOST_SYSTEM_PROCESSOR} not support")
        endif()
    else ()
        MESSAGE("use system default lib")
        if (DEFINED ENV{ASCEND_CUSTOM_PATH})
            set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
        else ()
            set(ASCEND_PATH /usr/local/Ascend)
        endif ()
        set(ASCEND_DRIVER_PATH ${ASCEND_PATH}/driver/lib64/common)
        set(ASCEND_DRIVER_BACK_PATH ${ASCEND_PATH}/driver/lib64/driver)
        set(ASCEND_RUNTIME_PATH ${ASCEND_PATH}/fwkacllib/lib64)
    endif()

    MESSAGE("USE DAV LIB PATH: ${ASCEND_PATH}")
    find_library(HCCL hccl ${ASCEND_RUNTIME_PATH})
    find_library(CCE_LIB cce ${ASCEND_RUNTIME_PATH})
    find_library(RUNTIME_LIB runtime ${ASCEND_RUNTIME_PATH})
    find_library(TSDCLIENT tsdclient HINTS ${ASCEND_RUNTIME_PATH} ${ASCEND_DRIVER_BACK_PATH})
    find_library(PROFILING msprof ${ASCEND_DRIVER_PATH})
    target_link_libraries(mindspore ge_runtime ${CCE_LIB} ${RUNTIME_LIB} ${TSDCLIENT} ${PROFILING} ${HCCL} ${TSDCLIENT})
endif()

# link protobuf
if (ENABLE_D)
    target_link_libraries(mindspore mindspore::protobuf)
endif()

# set c_expression building
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set_property(SOURCE "pipeline/init.cc" PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PIPELINE)
pybind11_add_module(_c_expression "pipeline/init.cc")

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(_c_expression PRIVATE -Wl,-init,mindspore_log_init)
    set(ORIGIN_PATH $ORIGIN)
elseif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(_c_expression PROPERTIES MACOSX_RPATH ON)
    set(ORIGIN_PATH @loader_path)
elseif (CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ORIGIN_PATH $ORIGIN)
else ()
    MESSAGE(FATAL_ERROR "other platform: ${CMAKE_SYSTEM_NAME}")
endif ()

set(ORIGIN_PATH ${ORIGIN_PATH}/lib)
set_target_properties(_c_expression PROPERTIES INSTALL_RPATH ${ORIGIN_PATH})

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(mindspore mindspore::pybind11_module)
    target_link_libraries(mindspore mindspore_gvar)
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore -Wl,--no-whole-archive)
else ()
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore -Wl,--no-whole-archive)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
    target_link_libraries(_c_expression PRIVATE mindspore_gvar)
endif ()

if (USE_GLOG)
    target_link_libraries(_c_expression PRIVATE mindspore::glog)
endif ()

if (ENABLE_DUMP_PROTO)
    target_link_libraries(_c_expression PRIVATE mindspore::protobuf)
endif ()

if (ENABLE_GPU)
    message("add gpu lib to c_expression")
    target_link_libraries(_c_expression PRIVATE gpu_cuda_lib gpu_queue cublas
                          ${CUDA_PATH}/lib64/libcurand.so
                          ${CUDNN_PATH}/lib64/libcudnn.so
                          ${CUDA_PATH}/lib64/libcudart.so
                          ${CUDA_PATH}/lib64/stubs/libcuda.so)
endif ()

if (ENABLE_CPU)
    target_link_libraries(_c_expression PRIVATE mindspore::dnnl mindspore::mkldnn)
endif ()

if (ENABLE_MINDDATA)
    add_subdirectory(mindrecord)
    add_subdirectory(dataset)
endif ()

# build inference
set(LOAD_ONNX_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/load_onnx/anf_converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/load_onnx/anf_model_parser.cc
        )
add_library(inference SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/session/session.cc
        ${LOAD_ONNX_SRC}
        )
target_link_libraries(inference PRIVATE ${PYTHON_LIB} ${SECUREC_LIBRARY}
        -Wl,--whole-archive mindspore -Wl,--no-whole-archive mindspore_gvar mindspore::protobuf)

if (ENABLE_CPU)
    target_link_libraries(inference PRIVATE mindspore::dnnl mindspore::mkldnn)
endif ()

if (USE_GLOG)
    target_link_libraries(inference PRIVATE mindspore::glog)
else()
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_options(inference PRIVATE -Wl,-init,mindspore_log_init)
    elseif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set_target_properties(inference PROPERTIES MACOSX_RPATH ON)
    endif ()
endif()
