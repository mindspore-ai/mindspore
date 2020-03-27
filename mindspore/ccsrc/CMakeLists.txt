include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(ENABLE_CPU)
    include(ExternalProject)
    add_compile_definitions(CPUSESSION)
    file(GLOB_RECURSE CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "device/cpu/*.cc"
    )
endif()

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
            )
    file(GLOB_RECURSE GPU_KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "kernel/gpu/*.cc"
            )
    list(APPEND CUDA_NVCC_FLAGS -arch=sm_53)
    list(REMOVE_ITEM GPU_SRC_LIST "device/gpu/blocking_queue.cc" "device/gpu/gpu_buffer_mgr.cc")
    add_library(gpu_queue SHARED "device/gpu/blocking_queue.cc" "device/gpu/gpu_buffer_mgr.cc")
    target_link_libraries(gpu_queue ${CMAKE_THREAD_LIBS_INIT} ${CUDA_PATH}/lib64/libcudart.so)


    file(GLOB_RECURSE MS_STEPS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "session/gpu_session.cc"
        )
    list(REMOVE_ITEM GPU_SRC_LIST "device/gpu/mpi/mpi_initializer.cc"
                                  "device/gpu/distribution/collective_wrapper.cc"
                                  "device/gpu/distribution/mpi_wrapper.cc"
                                  "device/gpu/distribution/nccl_wrapper.cc"
                                  )
    list(REMOVE_ITEM GPU_KERNEL_SRC_LIST "device/gpu/mpi/mpi_initializer.cc"
            "kernel/gpu/nccl/nccl_gpu_kernel.cc"
            )

    set(NVCC_TMP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-std=c++17" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    cuda_add_library(gpu_cuda_lib STATIC ${GPU_SRC_LIST})
    set(CMAKE_CXX_FLAGS ${NVCC_TMP_CMAKE_CXX_FLAGS})

    if(ENABLE_MPI)
      include(ExternalProject)

      file(GLOB_RECURSE GPU_NCCL_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
              "kernel/gpu/nccl/*.cc"
              )
      file(GLOB_RECURSE GPU_MPI_PYTHON_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
              "device/gpu/mpi/mpi_initializer.cc"
              )
      add_library(gpu_collective SHARED "device/gpu/distribution/collective_wrapper.cc"
                                        "device/gpu/distribution/mpi_wrapper.cc"
                                        "device/gpu/distribution/nccl_wrapper.cc"
                                        )
    endif()
endif()

include_directories("${CMAKE_BINARY_DIR}/predict/schema/inner")
file(GLOB_RECURSE FLATBUFFER_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "predict/schema/*.fbs")
set(FLATBUFFER_OU "${CMAKE_BINARY_DIR}/predict/schema/inner")
ms_build_flatbuffers("${FLATBUFFER_IN}" "${FLATBUFFER_IN}" GENERATED_OUTPUT_DIR "${FLATBUFFER_OU}")

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ir/*.cc"
        "ir/dtype/*.cc"
        "utils/*.cc"
        "common/*.cc"
        "parallel/*.cc"
        "pipeline/pipeline.cc"
        "pipeline/resource.cc"
        "pipeline/pass.cc"
        "pipeline/action.cc"
        "pipeline/validator.cc"
        "pipeline/remove_value_node_dup.cc"
        "pipeline/parse/*.cc"
        "pipeline/static_analysis/*.cc"
        "optimizer/*.cc"
        "debug/*.cc"
        "onnx/onnx_exporter.cc"
        "operator/*.cc"
        "transform/*.cc"
        "session/kernel_graph.cc"
        "utils/node_utils.cc"
        "session/session_basic.cc"
        "session/session_factory.cc"
        "session/anf_runtime_algorithm.cc"
        "vm/*.cc"
        "pynative/*.cc"
        "pybind_api/*.cc"
        "device/common/*.cc"
        "kernel/kernel_query.cc"
        "kernel/kernel_build_info.cc"
        "kernel/kash/*.cc"
        "device/kernel_info.cc"
        "device/kernel_runtime.cc"
        "device/kernel_runtime_manager.cc"
        "device/convert_tensor_utils.cc"
        "pre_activate/ascend/*.cc"
        "pre_activate/common/*.cc"
        "pre_activate/pass/*.cc"
        "pre_activate/gpu/*.cc"
        "pre_activate/mem_reuse/*.cc"
        "predict/predict.cc"
        "predict/generator/utils/ir_model_util.cc"
        "predict/converter/*.cc"
        "predict/converter/attr_utils/*.cc"
        "predict/converter/lite_model/*.cc"
        "predict/converter/lite_model/operations/*.cc"
        "kernel/common_utils.cc"
        "kernel/oplib/*.cc"
        "kernel/kash/*.cc"
        "device/gpu/distribution/collective_init.cc"
        )

file(GLOB_RECURSE MEM_REUSE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "pre_activate/mem_reuse/*.cc"
        )
if(NOT ENABLE_DUMP_E2E)
    list(REMOVE_ITEM MINDSPORE_SRC_LIST "debug/e2e_dump.cc")
endif()

file(COPY "${ms_onnx_INC}/onnx/onnx.proto" DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE ONNX_PROTO RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto")
message(“onnx proto path is : ${ONNX_PROTO}”)
ms_protobuf_generate(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${ONNX_PROTO_SRCS})

if(ENABLE_DUMP_PROTO)
    include_directories(${CMAKE_BINARY_DIR})
    file(GLOB_RECURSE PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "utils/node_strategy.proto"
            )
    ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

    file(GLOB_RECURSE PROTO_PY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "utils/anf_ir.proto"
            "utils/summary.proto"
            "utils/checkpoint.proto"
            )
    ms_protobuf_generate_py(PY_SRCS PY_HDRS PY_PYS ${PROTO_PY})

    list(APPEND MINDSPORE_PROTO_DUMP_LIST ${PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_DUMP_LIST ${PY_SRCS})
    list(APPEND MINDSPORE_SRC_LIST "debug/dump_proto.cc")
    list(APPEND MINDSPORE_SRC_LIST "parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
    add_compile_definitions(ENABLE_DUMP_PROTO)
endif()

if(ENABLE_D)
    include_directories("${CMAKE_BINARY_DIR}/kernel/aicpu")
    file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "kernel/aicpu/proto/*.proto"
            )
    ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN})

    include_directories("${CMAKE_BINARY_DIR}/predict/generator/ir")
    file(GLOB_RECURSE PROTO_INNER RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "predict/proto/*.proto"
            )
    ms_protobuf_generate(PREDICT_PROTOSRCS PREDICT_PROTOHDRS ${PROTO_INNER})

    file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "device/ascend/*.cc"
            "device/ascend/profiling/*.cc"
            "device/ascend/tasksink/*.cc"
            "kernel/akg/cce/*.cc"
            "device/kernel_adjust.cc"
            "kernel/kernel_fusion.cc"
            "kernel/tbe/*.cc"
            )
    list(APPEND MINDSPORE_SRC_LIST ${D_SRC_LIST})
    list(APPEND MINDSPORE_PROTO_AICPU_LIST ${PROTOSRCS})
    list(APPEND MINDSPORE_PROTO_PREDICT_LIST ${PREDICT_PROTOSRCS})

    file(GLOB_RECURSE MS_STEPS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "session/ascend_session.cc"
        )
    file(GLOB_RECURSE MS_TASKINFO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "device/ascend/tasksink/taskinfo/*.cc")
    file(GLOB_RECURSE MS_AICPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel/aicpu/*.cc"
        )
    file(GLOB_RECURSE MS_RT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel/mng/*.cc"
        )
    file(GLOB_RECURSE MS_HCCL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "kernel/hccl/*.cc"
        )
    file(GLOB_RECURSE MS_PREDICT_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "predict/generator/ir/*.cc"
        )
    add_compile_definitions(ENABLE_D)
endif()

file(GLOB_RECURSE MS_GVAR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "gvar/*.cc"
        )

add_library(mindspore_gvar SHARED ${MS_GVAR_SRC_LIST})
add_library(mindspore STATIC ${MINDSPORE_SRC_LIST})

if(ENABLE_D)
    list(APPEND MINDSPORE_PROTO_LIST ${MINDSPORE_PROTO_AICPU_LIST})
endif()
if(ENABLE_DUMP_PROTO)
    list(APPEND MINDSPORE_PROTO_LIST ${MINDSPORE_PROTO_DUMP_LIST})
endif()
list(APPEND MINDSPORE_PROTO_LIST ${MINDSPORE_PROTO_PREDICT_LIST})
if(MINDSPORE_PROTO_LIST)
    add_library(proto_input STATIC ${MINDSPORE_PROTO_LIST})
    set_target_properties(proto_input PROPERTIES COMPILE_FLAGS "-Wno-unused-variable")
    target_link_libraries(mindspore proto_input)
endif()

if(APPLE)
    set_target_properties(mindspore_gvar PROPERTIES MACOSX_RPATH ON)
endif()

link_directories(${CMAKE_SOURCE_DIR}/build/mindspore/graphengine)

if (ENABLE_GE)
    if(ENABLE_TRAIN)
        target_link_libraries(mindspore graph ge_client_train)
    else()
        target_link_libraries(mindspore graph ge_client)
    endif()
    target_link_libraries(mindspore tsdclient)
else()
    add_compile_definitions(NO_GE_CLIENT)
    target_link_libraries(mindspore graph)
endif()

if(ENABLE_D)
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
    else()
        MESSAGE("use system default lib")
        set(D_LIB_PATH "/usr/local/HiAI/runtime/lib64/")
    endif()

    MESSAGE("USE DAV LIB PATH: ${D_LIB_PATH}")
    find_library(HCCL hccl ${D_LIB_PATH})
    find_library(CCE_LIB cce ${D_LIB_PATH})
    find_library(RUNTIME_LIB runtime ${D_LIB_PATH})
    find_library(TSDCLIENT tsdclient ${D_LIB_PATH})
    find_library(PROFILING msprof ${D_LIB_PATH})
    target_link_libraries(mindspore ge_runtime ${CCE_LIB} ${RUNTIME_LIB} ${TSDCLIENT} ${PROFILING} ${HCCL} ${TSDCLIENT})
endif()

target_link_libraries(mindspore securec)
target_link_libraries(mindspore dl)
target_link_libraries(mindspore mindspore::flatbuffers)
# link protobuf
if (ENABLE_D)
    target_link_libraries(mindspore protobuf::libprotobuf)
endif()

# set c_expression building
set(PYTHON_MODULE_SOURCE
        pipeline/init.cc
        kernel/oplib/oplib.cc
        kernel/akg/akgkernelbuild.cc
        kernel/akg/akg_kernel_attrs_process.cc
    ${MS_STEPS_SRC_LIST} ${MS_CCE_SRC_LIST} ${MS_AICPU_SRC_LIST} ${MS_TASKINFO_LIST} ${MS_RT_SRC_LIST}
    ${GPU_NCCL_LIST} ${MS_HCCL_SRC_LIST} ${MS_PREDICT_SRC_LIST} ${CPU_SRC_LIST} ${MEM_REUSE_SRC_LIST} ${GPU_KERNEL_SRC_LIST})

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
pybind11_add_module(_c_expression ${PYTHON_MODULE_SOURCE})
target_link_options(_c_expression PRIVATE -Wl,-init,mindspore_log_init)

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(ORIGIN_PATH $ORIGIN)
elseif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(_c_expression PROPERTIES MACOSX_RPATH ON)
    set(ORIGIN_PATH @loader_path)
else ()
    MESSAGE(FATAL_ERROR "other platform: ${CMAKE_SYSTEM_NAME}")
endif ()


set(ORIGIN_PATH ${ORIGIN_PATH}/lib)
set_target_properties(_c_expression PROPERTIES INSTALL_RPATH ${ORIGIN_PATH})
target_link_libraries(_c_expression PRIVATE
        mindspore::pybind11_module
        mindspore
        mindspore_gvar
        )

if(ENABLE_GPU)
	execute_process(COMMAND bash ${CMAKE_SOURCE_DIR}/third_party/apply_patches.sh 
		${CMAKE_BINARY_DIR} 
		${dlpack_DIRPATH} 
		${dmlc_core_DIRPATH} 
		${rang_DIRPATH} 
		${incubator_tvm_gpu_DIRPATH})
    set(TVM_DIR "${CMAKE_BINARY_DIR}/incubator-tvm")
    # Utility functions
    include(${TVM_DIR}/cmake/util/Util.cmake)
    include(${TVM_DIR}/cmake/util/FindCUDA.cmake)

    # include directories
    include_directories(AFTER "${TVM_DIR}/include")
    include_directories(AFTER "${TVM_DIR}/src")
    include_directories(AFTER "${TVM_DIR}")
    include_directories(AFTER "${TVM_DIR}/src/schedule")

    include_directories(AFTER "${TVM_DIR}/3rdparty/dmlc-core/include")
    include_directories(AFTER "${TVM_DIR}/3rdparty/dlpack/include")
    include_directories(AFTER "${TVM_DIR}/3rdparty/compiler-rt")
    include_directories(AFTER "${TVM_DIR}/3rdparty/rang/include")

    # lib contain dlopen and dlclose
    set(TVM_RUNTIME_LINKER_LIBS ${CMAKE_DL_LIBS})

    # add source group
    file(GLOB_RECURSE GROUP_SOURCE "${TVM_DIR}/src/*.cc" "src/*.cc")
    file(GLOB_RECURSE GROUP_INCLUDE "${TVM_DIR}/src/*.h"
        "${TVM_DIR}/include/*.h" "src/*.h" "include/*.h")
    assign_source_group("Source" ${GROUP_SOURCE})
    assign_source_group("Include" ${GROUP_INCLUDE})

    file(GLOB COMPILER_SRCS
        ${TVM_DIR}/src/api/*.cc
        ${TVM_DIR}/src/arithmetic/*.cc
        ${TVM_DIR}/src/autotvm/*.cc
        ${TVM_DIR}/src/codegen/*.cc
        ${TVM_DIR}/src/lang/*.cc
        ${TVM_DIR}/src/pass/*.cc
        ${TVM_DIR}/src/op/*.cc
        ${TVM_DIR}/src/node/*.cc
        ${TVM_DIR}/src/schedule/*.cc
        ${TVM_DIR}/src/runtime/*.cc
        ${TVM_DIR}/src/runtime/cce/*.cc
        ${TVM_DIR}/src/runtime/vm/*.cc
        ${TVM_DIR}/src/runtime/vm/profiler/*.cc
        ${TVM_DIR}/src/codegen/stackvm/*.cc)

    file(GLOB_RECURSE RELAY_SRCS ${TVM_DIR}/src/relay/*.cc)
    list(APPEND COMPILER_SRCS ${RELAY_SRCS})

    file(GLOB DATATYPE_SRCS ${TVM_DIR}/src/codegen/datatype/*.cc)
    list(APPEND COMPILER_SRCS ${DATATYPE_SRCS})

    file(GLOB COMPILER_VERILOG_SRCS ${TVM_DIR}/src/codegen/verilog/*.cc)
    list(APPEND COMPILER_SRCS ${COMPILER_VERILOG_SRCS})

    file(GLOB TOPI_SRCS ${TVM_DIR}/topi/src/*.cc)

    file(GLOB RUNTIME_SRCS
        ${TVM_DIR}/src/runtime/*.cc
        ${TVM_DIR}/src/runtime/cce/*.cc
        ${TVM_DIR}/src/runtime/vm/*.cc
        ${TVM_DIR}/src/runtime/stub/*.cc
        ${TVM_DIR}/src/runtime/stackvm/*.cc)


    file(GLOB COMPILER_OFF_SRCS
        ${TVM_DIR}/src/codegen/opt/build_*_off.cc)
    set(USE_CUDA "OFF")
    if(ENABLE_GPU)
        list(REMOVE_ITEM COMPILER_OFF_SRCS
            ${TVM_DIR}/src/codegen/opt/build_cuda_off.cc)
        set(USE_CUDA "ON")
    endif()
    list(APPEND COMPILER_SRCS ${COMPILER_OFF_SRCS})
    # Module rules
    include(${TVM_DIR}/cmake/modules/CUDA.cmake)

    set(CMAKE_C_FLAGS_AKG  -pipe -Wall -fPIC -fstack-protector-all)
    set(CMAKE_C_FLAGS_AKG ${CMAKE_C_FLAGS_AKG} -Wl,-z,relro,-z,now,-z,noexecstack)

    set(CMAKE_CXX_FLAGS_AKG  -std=c++11 -pipe -Wall -fPIC -fstack-protector-all)
    set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -Wl,-z,relro,-z,now,-z,noexecstack)

    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        message("-- Build in Debug mode")
        set(CMAKE_C_FLAGS_AKG ${CMAKE_C_FLAGS_AKG} -O0 -g -rdynamic)
        set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -O0 -g -rdynamic)
    else()
        message("-- Build in Release mode")
        set(CMAKE_C_FLAGS_AKG ${CMAKE_C_FLAGS_AKG} -O2 -Werror)
        set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -O2 -Werror)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION
                                               VERSION_GREATER 7.0)
        set(CMAKE_CXX_FLAGS_AKG ${CMAKE_CXX_FLAGS_AKG} -faligned-new)
    endif()

    add_library(akg OBJECT ${COMPILER_SRCS} ${RUNTIME_SRCS} ${TOPI_SRCS})

    target_link_libraries(akg ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
    target_compile_options(akg PRIVATE
        $<$<COMPILE_LANGUAGE:C>:${CMAKE_C_FLAGS_AKG}>
        $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CXX_FLAGS_AKG}>)
    target_include_directories(akg PRIVATE "${TVM_DIR}/topi/include")

    add_dependencies(_c_expression akg)
    target_link_libraries(_c_expression PRIVATE akg)
endif()

if(ENABLE_DUMP_PROTO)
    target_link_libraries(_c_expression PRIVATE protobuf::libprotobuf)
endif()

if(ENABLE_GPU)
    message("add gpu lib to c_expression")
    target_link_libraries(_c_expression PRIVATE
                          gpu_cuda_lib
                          gpu_queue
                          cublas
                          ${CUDNN_PATH}/lib64/libcudnn.so
                          ${CUDA_PATH}/lib64/libcudart.so
                          ${CUDA_PATH}/lib64/stubs/libcuda.so)
    if(ENABLE_MPI)
        pybind11_add_module(_ms_mpi ${GPU_MPI_PYTHON_LIST})
        target_link_libraries(_ms_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)
        target_link_libraries(gpu_collective PRIVATE mindspore::ompi mindspore::nccl)
    endif()
endif()


if(ENABLE_CPU)
    target_link_libraries(_c_expression PRIVATE mindspore::dnnl mindspore::mkldnn)
endif()

if(ENABLE_MINDDATA)
    add_subdirectory(mindrecord)
    add_subdirectory(dataset)
endif()
set(MS_PACK_PATH ${CMAKE_SOURCE_DIR}/build/package/mindspore/)
set(MS_LIB_PATH ${CMAKE_SOURCE_DIR}/build/package/mindspore/lib/)

add_custom_target(add_ms_lib ALL
    COMMAND mkdir -pv ${MS_LIB_PATH}
    COMMAND cp ${MS_CCSRC_BUILD_PATH}/_c_expression* ${MS_PACK_PATH}
    COMMAND cp ${MS_CCSRC_BUILD_PATH}/libmindspore_gvar.so ${MS_LIB_PATH}
)
add_dependencies(add_ms_lib _c_expression)

if (NOT ENABLE_GE)
    if (ENABLE_D)
        add_custom_target(add_ge_lib ALL
                COMMAND cp ${MS_CCSRC_BUILD_PATH}/../../graphengine/src/common/graph/libgraph.so ${MS_LIB_PATH}
                COMMAND cp ${MS_CCSRC_BUILD_PATH}/../../graphengine/src/ge/common/libge_common.so ${MS_LIB_PATH}
                COMMAND cp ${MS_CCSRC_BUILD_PATH}/../../graphengine/src/ge/ge_runtime/libge_runtime.so ${MS_LIB_PATH}
                COMMAND cp /usr/local/HiAI/driver/lib64/libslog.so ${MS_LIB_PATH}
                COMMAND cp /usr/local/HiAI/driver/lib64/libc_sec.so ${MS_LIB_PATH}
        )
        add_dependencies(add_ge_lib add_ms_lib)
        add_dependencies(add_ge_lib graph)
        add_dependencies(add_ge_lib ge_runtime)
    else()
        add_custom_target(add_ge_lib ALL
                COMMAND cp ${MS_CCSRC_BUILD_PATH}/../../graphengine/src/common/graph/libgraph.so  ${MS_LIB_PATH}
                COMMAND cp ${CMAKE_SOURCE_DIR}/graphengine/third_party/prebuild/${CMAKE_HOST_SYSTEM_PROCESSOR}/libslog.so ${MS_LIB_PATH}
                COMMAND cp ${CMAKE_SOURCE_DIR}/graphengine/third_party/prebuild/${CMAKE_HOST_SYSTEM_PROCESSOR}/libc_sec.so ${MS_LIB_PATH}
        )
        add_dependencies(add_ge_lib add_ms_lib)
        add_dependencies(add_ge_lib graph)
    endif()
endif()

if (ENABLE_GPU)
    if (ENABLE_MPI)
        add_custom_target(add_mpi_lib ALL
            COMMAND cp ${MS_CCSRC_BUILD_PATH}/_ms_mpi* ${MS_PACK_PATH}
        )
        add_dependencies(add_mpi_lib _ms_mpi)
        add_custom_target(add_gpu_collective_lib ALL
            COMMAND mkdir -pv ${MS_LIB_PATH}
            COMMAND cp ${MS_CCSRC_BUILD_PATH}/libgpu_collective* ${MS_LIB_PATH}
            )
        add_dependencies(add_gpu_collective_lib gpu_collective)
    endif()   
    add_custom_target(add_gpu_queue_lib ALL
        COMMAND cp ${MS_CCSRC_BUILD_PATH}/libgpu_queue* ${MS_LIB_PATH}
            )
    add_dependencies(add_gpu_queue_lib add_ms_lib)
endif()

if (ENABLE_CPU)
    add_custom_target(add_cpu_lib ALL
            COMMAND cp ${mkl_dnn_LIBPATH}/libdnnl.so.1.1 ${MS_LIB_PATH}/libdnnl.so.1
            )
    add_dependencies(add_cpu_lib add_ms_lib)
endif()

if (ENABLE_MINDDATA)
    add_custom_target(add_minddata_lib ALL
        COMMAND cp ${MS_CCSRC_BUILD_PATH}/dataset/*.so ${MS_PACK_PATH}
        COMMAND cp ${MS_CCSRC_BUILD_PATH}/mindrecord/*.so ${MS_PACK_PATH}
        COMMAND cp ${opencv_LIBPATH}/libopencv_core.so.4.2.0 ${MS_LIB_PATH}/libopencv_core.so.4.2
        COMMAND cp ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.2.0 ${MS_LIB_PATH}/libopencv_imgcodecs.so.4.2
        COMMAND cp ${opencv_LIBPATH}/libopencv_imgproc.so.4.2.0 ${MS_LIB_PATH}/libopencv_imgproc.so.4.2
    )
    add_dependencies(add_minddata_lib add_ms_lib)
    add_dependencies(add_minddata_lib _c_mindrecord)
    add_dependencies(add_minddata_lib _c_dataengine)

    add_dependencies(_c_mindrecord mindspore)
    add_dependencies(_c_dataengine mindspore)
endif()

if (USE_GLOG)
    target_link_libraries(_c_expression PRIVATE mindspore::glog)
    add_custom_target(add_glog_lib ALL
        COMMAND cp ${glog_LIBPATH}/libglog*.so.0 ${MS_LIB_PATH}
            )
    add_dependencies(add_glog_lib add_ms_lib)
endif()
