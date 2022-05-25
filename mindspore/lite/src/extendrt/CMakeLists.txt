set(MODEL_LOADER_FRAMEWORK_SRC
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/model_loader.cc
)

set(MSLITE_EXTEND_RUNTIME_SRC ${MSLITE_EXTEND_RUNTIME_SRC}
        ${MODEL_LOADER_FRAMEWORK_SRC})

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE)
    set(ENABLE_CPU on)
    add_compile_definitions(USE_GLOG)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
    remove_definitions(-DBUILD_LITE_INFERENCE)
    set(MINDIR_MODEL_SRC
            ${MINDIR_MODEL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/mindir_model.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/mindir_model_util.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/mindir_model_convertor.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/mindir_model_loader.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/kernel_mod_util.cc
            )

    set(MINDIR_KERNEL_SRC
            ${MINDIR_KERNEL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/inner_kernel.cc)

    set(MSLITE_EXTEND_RUNTIME_SRC ${MSLITE_EXTEND_RUNTIME_SRC}
            ${MINDIR_MODEL_SRC}
            ${MINDIR_KERNEL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/less_test_kernel_mod.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mindir_loader/mindir_model/kernel_mod_mock.cc)

    set(FBS_FILES
            ${CCSRC_DIR}/../schema/cipher.fbs
            ${CCSRC_DIR}/../schema/fl_job.fbs
            )
    ms_build_flatbuffers_lite(FBS_FILES ${CCSRC_DIR}/../schema/ generated_fbs_files ${CMAKE_BINARY_DIR}/schema "")

    include_directories("${CCSRC_DIR}/ps/core")
    file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CCSRC_DIR}/ps/core/protos/*.proto")
    ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
    list(APPEND MSLITE_PROTO_SRC ${COMM_PROTO_SRCS})

    if(NOT ENABLE_SECURITY)
        include_directories("${CCSRC_DIR}/profiler/device/ascend")
        file(GLOB_RECURSE PROFILER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${CCSRC_DIR}/profiler/device/ascend/memory_profiling.proto")
        # ms_protobuf_generate_py(PROFILER_MEM_PROTO_PY PROFILER_MEM_PROTO_HDRS_PY PROFILER_MEM_PROTO_PYS
        #   ${PROFILER_PROTO_LIST})
        ms_protobuf_generate(PROFILER_MEM_PROTO_SRC PROFILER_MEM_PROTO_HDRS ${PROFILER_PROTO_LIST})
        list(APPEND MSLITE_PROTO_SRC ${PROFILER_MEM_PROTO_SRC})
    endif()

    add_library(mindspore-lite-proto OBJECT ${MSLITE_PROTO_SRC})

    add_library(mindspore-extendrt SHARED ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt fbs_inner_src)
    add_dependencies(mindspore-extendrt generated_fbs_files)
    add_dependencies(mindspore-extendrt mindspore-lite-proto)

    add_subdirectory(${CCSRC_DIR}/transform/graph_ir graph_ir)
    add_subdirectory(${CCSRC_DIR}/backend/common/session common_session)
    add_subdirectory(cxx_api)

    add_subdirectory(${CCSRC_DIR}/utils mindspore_ccsrc_utils)
    add_subdirectory(${CCSRC_DIR}/runtime/device mindspore_ccsrc_runtime_device)
    add_subdirectory(${CCSRC_DIR}/runtime/pynative mindspore_ccsrc_runtime_pynative)
    add_subdirectory(${CCSRC_DIR}/backend/common/optimizer mindspore_ccsrc_backend_cmmon_optimizer)
    add_subdirectory(${CCSRC_DIR}/kernel mindspore_ccsrc_kernel)
    add_subdirectory(${CCSRC_DIR}/backend/common/somas mindspore_ccsrc_backend_cmmon_somas)
    add_subdirectory(${CCSRC_DIR}/plugin/device/cpu/hal/device mindspore_ccsrc_plugin_device_cpu_hal_device)
    add_subdirectory(${CCSRC_DIR}/common mindspore_ccsrc_common)
    add_subdirectory(${CCSRC_DIR}/common/mem_reuse mindspore_ccsrc_common_mem_reuse)
    add_subdirectory(${CCSRC_DIR}/plugin/device/cpu/kernel mindspore_ccsrc_plugin_device_cpu_kernel)
    add_subdirectory(${CCSRC_DIR}/pybind_api mindspore_ccsrc_pybind_api)
    add_subdirectory(${CCSRC_DIR}/ps mindspore_ccsrc_ps)

    target_link_libraries(mindspore-extendrt mindspore_shared_lib_obj)
    # target_link_libraries(mindspore-extendrt _mindspore_backend_common_session_obj _mindspore_transform_graph_ir_obj)
    # target_link_libraries(mindspore-extendrt _mindspore_backend_common_session_obj
    #                     _mindspore_backend_common_optimizer_obj _mindspore_runtime_device_obj
    #                     _mindspore_utils_obj _mindspore_kernel_obj _mindspore_backend_common_somas_obj
    #                     _mindspore_kernel_obj _mindspore_plugin_device_cpu_hal_device_obj
    #                     _mindspore_runtime_pynative_obj
    #                     _mindspore_common_obj _mindspore_common_mem_reuse_obj
    #                     _mindspore_plugin_device_cpu_kernel_obj
    #                     _mindspore_ps_obj ps_cache)
    target_link_libraries(mindspore-extendrt mindspore_core mindspore::protobuf mindspore::pybind11_module)
    # target_link_libraries(mindspore-extendrt )

    if(MSLITE_ENABLE_ACL)
        include_directories(${TOP_DIR}/graphengine/inc/external)
        add_subdirectory(kernel/ascend)
        link_directories(${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH} ${ASCEND_CANN_RUNTIME_PATH})
    endif()

    if(SUPPORT_CUDA)
        set(CUDA_PATH $ENV{CUDA_HOME})
        include_directories(${CCSRC_DIR}/plugin/device/gpu/kernel)
        set(ENABLE_GPU on)
        add_definitions(-DENABLE_GPU)
        set(CUDA_VERSION 11.1)
        include_directories(${CUDA_PATH})
        include_directories(${CUDA_PATH}/include)
        find_package(CUDA)
        add_subdirectory(kernel/cuda)
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_53 --expt-relaxed-constexpr)
        add_subdirectory(${CCSRC_DIR}/plugin/device/gpu/kernel/cuda_impl/cuda_ops cuda_ops)
        target_link_libraries(mindspore-extendrt cuda_lite_kernel_mid cuda_ops)
    endif()

    if(SUPPORT_TENSORRT)
        add_compile_definitions(GPU_TENSORRT)
        set(TENSORRT_PATH $ENV{TENSORRT_PATH})
        set(CUDA_PATH $ENV{CUDA_HOME})
        set(TENSORRT_LIB_PATH ${TENSORRT_PATH}/lib)
        set(CUDA_LIB_PATH ${CUDA_PATH}/lib64)
        include_directories(${TENSORRT_PATH}/include)
        include_directories(${CUDA_PATH}/include)
        add_subdirectory(delegate/tensorrt)
        target_link_libraries(mindspore-extendrt tensorrt_kernel_mid cuda_kernel_mid gpu_distribution_collective)
    else()
        set(TENSORRT_STUB
            ${CMAKE_CURRENT_SOURCE_DIR}/delegate/tensorrt/distribution/distribution_base.cc
        )
        add_library(tensorrt_stub OBJECT ${TENSORRT_STUB})
        target_link_libraries(mindspore-extendrt tensorrt_stub)
    endif()
else()
    add_library(mindspore-extendrt OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt fbs_inner_src)
endif()