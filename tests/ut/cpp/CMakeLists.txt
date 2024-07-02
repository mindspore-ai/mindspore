message("build ut testcases...")

string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")
add_compile_definitions(ENABLE_DUMP_IR)
if(NOT ENABLE_GLIBCXX)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

#add python lib and include for all ut executables;
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MS_CCSRC_PATH})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/pass/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/cpu/kernel)

include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${MS_CCSRC_BUILD_PATH})

add_definitions(-D ENABLE_MINDDATA)
if(ENABLE_FFMPEG)
    add_definitions(-D ENABLE_FFMPEG)
endif()

link_directories(${MS_CCSRC_BUILD_PATH}/minddata/dataset)
link_directories(${MS_CCSRC_BUILD_PATH}/minddata/mindrecord)

# fetch ut test files
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image)

file(GLOB_RECURSE UT_CORE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./core/abstract/*.cc ./core/utils/*.cc
        ./ir/dtype/*.cc ./ir/*.cc ./mindapi/*.cc ./mindir/*.cc ./ops/*.cc ./ops/view/*.cc ./base/*.cc)
file(GLOB_RECURSE UT_MINDDATA_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./dataset/*.cc ./mindrecord/*.cc)
file(GLOB_RECURSE UT_MINDDATA_COMMON_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./dataset/common/*.cc)
file(GLOB_RECURSE UT_API_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./cxx_api/*.cc)
file(GLOB_RECURSE UT_FRONTEND_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./operator/*.cc ./optimizer/*.cc
        ./parallel/*.cc ./pipeline/*.cc ./pynative/*.cc ./func_graph_builder/*.cc)
file(GLOB_RECURSE UT_OLD_BACKEND_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./device/*.cc ./kernel/*.cc
        ./pre_activate/common/*.cc  ./session/*.cc ./runtime/device/move_to_test.cc
        ./transform/*.cc ./vm/*.cc ./runtime/graph_scheduler/*.cc ./runtime/device/gsm/*.cc)
file(GLOB_RECURSE UT_BACKEND_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./backend/*.cc ./runtime/graph_scheduler/*.cc)
file(GLOB_RECURSE UT_SYMBOL_ENGINE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./symbol_engine/*.cc)
file(GLOB_RECURSE UT_GRAPH_KERNEL_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./graph_kernel/*.cc)
# plugin/device/cpu/hal/test_ms_collective_topo.cc will also open 127.0.0.1:8090
file(GLOB_RECURSE UT_PS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./ps/*.cc ./plugin/device/cpu/hal/*.cc)
file(GLOB_RECURSE UT_CCSRC_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./ccsrc/*.cc)
file(GLOB_RECURSE UT_OTHERS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./debug/*.cc ./side_effect/*.cc ./utils/*.cc
        ./place/*.cc)
file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./stub/*.cc ./common/*.cc ./mock/*.cc)

if(NOT ENABLE_CPU OR WIN32 OR APPLE)
    file(GLOB_RECURSE UT_DISTRIBUTED_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ./distributed/persistent/*.cc
            ./distributed/rpc/tcp/*.cc
            ./distributed/cluster/*.cc
            ./distributed/cluster/topology/*.cc
            ./distributed/recovery/*.cc
            ./distributed/embedding_cache/*.cc)
    if(${ENABLE_RDMA} STREQUAL "ON")
        include_directories(/usr/include/umdk)
        file(GLOB_RECURSE UT_DISTRIBUTED_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./distributed/rpc/rdma/*.cc)
    endif()
    list(APPEND UT_PS_SRCS ${UT_DISTRIBUTED_SRCS})
endif()

# split minddata
list(REMOVE_ITEM UT_MINDDATA_SRCS ${UT_MINDDATA_COMMON_SRCS})
list(LENGTH UT_MINDDATA_SRCS UT_MINDDATA_SRCS_LENS)
math(EXPR UT_MINDDATA0_SRCS_LENS "(${UT_MINDDATA_SRCS_LENS} + 1) / 2")

list(SUBLIST UT_MINDDATA_SRCS 0 ${UT_MINDDATA0_SRCS_LENS} UT_MINDDATA0_SRCS)
math(EXPR UT_MINDDATA1_SRCS_LENS "${UT_MINDDATA_SRCS_LENS} - ${UT_MINDDATA0_SRCS_LENS}")
list(SUBLIST UT_MINDDATA_SRCS ${UT_MINDDATA0_SRCS_LENS} ${UT_MINDDATA1_SRCS_LENS} UT_MINDDATA1_SRCS)
list(APPEND UT_MINDDATA0_SRCS ${UT_MINDDATA_COMMON_SRCS})
list(APPEND UT_MINDDATA1_SRCS ${UT_MINDDATA_COMMON_SRCS})

set(ALL_UT_COMPS CORE MINDDATA0 MINDDATA1 API FRONTEND OLD_BACKEND BACKEND PS CCSRC OTHERS SYMBOL_ENGINE GRAPH_KERNEL)

set(REPEATED_DEFINED_FILE stub/ps/ps_core_stub.cc)
list(REMOVE_ITEM UT_SRCS ${REPEATED_DEFINED_FILE})

find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    find_package(Python3 COMPONENTS NumPy Development)
    if(Python3_NumPy_FOUND)
        include_directories(${Python3_NumPy_INCLUDE_DIRS})
    endif()
endif()

file(GLOB_RECURSE EXTEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        # dont remove the 4 lines above
        "../../../mindspore/ccsrc/pybind_api/*.cc"
        "../../../mindspore/ccsrc/frontend/optimizer/*.cc"
        "../../../mindspore/ccsrc/frontend/parallel/*.cc"
        "../../../mindspore/ccsrc/frontend/operator/*.cc"
        "../../../mindspore/ccsrc/frontend/expander/*.cc"
        )
list(REMOVE_ITEM EXTEND_SRC_LIST
        "../../../mindspore/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/debug/data_dump/dump_json_parser.cc"
        "../../../mindspore/ccsrc/debug/common.cc"
        "../../../mindspore/ccsrc/debug/utils.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_scheduler.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_offload_strategy.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_info.cc"
        "../../../mindspore/ccsrc/runtime/device/launch_kernel.cc"
        "../../../mindspore/ccsrc/runtime/graph_scheduler/*.cc"
        "../../../mindspore/ccsrc/runtime/pipeline/*.cc"
        "../../../mindspore/ccsrc/runtime/pipeline/task/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/profiling/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_launch_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_launch_mul.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_launch_transdata.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/launch_transdata.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_select_graph_kernel.cc"
        "../../../mindspore/ccsrc/runtime/device/convert_tensor_utils.cc"
        "../../../mindspore/ccsrc/runtime/device/context_extends.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_event.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/kernel_build_ascend.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_device_synchronizer.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_runtime_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_gmem_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_device_address.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_stream_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_memory_pool.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_vmm_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/lic_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/tasksink/rtmodel_zero_copy.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_communication_group.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_context.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_res_manager.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_kernel_executor.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_executor.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/common/ascend_utils.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_somas.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_optimization.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/hardware/ms_collective_topo.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/hardware/cpu_memory_pool.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/device/cpu_device_address.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/device/cpu_device_synchronizer.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/device/cpu_hash_table.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/factory/ms_factory.h"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_ftrl_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_lazy_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/unique_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/unique_with_pad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/adam_delta_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/fused_ada_factor_cpu_kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/optimizer/*.cc"
        "../../../mindspore/ccsrc/plugin/device/gpu/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/hccl/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/kernel_query.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/rts/rt_kernel_info.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/mindir/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge_backend_optimization.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge_optimization.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/get_value_helper.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/backend_common_unify_mindir.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/format_type/insert_cast.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/format_type/set_fracz_group_attr.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/format_type/utils.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/format_type/deal_ref_output.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/cdist_fission.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/bn_split.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/bn_grad_split.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/seed_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/tensor_scatter_fission.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/adam_weight_decay_fission.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/batch_norm_grad_infer_fission.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fusion/batchnorm_to_bninfer.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/
        ascend_convert_tuple_input_to_dynamic_input.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fusion/batchnormgrad_to_bninfergrad.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fusion/histogram_fixed_width_fusion.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fusion/adaptive_max_pool2d_fusion.cc"
        "../../../mindspore/ccsrc/backend/common/graph_kernel/*.cc"
        "../../../mindspore/ccsrc/backend/common/expander/*.cc"
        "../../../mindspore/ccsrc/common/expander/*.cc"
        "../../../mindspore/ccsrc/common/pynative/*.cc"
        "../../../mindspore/ccsrc/common/symbol_engine/*.cc"
        "../../../mindspore/ccsrc/common/np_dtype/*.cc"
        "../../../mindspore/ccsrc/backend/common/session/anf_runtime_algorithm.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_session.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_auto_monad.cc"
        "../../../mindspore/ccsrc/backend/common/session/kernel_graph.cc"
        "../../../mindspore/ccsrc/backend/common/session/kernel_graph_mgr.cc"
        "../../../mindspore/ccsrc/backend/common/session/exec_order_builder.cc"
        "../../../mindspore/ccsrc/backend/common/session/session_basic.cc"
        "../../../mindspore/ccsrc/backend/common/session/executor.cc"
        "../../../mindspore/ccsrc/backend/common/session/executor_manager.cc"
        "../../../mindspore/ccsrc/backend/common/session/session_factory.cc"
        "../../../mindspore/ccsrc/backend/common/session/kernel_build_client.cc"
        "../../../mindspore/ccsrc/backend/operator/*.cc"
        "../../../mindspore/ccsrc/ps/*.cc"
        "../../../mindspore/ccsrc/distributed/cluster/actor_route_table_service.cc"
        "../../../mindspore/ccsrc/distributed/cluster/actor_route_table_proxy.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/rpc/rpc_recv_kernel.cc"
        "../../../mindspore/ccsrc/distributed/persistent/*.cc"
        "../../../mindspore/ccsrc/distributed/rpc/tcp/*.cc"
        "../../../mindspore/ccsrc/distributed/rpc/rdma/*.cc"
        "../../../mindspore/ccsrc/distributed/cluster/topology/*.cc"
        "../../../mindspore/ccsrc/distributed/embedding_cache/*.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/*.cc"
        "../../../mindspore/ccsrc/debug/profiler/profiling.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/profiler/cpu_profiling.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/hal/profiler/cpu_data_saver.cc"
        "../../../mindspore/ccsrc/debug/profiler/data_saver.cc"
        "../../../mindspore/ccsrc/debug/common/csv_writer.cc"
        "../../../mindspore/ccsrc/plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.c"
        "../../../mindspore/ccsrc/kernel/kernel.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/ascend_kernel_mod.cc"
        "../../../mindspore/ccsrc/backend/common/optimizer/helper.cc"
        )

list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/distributed/rpc/rdma/rdma_client.cc"
        "../../../mindspore/ccsrc/distributed/rpc/rdma/rdma_server.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/mindir/conv2d_unify_mindir.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge/dropout_for_ge.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge/convert_attr_to_input.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge/convert_resize_nearest_neighbor_x_dtype.cc"
        "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/ge/batchnorm_transform.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/util.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/ascend/optimizer/create_node_helper.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/scheduler.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_add_relu_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/post_batch_norm_add_relu_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_add_relu_grad_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_relu_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/gpu/optimizer/batch_norm_relu_grad_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/tbe_kernel_compile.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/tbe_version.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/tbe/tiling/op_tiling_adapter.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg/akg_cpu_kernel_mod.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/rpc_node_scheduler.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/send_actor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/recv_actor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/mux_send_actor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/mux_recv_actor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/rpc/rpc_actor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/runtime/graph_scheduler/embedding_cache_scheduler.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/runtime/graph_scheduler/actor/embedding_cache/embedding_cache_prefetch_actor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/profiler/parallel_strategy_profiling.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/hccl/hcom_matmul_all_reduce.cc")

add_library(_ut_mindspore_obj STATIC ${MINDSPORE_SRC_LIST} $<TARGET_OBJECTS:core_proto_obj> $<TARGET_OBJECTS:mindrt_mid>
        $<TARGET_OBJECTS:common_shared_lib_obj> $<TARGET_OBJECTS:_mindspore_utils_obj>
        $<TARGET_OBJECTS:_mindspore_common_obj> ${dataengine_submodules} $<TARGET_OBJECTS:mindrecord_obj>
        $<TARGET_OBJECTS:md_log_adapter_obj> $<TARGET_OBJECTS:_mindspore_transform_symbol_obj>)
add_dependencies(_ut_mindspore_obj proto_input_ut)

foreach(number RANGE 1 ${CORE_OBJECT_COUNT})
    list(APPEND CORE_OBJECT_LIST $<TARGET_OBJECTS:core_obj_${number}>)
endforeach()
include_directories("${CMAKE_BINARY_DIR}/plugin/device/ascend/kernel/aicpu")
file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/proto/*.proto")
ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN}
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/ascend/optimizer/ir_fission/random_status.proto)

file(GLOB_RECURSE PROTO_DUMP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/plugin/device/ascend/hal/device/dump/proto/*.proto")
ms_protobuf_generate(DUMP_PROTOSRCS PROTOHDRS ${PROTO_DUMP})

list(APPEND MINDSPORE_PROTO_LIST ${PROTOSRCS})
list(APPEND MINDSPORE_PROTO_LIST ${PREDICT_PROTOSRCS})
list(APPEND MINDSPORE_PROTO_LIST ${DUMP_PROTOSRCS})

if(MINDSPORE_PROTO_LIST)
    add_library(proto_input_ut STATIC ${MINDSPORE_PROTO_LIST})
    set_target_properties(proto_input_ut PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-array-bounds")
endif()

add_library(backend_static STATIC
        $<TARGET_OBJECTS:_mindspore_debug_obj>
        $<TARGET_OBJECTS:_mindspore_common_debug_obj>
        $<TARGET_OBJECTS:_mindspore_debug_profiler_obj>
        $<TARGET_OBJECTS:_mindspore_ps_obj>
        $<TARGET_OBJECTS:_mindspore_distributed_obj>
        $<TARGET_OBJECTS:_mindspore_kernel_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_mem_reuse_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_optimizer_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_pass_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_session_obj>
        $<TARGET_OBJECTS:_mindspore_backend_common_somas_obj>
        $<TARGET_OBJECTS:_mindspore_backend_graph_compiler_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_collective_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_device_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_graph_scheduler_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_hardware_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_pynative_obj>
        $<TARGET_OBJECTS:_mindspore_runtime_data_queue_obj>)

add_library(_ut_obj OBJECT ${UT_SRCS} ${EXTEND_SRC_LIST})
add_dependencies(_ut_obj engine-cache-server)
foreach(comp ${ALL_UT_COMPS})
    add_library(_ut_${comp}_obj OBJECT ${UT_${comp}_SRCS})
    add_dependencies(_ut_${comp}_obj engine-cache-server)
    set(ut_${comp}_objects $<TARGET_OBJECTS:_ut_${comp}_obj> $<TARGET_OBJECTS:_ut_obj> ${CORE_OBJECT_LIST})
    add_executable(ut_${comp}_tests ${ut_${comp}_objects})
    # ci envs have no enough memory, so make link target sequentially
    if(LAST_COMP)
        add_dependencies(ut_${comp}_tests ${LAST_COMP})
    endif()
    set(LAST_COMP ut_${comp}_tests)
    target_link_libraries(ut_${comp}_tests PRIVATE -Wl,--start-group mindspore backend_static proto_input
            _ut_mindspore_obj -Wl,--end-group proto_input_ut)

    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(ut_${comp}_tests PRIVATE mindspore::gtest mindspore::gmock mindspore::mockcpp
                mindspore::event mindspore::event_pthreads mindspore::event_openssl mindspore::ssl mindspore::crypto
                ${PYTHON_LIBRARIES} pthread util dl)
        target_link_libraries(ut_${comp}_tests PRIVATE mindspore::sqlite mindspore::jpeg_turbo mindspore::turbojpeg
                mindspore::opencv_core mindspore::opencv_imgcodecs mindspore::opencv_imgproc mindspore::tinyxml2
                mindspore::sentencepiece mindspore::sentencepiece_train mindspore::icuuc mindspore::icudata
                mindspore::icui18n mindspore::z)
        if(ENABLE_FFMPEG)
            target_link_libraries(ut_${comp}_tests PRIVATE mindspore::avcodec mindspore::avdevice mindspore::avfilter
                    mindspore::avformat mindspore::avutil mindspore::swresample mindspore::swscale)
        endif()
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(ut_${comp}_tests PRIVATE  mindspore::z)
    else()
        target_link_libraries(ut_${comp}_tests PRIVATE mindspore::gtest mindspore::gmock mindspore::mockcpp
                ${PYTHON_LIBRARIES})
    endif()
    target_link_libraries(ut_${comp}_tests PRIVATE mindspore::glog)
    target_link_libraries(ut_${comp}_tests PRIVATE securec mindspore::grpc++ mindspore::protobuf)
endforeach()
