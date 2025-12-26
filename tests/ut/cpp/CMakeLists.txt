message("build ut testcases...")

string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")
add_compile_definitions(ENABLE_DUMP_IR)
if(NOT ENABLE_GLIBCXX)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
endif()

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    MESSAGE(STATUS "MS_INTERNAL_KERNEL_HOME=$ENV{MS_INTERNAL_KERNEL_HOME} found.")
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME})
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel/include)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)

    link_directories($ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel/lib)
    link_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/lib)
endif()

file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

#add python lib and include for all ut executables;
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MS_CCSRC_PATH})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/pass/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops/kernel/cpu)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops/kernel/cpu/native)

include(${CMAKE_SOURCE_DIR}/cmake/metadef_variables.cmake)
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
        ./ir/dtype/*.cc ./ir/*.cc ./mindapi/*.cc ./mindir/*.cc ./ops/*.cc ./ops/view/*.cc ./base/*.cc ./inferinfo/*.cc)
file(GLOB_RECURSE UT_MINDDATA_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./dataset/*.cc ./mindrecord/*.cc)
file(GLOB_RECURSE UT_MINDDATA_COMMON_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./dataset/common/*.cc)
file(GLOB_RECURSE UT_FRONTEND_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./operator/*.cc ./optimizer/*.cc
        ./parallel/*.cc ./pipeline/*.cc ./func_graph_builder/*.cc  ./side_effect/*.cc ./utils/*.cc)
file(GLOB_RECURSE UT_PYNATIVE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./pynative/*.cc)
file(GLOB_RECURSE UT_OLD_BACKEND_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./device/*.cc ./kernel/*.cc ./kernel/*/*.cc
        ./pre_activate/common/*.cc  ./session/*.cc ./runtime/device/move_to_test.cc
        ./transform/*.cc ./vm/*.cc ./runtime/core/*.cc ./runtime/device/gsm/*.cc)
file(GLOB_RECURSE UT_BACKEND_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./backend/*.cc ./runtime/graph_scheduler/*.cc
        ./runtime/pipeline/*.cc ./plugin/device/ascend/hal/hardware/*.cc)
file(GLOB_RECURSE UT_GRAPH_KERNEL_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./backend/ms_backend/graph_fusion/*.cc)
file(GLOB_RECURSE UT_ERROR_HANDLE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./tools/error_handle/*.cc)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    file(GLOB_RECURSE UT_INTERNAL_KERNEL_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./internal/*.cc ./ops/utils/*.cc
    ./backend/graph_optimizer_test_framework.cc)
endif()

# build backend/graph_optimizer_test_framework.cc in symbol_engine obj to use the ConstructGraph functions.
file(GLOB_RECURSE UT_SYMBOL_ENGINE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ./backend/graph_optimizer_test_framework.cc ./symbol_engine/*.cc)
# plugin/device/cpu/hal/test_ms_collective_topo.cc will also open 127.0.0.1:8090
file(GLOB_RECURSE UT_PS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./cluster/rpc/core/*.cc ./cluster/rpc/tcp/*.cc
        ./cluster/topology/*.cc ./plugin/device/cpu/hal/*.cc)
file(GLOB_RECURSE UT_CCSRC_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./ccsrc/*.cc)
file(GLOB_RECURSE UT_OTHERS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./debug/*.cc ./place/*.cc)
file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./stub/*.cc ./common/*.cc ./mock/*.cc
        ../../../mindspore/ccsrc/pybind_api/frontend/manager.cc
        ../../../mindspore/ccsrc/common/python_utils.cc
        ../../../mindspore/ccsrc/pybind_api/utils/log_adapter_py.cc
        ../../../mindspore/ccsrc/pybind_api/pynative/tensor/tensor_py.cc
        ../../../mindspore/ccsrc/pybind_api/pynative/tensor/dlpack_utils.cc
        ../../../mindspore/ccsrc/pybind_api/pynative/tensor/device_type_utils.cc)

# split minddata
list(REMOVE_ITEM UT_MINDDATA_SRCS ${UT_MINDDATA_COMMON_SRCS})
list(LENGTH UT_MINDDATA_SRCS UT_MINDDATA_SRCS_LENS)
math(EXPR UT_MINDDATA0_SRCS_LENS "(${UT_MINDDATA_SRCS_LENS} + 1) / 2")

list(SUBLIST UT_MINDDATA_SRCS 0 ${UT_MINDDATA0_SRCS_LENS} UT_MINDDATA0_SRCS)
math(EXPR UT_MINDDATA1_SRCS_LENS "${UT_MINDDATA_SRCS_LENS} - ${UT_MINDDATA0_SRCS_LENS}")
list(SUBLIST UT_MINDDATA_SRCS ${UT_MINDDATA0_SRCS_LENS} ${UT_MINDDATA1_SRCS_LENS} UT_MINDDATA1_SRCS)
list(APPEND UT_MINDDATA0_SRCS ${UT_MINDDATA_COMMON_SRCS})
list(APPEND UT_MINDDATA1_SRCS ${UT_MINDDATA_COMMON_SRCS})

set(ALL_UT_COMPS CORE MINDDATA0 MINDDATA1 FRONTEND OLD_BACKEND BACKEND PS CCSRC OTHERS SYMBOL_ENGINE
        GRAPH_KERNEL PYNATIVE ERROR_HANDLE)
set(UT_COMPS CORE FRONTEND OTHERS SYMBOL_ENGINE PYNATIVE)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    list(APPEND ALL_UT_COMPS INTERNAL_KERNEL)
endif()

set(REPEATED_DEFINED_FILE stub/ps/ps_core_stub.cc)
list(REMOVE_ITEM UT_SRCS ${REPEATED_DEFINED_FILE})

set(TEST_DEVICE_ADDRESS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/common/test_device_address.cc)
add_library(_test_device_address_static STATIC ${TEST_DEVICE_ADDRESS_SRC})
list(REMOVE_ITEM UT_SRCS ${TEST_DEVICE_ADDRESS_SRC})
target_link_libraries(mindspore_hardware_abstract PRIVATE
        -Wl,--whole-archive
        _test_device_address_static
        -Wl,--no-whole-archive)

find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    find_package(Python3 COMPONENTS NumPy Development)
    if(Python3_NumPy_FOUND)
        include_directories(${Python3_NumPy_INCLUDE_DIRS})
    endif()
endif()

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/cluster/topology/*.cc"
        "../../../mindspore/ccsrc/cluster/rpc/tcp/*.cc"
        "../../../mindspore/ccsrc/cluster/rpc/core/*.cc"
        "../../../mindspore/ccsrc/plugin/cpu/res_manager/cpu_res_manager.cc"
        "../../../mindspore/ccsrc/plugin/cpu/res_manager/mem_manager/cpu_memory_pool.cc"
        "../../../mindspore/ccsrc/plugin/cpu/res_manager/collective/ms_collective_topo.cc"
        "../../../mindspore/ops/kernel/cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/adam_delta_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/akg/*.cc"
        "../../../mindspore/ops/kernel/cpu/fused_ada_factor_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/nnacl/fp32/adam_fp32.c"
        "../../../mindspore/ops/kernel/cpu/sparse_apply_adam_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/sparse_apply_ftrl_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/sparse_apply_lazy_adam_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/unique_cpu_kernel.cc"
        "../../../mindspore/ops/kernel/cpu/unique_with_pad_cpu_kernel.cc"
        )

set(ASCEND_PLUGIN_PATH "../../../mindspore/ccsrc/plugin/ascend")
file(GLOB_RECURSE MINDSPORE_ASCEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${ASCEND_PLUGIN_PATH}/res_manager/device_context_conf/op_tuning_conf.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/device_context_conf/op_debug_conf.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/abstract_ascend_memory_pool_support.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/ascend_dynamic_mem_adapter.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/event/ascend_event.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/ascend_gmem_adapter.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/ascend_memory_adapter.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/ascend_memory_pool.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/stream_manager/ascend_stream_manager.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/stream_manager/callback_thread.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/hal_manager/ascend_hal_manager.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/hal_manager/ascend_error_manager.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/ascend_two_pointer_mem_adapter.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/mem_manager/ascend_vmm_adapter.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/error_manager/collective_comm_monitor.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/ascend_res_manager.cc"
        "../../../mindspore/ccsrc/backend/ge_backend/executor/ge_memory_manager.cc"
        "../../../mindspore/ccsrc/plugin/ascend/profiler/*.cc"
        "../../../mindspore/ops/kernel/ascend/hccl/*.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/backend_common_unify_mindir.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/format_type/deal_ref_output.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/format_type/insert_identity.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/format_type/set_fracz_group_attr.cc"
        "../../../mindspore/ccsrc/backend/ge_backend/pass/*.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ascend_pass_optimization.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion/grouped_matmul_assignadd_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion/matmul_assignadd_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/add_layer_norm_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/add_rms_norm_quant_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/inference_matmul_split_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/inference_weight_preprocess_utils.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_fusion_base.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_add_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/inference_swiglu_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/matmul_sigmoid_add_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/matmul_elemwise_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/ir_fusion_infer/swiglu_dynamic_quant_fusion.cc"
        "${ASCEND_PLUGIN_PATH}/graph_optimizer/pass/mindir/*.cc"
        "${ASCEND_PLUGIN_PATH}/res_manager/symbol_interface/*.cc"
        )

list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/cluster/rpc/core/communicator/task_executor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg_cpu_kernel_mod.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/plugin/device/cpu/kernel/akg_cpu_kernel_build.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/cluster/topology/cluster_context.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/cluster/topology/actor_route_table_proxy.cc")

# list(REMOVE_ITEM MINDSPORE_ASCEND_SRC_LIST
#         "../../../mindspore/ccsrc/plugin/ascend/profiler/parallel_strategy_profiling.cc")
list(REMOVE_ITEM MINDSPORE_ASCEND_SRC_LIST
        "../../../mindspore/ops/kernel/ascend/hccl/hcom_matmul_all_reduce.cc")
list(REMOVE_ITEM MINDSPORE_ASCEND_SRC_LIST
        "../../../mindspore/ops/kernel/ascend/hccl/hcom_send.cc")
list(REMOVE_ITEM MINDSPORE_ASCEND_SRC_LIST
        "../../../mindspore/ops/kernel/ascend/hccl/hcom_receive.cc")

add_library(_ut_mindspore_obj STATIC ${MINDSPORE_SRC_LIST})
target_link_libraries(_ut_mindspore_obj PRIVATE mindspore_ms_backend mindspore_core _c_dataengine proto_input
        mindspore_common mindspore_backend_manager mindspore_hardware_abstract
        mindspore_backend_common mindspore_runtime_utils)

add_library(_ut_mindspore_ascend_obj STATIC ${MINDSPORE_ASCEND_SRC_LIST})
target_link_libraries(_ut_mindspore_ascend_obj PRIVATE mindspore_ms_backend mindspore_core
        _c_dataengine mindspore_common mindspore_backend_manager mindspore_hardware_abstract
        mindspore_profiler mindspore_runtime_pipeline mindspore_backend_common mindspore_runtime_utils)

foreach(number RANGE 1 ${CORE_OBJECT_COUNT})
    list(APPEND CORE_OBJECT_LIST $<TARGET_OBJECTS:core_obj_${number}>)
endforeach()
foreach(number RANGE 1 ${OPS_OBJECT_COUNT})
    list(APPEND OPS_OBJECT_LIST $<TARGET_OBJECTS:ops_obj_${number}>)
endforeach()
include_directories("${CMAKE_BINARY_DIR}/plugin/device/ascend/kernel/aicpu")

add_library(_ut_obj_static STATIC ${UT_SRCS})
target_link_libraries(_ut_obj_static PRIVATE mindspore_ms_backend mindspore_pyboost mindspore_backend_manager
                      mindspore_hardware_abstract mindspore_profiler mindspore_backend_common
                      mindspore_pynative_utils mindspore_runtime_pipeline)
foreach(comp ${ALL_UT_COMPS})
    add_executable(ut_${comp}_tests ${UT_${comp}_SRCS})

    # ci envs have no enough memory, so make link target sequentially
    list(FIND UT_COMPS ${comp} comp_index)
    if(${comp_index} LESS 0)
        message("${comp} start link _ut_mindspore_obj")
        target_link_libraries(ut_${comp}_tests PRIVATE -Wl,--start-group _ut_mindspore_ascend_obj _ut_mindspore_obj)
        target_link_libraries(ut_${comp}_tests PRIVATE _c_dataengine)
    endif()

    # ut/cpp needs to link _c_mindrecord because it will call back python and get the graph.
    target_link_libraries(ut_${comp}_tests PRIVATE _c_mindrecord)

    target_link_libraries(ut_${comp}_tests PRIVATE -Wl,--start-group mindspore_frontend mindspore_pynative
            _ut_obj_static -Wl,--end-group)
    target_link_libraries(ut_${comp}_tests PRIVATE mindspore_ms_backend mindspore_core mindspore_ops mindspore_common
            mindspore_backend_manager mindspore_profiler mindspore_memory_pool mindspore_runtime_utils
            mindspore_backend_common mindspore_tools)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_libraries(ut_${comp}_tests PRIVATE mindspore::gtest mindspore::gmock mindspore::mockcpp
                ${PYTHON_LIBRARIES} pthread util dl)
        target_link_libraries(ut_${comp}_tests PRIVATE mindspore::sqlite mindspore::jpeg_turbo mindspore::turbojpeg
                mindspore::opencv_core mindspore::opencv_imgcodecs mindspore::opencv_imgproc mindspore::tinyxml2
                mindspore::sentencepiece mindspore::sentencepiece_train mindspore::icuuc mindspore::icudata
                mindspore::icui18n mindspore::z)
        if(ENABLE_FFMPEG)
            target_link_libraries(ut_${comp}_tests PRIVATE mindspore::avcodec mindspore::avdevice mindspore::avfilter
                    mindspore::avformat mindspore::avutil mindspore::swresample mindspore::swscale)
        endif()
        if(${comp} STREQUAL "PS")
            target_link_libraries(ut_${comp}_tests PRIVATE mindspore::event mindspore::event_pthreads
                    mindspore::event_openssl mindspore::ssl mindspore::crypto)
        endif()
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(ut_${comp}_tests PRIVATE  mindspore::z)
    else()
        target_link_libraries(ut_${comp}_tests PRIVATE mindspore::gtest mindspore::gmock mindspore::mockcpp
                ${PYTHON_LIBRARIES})
    endif()
    target_link_libraries(ut_${comp}_tests PRIVATE mindspore::glog)
    target_link_libraries(ut_${comp}_tests PRIVATE mindspore::securec)
endforeach()
