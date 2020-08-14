message("build ut testcases...")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")
add_compile_definitions(ENABLE_DUMP_E2E)
if(ENABLE_DUMP_IR)
    add_compile_definitions(ENABLE_DUMP_IR)
endif(ENABLE_DUMP_IR)
if(ENABLE_D)
    add_compile_definitions(ENABLE_D)
endif()

#add python lib and include for all ut executables;
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MS_CCSRC_PATH})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${MS_CCSRC_BUILD_PATH})

if(ENABLE_MINDDATA)
    add_definitions(-D ENABLE_MINDDATA)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/dataset)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/mindrecord)
endif()
# fetch ut test files
if(ENABLE_MINDDATA)
    file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.cc)
    if(NOT ENABLE_PYTHON)
        set(PYTHON_RELATED_SRCS
            dataset/filter_op_test.cc
            dataset/voc_op_test.cc
            dataset/manifest_op_test.cc
            dataset/sentence_piece_vocab_op_test.cc
        )
        list(REMOVE_ITEM UT_SRCS ${PYTHON_RELATED_SRCS})
    endif()
else()
    file(GLOB_RECURSE TEMP_UT_SRCS ./*.cc)
    foreach(OBJ ${TEMP_UT_SRCS})
        if (NOT ${OBJ} MATCHES "./dataset/" AND NOT ${OBJ} MATCHES "./mindrecord/")
            list(APPEND UT_SRCS ${OBJ})
        endif()
    endforeach ()
endif()
# removing serving ut
file(GLOB_RECURSE SERVING_ACL_UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} serving/*.cc)
list(REMOVE_ITEM UT_SRCS ${SERVING_ACL_UT_SRCS})
add_subdirectory(serving)

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/core/base/*.cc"
        "../../../mindspore/core/abstract/*.cc"
        "../../../mindspore/core/ir/*.cc"
        "../../../mindspore/core/utils/*.cc"
        "../../../mindspore/core/c_ops/*.cc"
        "../../../mindspore/ccsrc/common/*.cc"
        "../../../mindspore/ccsrc/utils/*.cc"
        "../../../mindspore/ccsrc/pipeline/jit/parse/*.cc"
        "../../../mindspore/ccsrc/pipeline/jit/static_analysis/*.cc"
        "../../../mindspore/ccsrc/pipeline/jit/pipeline.cc"
        "../../../mindspore/ccsrc/pipeline/jit/resource.cc"
        "../../../mindspore/ccsrc/pipeline/jit/pass.cc"
        "../../../mindspore/ccsrc/pipeline/jit/action.cc"
        "../../../mindspore/ccsrc/pipeline/jit/validator.cc"
        "../../../mindspore/ccsrc/pipeline/jit/remove_value_node_dup.cc"
        "../../../mindspore/ccsrc/frontend/optimizer/*.cc"
        "../../../mindspore/ccsrc/frontend/parallel/*.cc"
        "../../../mindspore/ccsrc/debug/*.cc"
        "../../../mindspore/ccsrc/frontend/operator/*.cc"
        "../../../mindspore/ccsrc/transform/graph_ir/*.cc"
        "../../../mindspore/ccsrc/backend/session/anf_runtime_algorithm.cc"
        "../../../mindspore/ccsrc/backend/session/ascend_session.cc"
        "../../../mindspore/ccsrc/backend/session/ascend_control_parser.cc"
        "../../../mindspore/ccsrc/backend/session/kernel_graph.cc"
        "../../../mindspore/ccsrc/backend/session/session_basic.cc"
        "../../../mindspore/ccsrc/backend/session/session_factory.cc"
        "../../../mindspore/ccsrc/backend/session/kernel_build_client.cc"
        "../../../mindspore/ccsrc/vm/*.cc"
        "../../../mindspore/ccsrc/pipeline/pynative/*.cc"
        "../../../mindspore/ccsrc/pybind_api/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/akg/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/kash/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/rts/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/hccl/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/kernel_query.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/kernel_build_info.cc"
        "../../../mindspore/ccsrc/backend/optimizer/ascend/*.cc"
        "../../../mindspore/ccsrc/backend/optimizer/common/*.cc"
        "../../../mindspore/ccsrc/backend/optimizer/gpu/*.cc"
        "../../../mindspore/ccsrc/backend/optimizer/mem_reuse/*.cc"
        "../../../mindspore/ccsrc/backend/optimizer/pass/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/aicpu/aicpu_kernel_metadata.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/rts/rt_kernel_info.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/common_utils.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/oplib/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/tbe/*.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_info.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/profiling/*.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/kernel_select_graph_kernel.cc"
        "../../../mindspore/ccsrc/runtime/device/convert_tensor_utils.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/kernel_build_ascend.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_memory_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_device_address.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_memory_pool.cc"
        "../../../mindspore/ccsrc/predict/generator/utils/ir_model_util.cc"
        "../../../mindspore/ccsrc/predict/predict.cc"
        "../../../mindspore/ccsrc/predict/converter/*.cc"
        "../../../mindspore/ccsrc/predict/converter/attr_utils/*.cc"
        "../../../mindspore/ccsrc/predict/converter/lite_model/*.cc"
        "../../../mindspore/ccsrc/predict/converter/lite_model/operations/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/cpu_kernel_factory.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_ftrl_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_lazy_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/unique_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/unique_with_pad_cpu_kernel.cc"
        )

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/common/duplex_pipe.cc")
else()
    list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/common/duplex_pipe_win.cc")
endif()
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/dump_proto.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/core/ir/lite/tensor.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/frontend/parallel/ps/util.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/frontend/parallel/ps/scheduler.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/frontend/parallel/ps/optimizer_info.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/frontend/parallel/ps/optimizer_info_builder.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/utils/anf_ir.pb.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/utils/node_strategy.pb.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/utils/load_onnx/anf_model_parser.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/utils/load_onnx/anf_converter.cc")

# remove files for debugger
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/debugger/debugger.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/debugger/grpc_client.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/debug_services.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/debugger/proto_exporter.cc")

file(GLOB_RECURSE UT_SUTB_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "stub/aicpu/*.cc"
        "stub/cce/*.cc"
        "stub/tdt/*.cc"
        "stub/tasksink/*.cc"
        "stub/device/*.cc"
        "stub/kernel/*.cc"
        "stub/runtime/*.cc"
        "stub/anf_ir/*.cc"
        "stub/profiling/*.cc"
        "stub/parallel_strategy_checkpoint/*.cc"
        "stub/hccl/*.cc"
        "stub/ge/*.cc"
        )

add_executable(ut_tests ${UT_SRCS} ${MINDSPORE_SRC_LIST} ${UT_SUTB_SRC_LIST} $<TARGET_OBJECTS:ut_serving_obj>)

if (ENABLE_GE)
    if(ENABLE_TRAIN)
        target_link_libraries(ut_tests PRIVATE graph ge_runner)
    else()
        target_link_libraries(ut_tests PRIVATE graph ge_client)
    endif()

    target_link_libraries(mindspore PRIVATE tsdclient)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ut_tests PRIVATE mindspore::gtest mindspore_gvar ${PYTHON_LIBRARIES} pthread util dl)
    if (ENABLE_MINDDATA)
        target_link_libraries(ut_tests PRIVATE _c_dataengine _c_mindrecord)
    endif()
else()
    target_link_libraries(ut_tests PRIVATE mindspore::gtest mindspore_gvar ${PYTHON_LIBRARIES})
endif()
if (USE_GLOG)
    target_link_libraries(ut_tests PRIVATE mindspore::glog)
endif()

target_link_libraries(ut_tests PRIVATE securec graph)

# link grpc
if (EXISTS ${grpc_ROOT}/lib64)
    set(gRPC_DIR "${grpc_ROOT}/lib64/cmake/grpc")
else ()
    set(gRPC_DIR "${grpc_ROOT}/lib/cmake/grpc")
endif ()
find_package(gRPC CONFIG REQUIRED)
target_link_libraries(ut_tests PRIVATE gRPC::grpc++)
target_link_libraries(ut_tests PRIVATE gRPC::grpc++_reflection)
target_link_libraries(ut_tests PRIVATE protobuf::libprotobuf)