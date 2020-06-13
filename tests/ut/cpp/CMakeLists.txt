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
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${MS_CCSRC_BUILD_PATH})

if(ENABLE_MINDDATA)
    add_definitions(-D ENABLE_MINDDATA)
    link_directories(${MS_CCSRC_BUILD_PATH}/dataset)
    link_directories(${MS_CCSRC_BUILD_PATH}/mindrecord)
endif()
# fetch ut test files
if(ENABLE_MINDDATA)
    file(GLOB_RECURSE UT_SRCS ./*.cc)
else()
    file(GLOB_RECURSE TEMP_UT_SRCS ./*.cc)
    foreach(OBJ ${TEMP_UT_SRCS})
        if (NOT ${OBJ} MATCHES "./dataset/" AND NOT ${OBJ} MATCHES "./mindrecord/")
            list(APPEND UT_SRCS ${OBJ})
        endif()
    endforeach ()
endif()

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/ir/*.cc"
        "../../../mindspore/ccsrc/common/*.cc"
        "../../../mindspore/ccsrc/utils/*.cc"
        "../../../mindspore/ccsrc/parallel/*.cc"
        "../../../mindspore/ccsrc/pipeline/parse/*.cc"
        "../../../mindspore/ccsrc/pipeline/static_analysis/*.cc"
        "../../../mindspore/ccsrc/pipeline/pipeline.cc"
        "../../../mindspore/ccsrc/pipeline/resource.cc"
        "../../../mindspore/ccsrc/pipeline/pass.cc"
        "../../../mindspore/ccsrc/pipeline/action.cc"
        "../../../mindspore/ccsrc/pipeline/validator.cc"
        "../../../mindspore/ccsrc/pipeline/remove_value_node_dup.cc"
        "../../../mindspore/ccsrc/optimizer/*.cc"
        "../../../mindspore/ccsrc/debug/*.cc"
        "../../../mindspore/ccsrc/operator/*.cc"
        "../../../mindspore/ccsrc/transform/*.cc"
        "../../../mindspore/ccsrc/session/anf_runtime_algorithm.cc"
        "../../../mindspore/ccsrc/session/ascend_session.cc"
        "../../../mindspore/ccsrc/session/ascend_control_parser.cc"
        "../../../mindspore/ccsrc/session/kernel_graph.cc"
        "../../../mindspore/ccsrc/session/session_basic.cc"
        "../../../mindspore/ccsrc/session/session_factory.cc"
        "../../../mindspore/ccsrc/vm/*.cc"
        "../../../mindspore/ccsrc/pynative/*.cc"
        "../../../mindspore/ccsrc/pybind_api/*.cc"
        "../../../mindspore/ccsrc/kernel/akg/*.cc"
        "../../../mindspore/ccsrc/kernel/kash/*.cc"
        "../../../mindspore/ccsrc/kernel/cce/*.cc"
        "../../../mindspore/ccsrc/kernel/rts/*.cc"
        "../../../mindspore/ccsrc/kernel/hccl/*.cc"
        "../../../mindspore/ccsrc/kernel/kernel_query.cc"
        "../../../mindspore/ccsrc/kernel/kernel_build_info.cc"
        "../../../mindspore/ccsrc/pre_activate/ascend/*.cc"
        "../../../mindspore/ccsrc/pre_activate/common/*.cc"
        "../../../mindspore/ccsrc/pre_activate/gpu/*.cc"
        "../../../mindspore/ccsrc/pre_activate/mem_reuse/*.cc"
        "../../../mindspore/ccsrc/pre_activate/pass/*.cc"
        "../../../mindspore/ccsrc/kernel/aicpu/aicpu_kernel_metadata.cc"
        "../../../mindspore/ccsrc/kernel/rts/rt_kernel_info.cc"
        "../../../mindspore/ccsrc/kernel/common_utils.cc"
        "../../../mindspore/ccsrc/kernel/oplib/*.cc"
        "../../../mindspore/ccsrc/kernel/tbe/*.cc"
        "../../../mindspore/ccsrc/device/kernel_runtime.cc"
        "../../../mindspore/ccsrc/device/memory_manager.cc"
        "../../../mindspore/ccsrc/device/kernel_runtime_manager.cc"
        "../../../mindspore/ccsrc/device/kernel_info.cc"
        "../../../mindspore/ccsrc/device/ascend/profiling/*.cc"
        "../../../mindspore/ccsrc/device/ascend/kernel_select_ascend.cc"
        "../../../mindspore/ccsrc/device/ascend/kernel_select_composite.cc"
        "../../../mindspore/ccsrc/device/convert_tensor_utils.cc"
        "../../../mindspore/ccsrc/device/ascend/kernel_build_ascend.cc"
        "../../../mindspore/ccsrc/device/ascend/ascend_kernel_runtime.cc"
        "../../../mindspore/ccsrc/device/ascend/ascend_memory_manager.cc"
        "../../../mindspore/ccsrc/device/ascend/ascend_device_address.cc"
        "../../../mindspore/ccsrc/device/ascend/ascend_memory_pool.cc"
        "../../../mindspore/ccsrc/predict/generator/utils/ir_model_util.cc"
        "../../../mindspore/ccsrc/predict/predict.cc"
        "../../../mindspore/ccsrc/predict/converter/*.cc"
        "../../../mindspore/ccsrc/predict/converter/attr_utils/*.cc"
        "../../../mindspore/ccsrc/predict/converter/lite_model/*.cc"
        "../../../mindspore/ccsrc/predict/converter/lite_model/operations/*.cc"
        )

list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/debug/dump_proto.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/utils/anf_ir.pb.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/utils/node_strategy.pb.cc")

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

add_executable(ut_tests ${UT_SRCS} ${MINDSPORE_SRC_LIST} ${UT_SUTB_SRC_LIST})

if (ENABLE_GE)
    if(ENABLE_TRAIN)
        target_link_libraries(ut_tests PRIVATE graph ge_client_train)
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
