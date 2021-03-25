message("build ut testcases...")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")
if(ENABLE_DUMP_IR)
    add_compile_definitions(ENABLE_DUMP_IR)
endif()
if(ENABLE_D)
    add_compile_definitions(ENABLE_D)
endif()
if(NOT ENABLE_GLIBCXX)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
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
    include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image)
    file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ./stub/*.cc
            ./common/*.cc
            ./abstract/*.cc
            ./base/*.cc
            ./dataset/*.cc
            ./ir/dtype/*.cc
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image/lite_cv/*.cc
            ./debug/*.cc
            ./device/*.cc
            ./ir/*.cc
            ./kernel/*.cc
            ./mindrecord/*.cc
            ./operator/*.cc
            ./optimizer/*.cc
            ./parallel/*.cc
            ./pipeline/*.cc
            ./pre_activate/*.cc
            ./pynative/*.cc
            ./session/*.cc
            ./transform/*.cc
            ./utils/*.cc
            ./vm/*.cc
            ./ps/*.cc
            ./cxx_api/*.cc
            )

    if(NOT ENABLE_PYTHON)
        set(PYTHON_RELATED_SRCS
                dataset/filter_op_test.cc
                dataset/voc_op_test.cc
                dataset/manifest_op_test.cc
                dataset/sentence_piece_vocab_op_test.cc
                )
        list(REMOVE_ITEM UT_SRCS ${PYTHON_RELATED_SRCS})
    endif()

    if(NOT ENABLE_ACL)
        set(ASCEND310_RELATED_SRCS
                dataset/dvpp_decode_jpeg_test.cc
                )
        list(REMOVE_ITEM UT_SRCS ${ASCEND310_RELATED_SRCS})
    endif()
else()
    file(GLOB_RECURSE TEMP_UT_SRCS ./*.cc)
    foreach(OBJ ${TEMP_UT_SRCS})
        if(NOT ${OBJ} MATCHES "./dataset/" AND NOT ${OBJ} MATCHES "./mindrecord/")
            list(APPEND UT_SRCS ${OBJ})
        endif()
    endforeach()
endif()

file(GLOB_RECURSE MINDSPORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../mindspore/ccsrc/pybind_api/*.cc"
        "../../../mindspore/ccsrc/frontend/optimizer/*.cc"
        "../../../mindspore/ccsrc/frontend/parallel/*.cc"
        "../../../mindspore/ccsrc/frontend/operator/*.cc"
        # dont remove the 4 lines above
        "../../../mindspore/ccsrc/debug/data_dump/dump_json_parser.cc"
        "../../../mindspore/ccsrc/debug/common.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime.cc"
        "../../../mindspore/ccsrc/runtime/device/memory_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_runtime_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/kernel_info.cc"
        "../../../mindspore/ccsrc/runtime/device/bucket.cc"
        "../../../mindspore/ccsrc/runtime/device/launch_kernel.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/profiling/*.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/kernel_select_ascend.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_launch_kernel.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_launch_mul.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_launch_atomic_clean.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/kernel_select_graph_kernel.cc"
        "../../../mindspore/ccsrc/runtime/device/convert_tensor_utils.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_bucket.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_event.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/kernel_build_ascend.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/signal_util.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_memory_manager.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_device_address.cc"
        "../../../mindspore/ccsrc/runtime/device/ascend/ascend_memory_pool.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/cpu_kernel_factory.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_ftrl_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_lazy_adam_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/unique_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/unique_with_pad_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/cpu/adam_delta_cpu_kernel.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/akg/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/rts/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/hccl/*.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/kernel_query.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/aicpu/aicpu_kernel_metadata.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/rts/rt_kernel_info.cc"
        "../../../mindspore/ccsrc/backend/kernel_compiler/tbe/*.cc"
        "../../../mindspore/ccsrc/backend/optimizer/ascend/*.cc"
        "../../../mindspore/ccsrc/backend/optimizer/graph_kernel/*.cc"
        "../../../mindspore/ccsrc/backend/session/anf_runtime_algorithm.cc"
        "../../../mindspore/ccsrc/backend/session/ascend_session.cc"
        "../../../mindspore/ccsrc/backend/session/ascend_auto_monad.cc"
        "../../../mindspore/ccsrc/backend/session/ascend_control_parser.cc"
        "../../../mindspore/ccsrc/backend/session/kernel_graph.cc"
        "../../../mindspore/ccsrc/backend/session/session_basic.cc"
        "../../../mindspore/ccsrc/backend/session/executor.cc"
        "../../../mindspore/core/ops/*.cc"
        "../../../mindspore/ccsrc/backend/session/executor_manager.cc"
        "../../../mindspore/ccsrc/backend/session/session_factory.cc"
        "../../../mindspore/ccsrc/backend/session/kernel_build_client.cc"
        "../../../mindspore/ccsrc/transform/graph_ir/*.cc"
        "../../../mindspore/ccsrc/transform/graph_ir/op_declare/*.cc"
        "../../../mindspore/ccsrc/ps/*.cc"
        "../../../mindspore/ccsrc/profiler/device/common/*.cc"
        )

list(REMOVE_ITEM MINDSPORE_SRC_LIST
  "../../../mindspore/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/util.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/internal/worker.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/internal/parameter_server.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/scheduler.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/optimizer_info.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/optimizer_info_builder.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/worker.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/parameter_server.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/ps_cache/gpu/gpu_ps_cache.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/ps_cache/ascend/ascend_ps_cache.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/ps/ps_cache/ps_cache_manager.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/backend/optimizer/gpu/batch_norm_add_relu_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/backend/optimizer/gpu/post_batch_norm_add_relu_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/backend/optimizer/gpu/batch_norm_add_relu_grad_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/backend/optimizer/gpu/batch_norm_relu_fusion.cc")
list(REMOVE_ITEM MINDSPORE_SRC_LIST "../../../mindspore/ccsrc/backend/optimizer/gpu/batch_norm_relu_grad_fusion.cc")

add_library(_ut_mindspore_obj OBJECT ${MINDSPORE_SRC_LIST})
add_library(_ut_ut_obj OBJECT ${UT_SRCS})
add_dependencies(_ut_ut_obj engine-cache-server)
add_executable(ut_tests $<TARGET_OBJECTS:_ut_ut_obj>
        $<TARGET_OBJECTS:_ut_mindspore_obj>)

if(ENABLE_GE)
    if(ENABLE_TRAIN)
        target_link_libraries(ut_tests PRIVATE graph ge_runner)
    else()
        target_link_libraries(ut_tests PRIVATE graph ge_client)
    endif()

    target_link_libraries(mindspore PRIVATE tsdclient)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ut_tests PRIVATE mindspore::gtest mindspore::event mindspore::event_pthreads
                          mindspore_gvar ${PYTHON_LIBRARIES} pthread util dl)
    if(ENABLE_MINDDATA)

        # AUX_SOURCE_DIRECTORY(LITE_CV_FILES)
        # message(STATUS "xxxxxxxxxxxxxxxxx"${LITE_CV_FILES} )
        # add_library(_live_cv OBJECT ${LITE_CV_FILES})

        target_link_libraries(ut_tests PRIVATE _c_dataengine _c_mindrecord)
    endif()
else()
    target_link_libraries(ut_tests PRIVATE mindspore::gtest mindspore_gvar ${PYTHON_LIBRARIES})
endif()
if(USE_GLOG)
    target_link_libraries(ut_tests PRIVATE mindspore::glog)
endif()

target_link_libraries(mindspore mindspore_core)
target_link_libraries(ut_tests PRIVATE mindspore mindspore_shared_lib securec graph)
