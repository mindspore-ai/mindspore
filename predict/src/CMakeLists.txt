cmake_minimum_required(VERSION 3.12)
project(mspredict)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../module/tvm_kernel/incubator-tvm/3rdparty/dlpack/include/)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../output/lib/)

if (ENABLE_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -o0 -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined")
endif()

set(MSPREDICT_SRC
        runtime/allocator.cc
        runtime/allocator.h
        runtime/thread_pool.cc
        runtime/thread_pool.h
        runtime/workspace_pool.cc
        runtime/workspace_pool.h
        runtime/runtime_api.cc
        runtime/runtime_api.h
        context.cc
        graph.cc
        graph.h
        graph_execution.cc
        graph_execution.h
        node.cc
        node.h
        op.cc
        op.h
        op_factory.cc
        op_factory.h
        op_registry.cc
        op_registry.h
        session.cc
        tensor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/operator/cpu/common/op_func_comm.cc)

set(MSPREDICT_SRC ${MSPREDICT_SRC}
	       ${CMAKE_CURRENT_SOURCE_DIR}/../common/graph_util.cc
	       ${CMAKE_CURRENT_SOURCE_DIR}/../common/utils.cc
	       ${CMAKE_CURRENT_SOURCE_DIR}/../common/mslog.cc
	       ${CMAKE_CURRENT_SOURCE_DIR}/../common/module_registry.cc)

add_library(mspredict SHARED ${MSPREDICT_SRC})

if(ENABLE_PREDICT_ARM64 OR ENABLE_PREDICT_ARM32)
  target_link_libraries(mspredict ${ANDROID_NDK} android log tvm_kernel libsecurec.a)
else()
  target_link_libraries(mspredict pthread tvm_kernel libsecurec.a)
endif()

add_dependencies(mspredict tvm_kernel)
add_dependencies(mspredict securec)
add_dependencies(mspredict gtest)

add_custom_command(TARGET mspredict POST_BUILD
        COMMAND mkdir -pv ${PREDICT_DIR}/output/lib
        COMMAND cp ${PREDICT_BUILD_DIR}/src/libmspredict.so ${PREDICT_DIR}/output/lib/
        COMMAND cp ${PREDICT_BUILD_DIR}/module/tvm_kernel/lite/libtvm_kernel.so ${PREDICT_DIR}/output/lib/
        COMMAND mkdir -pv ${PREDICT_DIR}/output/include
        COMMAND cp -r ${PREDICT_DIR}/include/* ${PREDICT_DIR}/output/include
        COMMAND mkdir -pv ${PREDICT_DIR}/output/include/schema/inner
        COMMAND cp ${PREDICT_DIR}/schema/ms_generated.h ${PREDICT_DIR}/output/include/schema/inner
        COMMAND cp ${PREDICT_DIR}/schema/op_generated.h ${PREDICT_DIR}/output/include/schema/inner
        COMMAND mkdir -pv ${PREDICT_DIR}/output/include/dlpack/
        COMMAND cp ${PREDICT_DIR}/module/tvm_kernel/incubator-tvm/3rdparty/dlpack/include/dlpack/dlpack.h ${PREDICT_DIR}/output/include/dlpack/)
