# find python library
if(MODE_ASCEND_ACL)
    get_filename_component(PYTHON_LIB_REALPATH ${PYTHON_LIBRARIES} ABSOLUTE)
    get_filename_component(PYTHON_LIB_DIR ${PYTHON_LIB_REALPATH} PATH)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import distutils.sysconfig as sysconfig; \
                  print(sysconfig.get_config_var('PY3LIBRARY'))"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE PY3LIBRARY)
    set(PY3_LIBG ${PYTHON_LIB_DIR}/${PY3LIBRARY})
    string(STRIP "${PY3_LIBG}" PY3_LIBG)
    message("Python3 general library = " ${PY3_LIBG})
endif()
# build mslite_shared_lib
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)

if(ENABLE_D OR ENABLE_ACL)
    # build 910 and 310 code into one distro, files needed for 310 mode
    add_compile_definitions(ENABLE_ACL)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/910/ge)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)

    file(GLOB_RECURSE API_ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "acl/*.cc"
        "acl/model_converter_utils/*.cc"
        "acl/acl_vm/*.cc"
        )
endif()

if(ENABLE_D)
    # for 910 mode
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend/*.cc")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu/*.cc")
endif()

set(MSLIB_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/types.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cell.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_data.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/predict_task_queue.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_worker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_pool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_parallel_runner.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_parallel_runner_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/resource_manager.cc
    ${API_MS_INFER_SRC}
    ${API_ACL_SRC}
    ${API_OPS_SRC}
    )

if(ENABLE_D)
    list(APPEND MSLIB_INFER_SRC
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/strategy_checkpoint/strategy_checkpoint_info.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/array.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/map.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/arrangement.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/shape_util.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/tensor_layout.cc")
endif()

add_library(mindspore_infer_shared_lib_obj OBJECT ${MSLIB_INFER_SRC})
add_dependencies(mindspore_infer_shared_lib_obj fbs_inner_src)
