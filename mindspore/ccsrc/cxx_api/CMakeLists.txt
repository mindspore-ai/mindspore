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
# build mindspore_shared_lib
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
if(NOT BUILD_LITE)
    set(LOAD_MINDIR_SRC
            ${CMAKE_SOURCE_DIR}/mindspore/core/load_mindir/load_model.cc
            ${CMAKE_SOURCE_DIR}/mindspore/core/load_mindir/anf_model_parser.cc
            )
endif()

file(GLOB_RECURSE API_OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR} "ops/*.cc")

if(ENABLE_D OR ENABLE_ACL)
    # build 910 and 310 code into one distro, files needed for 310 mode
    add_compile_definitions(ENABLE_ACL)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/ge)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)

    file(GLOB_RECURSE API_ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
            "akg_kernel_register.cc"
            "model/acl/*.cc"
            "model/model_converter_utils/*.cc"
            "graph/acl/*.cc"
            )

    if(NOT ENABLE_D)
        list(APPEND API_ACL_SRC $<TARGET_OBJECTS:_mindspore_transform_graph_ir_obj>)
        list(APPEND API_ACL_SRC $<TARGET_OBJECTS:_mindspore_vm_obj>)
    endif()
endif()

if(ENABLE_D)
    # for 910 mode
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "model/ms/*.cc" "graph/ascend/*.cc")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "akg_kernel_register.cc" "model/ms/*.cc" "graph/gpu/*.cc")
endif()

set(MSLIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/types.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/context.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/cell.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_data.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/model/model_impl.cc
              ${API_MS_INFER_SRC}
              ${API_ACL_SRC}
              ${API_OPS_SRC}
              ${LOAD_MINDIR_SRC})

if(BUILD_LITE)
    list(APPEND MSLIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../mindspore/ccsrc/utils/config_manager.cc")
    list(REMOVE_ITEM MSLIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/akg_kernel_register.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/acl/acl_model_multi.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/acl/acl_model.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/model_impl.cc"
            $<TARGET_OBJECTS:_mindspore_vm_obj>)
endif()

add_library(mindspore_shared_lib SHARED ${MSLIB_SRC})
if(NOT BUILD_LITE)
    set_target_properties(mindspore_shared_lib PROPERTIES OUTPUT_NAME mindspore)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_shared_lib PRIVATE ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
            -Wl,-force_load mindspore -Wl,-noall_load mindspore_core proto_input mindspore_gvar mindspore::protobuf)
else()
    if(ENABLE_D OR ENABLE_GPU)
        target_link_libraries(mindspore_shared_lib PRIVATE -Wl,--as-needed ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
            -Wl,--whole-archive mindspore -Wl,--no-whole-archive mindspore_core proto_input mindspore_gvar
            mindspore::protobuf)
    else()
        if(BUILD_LITE)
            target_link_libraries(mindspore_shared_lib PRIVATE ${SECUREC_LIBRARY})
        else()
            target_link_libraries(mindspore_shared_lib PRIVATE ${PY3_LIBG} ${SECUREC_LIBRARY}
                    mindspore mindspore_core proto_input mindspore_gvar mindspore::protobuf)
        endif()
    endif()
endif()

if(ENABLE_CPU)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()

if(USE_GLOG)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore::glog)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(mindspore_shared_lib PRIVATE -Wl,-init,common_log_init)
endif()

if(ENABLE_D OR ENABLE_ACL)
    # 310 mode
    add_compile_definitions(ENABLE_DVPP_INTERFACE)
    find_library(acl libascendcl.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_cblas libacl_cblas.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_dvpp libacl_dvpp.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_runtime libruntime.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ge_compiler libge_compiler.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libplatform libplatform.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libcompress libcompress.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libopskernel libopskernel.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicore_utils libaicore_utils.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicpu_engine_common libaicpu_engine_common.so ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

    target_link_libraries(mindspore_shared_lib PRIVATE -Wl,--no-as-needed graph ${ge_compiler}
            ${acl_retr} ${acl_cblas} ${acl_dvpp} ${acl_runtime} ${libplatform} ${libcompress} ${libopskernel}
            ${libaicore_utils} ${libaicpu_engine_common} ${acl})
    add_dependencies(mindspore_shared_lib graph)
endif()


# Before build inference
if(ENABLE_D)
    find_library(adump_server libadump_server.a ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_shared_lib PRIVATE ${adump_server})
endif()

if(ENABLE_GPU)
    target_link_libraries(mindspore_shared_lib PRIVATE  gpu_cuda_lib gpu_queue cublas
                          ${CUDA_PATH}/lib64/libcurand.so
                          ${CUDNN_LIBRARY_PATH}
                          ${CUDA_PATH}/lib64/libcudart.so
                          ${CUDA_PATH}/lib64/stubs/libcuda.so
                          ${CUDA_PATH}/lib64/libcusolver.so
                          ${CUDA_PATH}/lib64/libcufft.so)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(MINDSPORE_RPATH $ORIGIN)
    if(ENABLE_D OR ENABLE_ACL)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/fwkacllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/fwkacllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(MINDSPORE_RPATH
            ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)

        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/atc/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/atc/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/acllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/acllib/lib64)
    endif()

    set_target_properties(mindspore_shared_lib PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
endif()
