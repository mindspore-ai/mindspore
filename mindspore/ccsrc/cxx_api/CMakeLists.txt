# build mindspore_shared_lib
set(LOAD_MINDIR_SRC
        ${CMAKE_SOURCE_DIR}/mindspore/core/load_mindir/load_model.cc
        ${CMAKE_SOURCE_DIR}/mindspore/core/load_mindir/anf_model_parser.cc
        )
file(GLOB_RECURSE API_OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR} "ops/*.cc")

if(ENABLE_ACL)
    add_compile_definitions(ENABLE_ACL)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/ge)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)
    file(GLOB_RECURSE API_ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
            "akg_kernel_register.cc"
            "model/acl/*.cc"
            "model/model_converter_utils/*.cc"
            "graph/acl/*.cc"
            )
    list(APPEND API_ACL_SRC $<TARGET_OBJECTS:_mindspore_transform_graph_ir_obj>)
endif()

if(ENABLE_D)
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "akg_kernel_register.cc" "model/ms/*.cc" "graph/ascend/*.cc")
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
              ${API_MS_INFER_SRC}
              ${API_ACL_SRC}
              ${API_OPS_SRC}
              ${LOAD_MINDIR_SRC})

add_library(mindspore_shared_lib SHARED ${MSLIB_SRC})
set_target_properties(mindspore_shared_lib PROPERTIES OUTPUT_NAME mindspore)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_shared_lib PRIVATE ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
            -Wl,-force_load mindspore -Wl,-noall_load proto_input mindspore_gvar mindspore::protobuf)
else()
    if(ENABLE_D OR ENABLE_GPU)
        target_link_libraries(mindspore_shared_lib PRIVATE ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
            -Wl,--whole-archive mindspore -Wl,--no-whole-archive proto_input mindspore_gvar mindspore::protobuf)
    else()
        target_link_libraries(mindspore_shared_lib PRIVATE ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
          mindspore proto_input mindspore_gvar mindspore::protobuf)
    endif()
endif()

if(ENABLE_CPU)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore::dnnl mindspore::mkldnn)
endif()

if(USE_GLOG)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore::glog)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(mindspore_shared_lib PRIVATE -Wl,-init,common_log_init)
endif()

if(ENABLE_ACL)
    if(DEFINED ENV{ASCEND_CUSTOM_PATH})
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH /usr/local/Ascend)
    endif()
    set(ACL_LIB_DIR ${ASCEND_PATH}/acllib/)
    set(ATLAS_ACL_LIB_DIR ${ASCEND_PATH}/ascend-toolkit/latest/acllib)
    set(ATC_DIR ${ASCEND_PATH}/atc/)
    set(ATLAS_ATC_DIR ${ASCEND_PATH}/ascend-toolkit/latest/atc)
    MESSAGE("acl lib dir " ${ACL_LIB_DIR} ", atc dir " ${ATC_DIR})
    MESSAGE("atlas acl lib dir " ${ATLAS_ACL_LIB_DIR} ", atc dir " ${ATLAS_ATC_DIR})

    include_directories(${ACL_LIB_DIR}/include/)
    include_directories(${ATLAS_ACL_LIB_DIR}/include/)
    add_compile_definitions(ENABLE_DVPP_INTERFACE)
    find_library(acl libascendcl.so ${ACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)
    find_library(acl_retr libacl_retr.so ${ACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)
    find_library(acl_cblas libacl_cblas.so ${ACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)
    find_library(acl_dvpp libacl_dvpp.so ${ACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)
    find_library(acl_runtime libruntime.so ${ACL_LIB_DIR}/lib64 ${ATLAS_ACL_LIB_DIR}/lib64)
    find_library(ge_compiler libge_compiler.so ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)
    find_library(libplatform libplatform.so ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)
    find_library(libcompress libcompress.so ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)
    find_library(libopskernel libopskernel.so ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)
    find_library(libaicore_utils libaicore_utils.so ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)
    find_library(libaicpu_engine_common libaicpu_engine_common.so ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)
    find_library(atc_protobuf libascend_protobuf.so.3.8.0.0 ${ATC_DIR}/lib64 ${ATLAS_ATC_DIR}/lib64)

    target_link_libraries(mindspore_shared_lib PRIVATE -Wl,--no-as-needed graph ${ge_compiler} ${atc_protobuf}
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
    if(ENABLE_D)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/fwkacllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/fwkacllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/fwkacllib/lib64/plugin/opskernel)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/plugin/opskernel)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/add-ons)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(MINDSPORE_RPATH
            ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
    elseif(ENABLE_ACL)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/atc/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/atc/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/atc/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/acllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/acllib/lib64)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/add-ons)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(MINDSPORE_RPATH ${MINDSPORE_RPATH}:/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(MINDSPORE_RPATH
            ${MINDSPORE_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
    endif()

    set_target_properties(mindspore_shared_lib PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
endif()
