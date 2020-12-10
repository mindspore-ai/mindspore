# build mindspore_shared_lib
set(LOAD_MINDIR_SRC
        ${CMAKE_SOURCE_DIR}/mindspore/core/load_mindir/load_model.cc
        ${CMAKE_SOURCE_DIR}/mindspore/core/load_mindir/anf_model_parser.cc
        )
file(GLOB_RECURSE API_OPS_SRC ${CMAKE_CURRENT_SOURCE_DIR} "ops/*.cc")

if (ENABLE_ACL)
    add_compile_definitions(ENABLE_ACL)
    include_directories(${CMAKE_SOURCE_DIR}/graphengine/ge)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)
    file(GLOB_RECURSE API_ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
            "python_utils.cc"
            "model/acl/*.cc"
            "model/model_converter_utils/*.cc"
            "graph/acl/*.cc"
            )

endif ()
if (ENABLE_D)
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR} "python_utils.cc" "model/ms/*.cc" "graph/ms/*.cc")
endif ()

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

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_shared_lib PRIVATE ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
            -Wl,-force_load mindspore -Wl,-noall_load proto_input mindspore_gvar mindspore::protobuf)
else ()
    target_link_libraries(mindspore_shared_lib PRIVATE ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY}
            -Wl,--whole-archive mindspore -Wl,--no-whole-archive proto_input mindspore_gvar mindspore::protobuf)
endif ()

if (ENABLE_CPU)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore::dnnl mindspore::mkldnn)
endif ()

if (USE_GLOG)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore::glog)
endif ()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(mindspore_shared_lib PRIVATE -Wl,-init,common_log_init)
endif ()

if (ENABLE_ACL)
    if (DEFINED ENV{ASCEND_CUSTOM_PATH})
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else ()
        set(ASCEND_PATH /usr/local/Ascend)
    endif ()
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
    target_link_libraries(mindspore_shared_lib PRIVATE ${acl} ${acl_retr} ${acl_cblas} ${acl_dvpp} ${acl_runtime}
            ${ge_compiler} mindspore::jpeg_turbo)
endif ()


# Before build inference
if (ENABLE_D)
    find_library(adump_server libadump_server.a ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_shared_lib PRIVATE ${adump_server})
endif ()

