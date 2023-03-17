include_directories(${TOP_DIR}/graphengine/910/metadef/inc/external)
include_directories(${TOP_DIR}/graphengine/910/metadef/inc)
include_directories(${TOP_DIR}/graphengine/910/inc)
include_directories(${TOP_DIR}/graphengine/910/inc/external)
include_directories(${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc)
include_directories(${CCSRC_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/ascend)

file(STRINGS "${TOP_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")
add_compile_definitions(ENABLE_SECURITY)

#link_directories(${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

file(GLOB GE_EXECUTOR_SRC
        ${CCSRC_DIR}/runtime/device/ms_device_shape_transfer.cc
        ${CCSRC_DIR}/utils/config_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${TOP_DIR}/mindspore/lite/tools/converter/adapter/acl/mapper/*.cc
        ${TOP_DIR}/mindspore/lite/tools/converter/adapter/acl/common/utils.cc
        )
list(APPEND GE_EXECUTOR_SRC $<TARGET_OBJECTS:_mindspore_transform_graph_ir_obj>)


add_library(ascend_ge_plugin SHARED ${GE_EXECUTOR_SRC})

find_library(ge_graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl_cblas libacl_cblas.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl_runtime libruntime.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ge_compiler libge_compiler.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(libplatform libplatform.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(libcompress libcompress.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(libopskernel libopskernel.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(libaicore_utils libaicore_utils.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(libaicpu_engine_common libaicpu_engine_common.so ${ASCEND_CANN_RUNTIME_PATH}
    ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ge_runner libge_runner.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

target_link_libraries(ascend_ge_plugin ${ge_graph} ${ge_compiler} ${acl_retr} ${acl_cblas} ${acl_dvpp}
                      ${acl_runtime} ${libplatform} ${libcompress} ${libopskernel} ${libaicore_utils}
                      ${libaicpu_engine_common} ${acl} ${ge_runner} mindspore_converter mindspore_core)

