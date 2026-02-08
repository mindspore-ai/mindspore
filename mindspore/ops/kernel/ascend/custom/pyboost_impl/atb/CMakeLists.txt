include_directories(${ASCEND_NNAL_ATB_PATH}/include)
file(GLOB_RECURSE EXTENSION_ASCEND_ATB_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)

set_property(SOURCE ${EXTENSION_ASCEND_ATB_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_EXTENSION)
add_library(mindspore_extension_ascend_atb STATIC ${EXTENSION_ASCEND_ATB_SRC_LIST})

find_library(ATB_KENREL atb ${ASCEND_NNAL_ATB_OPP_PATH})
target_link_libraries(mindspore_extension_ascend_atb PRIVATE mindspore_extension ${ATB_KENREL})
