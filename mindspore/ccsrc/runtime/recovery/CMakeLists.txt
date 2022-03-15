file(GLOB_RECURSE RECOVERY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "recovery_context.cc")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${RECOVERY_SRC_LIST} PROPERTY SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_recovery_obj OBJECT ${RECOVERY_SRC_LIST})
