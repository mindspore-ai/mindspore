include(${CMAKE_SOURCE_DIR}/cmake/metadef_variables.cmake)

# AICPU
add_subdirectory(aicpu/aicpu_ops)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)
endif()

if(DEFINED ENV{MS_SYMMETRIC_MEMORY_KERNEL_HOME})
    add_compile_definitions(ENABLE_SYMMETRIC_MEMORY_KERNELS)
endif()

set(ASCEND_KERNEL_COMP
        aclop
        acl_ir
        aclnn
        akg
        custom
        hccl
        kernel_packet
        simu
        dvm
)

foreach(_comp ${ASCEND_KERNEL_COMP})
    add_subdirectory(${_comp})
    if(TARGET _mindspore_ops_kernel_ascend_${_comp}_obj)
        list(APPEND ASCEND_KERNEL_SRC $<TARGET_OBJECTS:_mindspore_ops_kernel_ascend_${_comp}_obj>)
    endif()
endforeach()

list(APPEND ASCEND_KERNEL_SRC "atb/kernel_mod_impl/atb_kernel_build.cc")
list(APPEND ASCEND_KERNEL_SRC "internal/internal_kernel_build.cc")
list(APPEND ASCEND_KERNEL_SRC "internal/internal_ascend_adapter.cc")
list(APPEND ASCEND_KERNEL_SRC "symmetric_memory/symmetric_memory_kernel_build.cc")

add_subdirectory(atb)
add_subdirectory(internal)
add_subdirectory(symmetric_memory)

set_property(SOURCE ${ASCEND_KERNEL_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(mindspore_ops_ascend SHARED ${ASCEND_KERNEL_SRC})
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    target_compile_definitions(mindspore_ops_ascend PRIVATE OPS_ASCEND_DLL)
endif()
target_link_libraries(mindspore_ops_ascend PRIVATE mindspore_core mindspore_ops mindspore_common
                      mindspore_pyboost mindspore_backend_common
                      mindspore_hardware_abstract d_collective multi_ascend_collective
                      mindspore_ascend_res_manager mindspore_runtime_utils ${DVM_LIB}
                      proto_input mindspore::protobuf)
add_dependencies(mindspore_ops_ascend dvm)
set_target_properties(mindspore_ops_ascend PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
