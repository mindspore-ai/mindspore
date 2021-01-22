include_directories(${DDK_PATH})
file(GLOB_RECURSE NPU_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/npu/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cc
        )
add_library(hiai SHARED IMPORTED)
set_target_properties(hiai PROPERTIES IMPORTED_LOCATION
        ${DDK_LIB_PATH}/libhiai.so)
add_library(hiai_ir SHARED IMPORTED)
set_target_properties(hiai_ir PROPERTIES IMPORTED_LOCATION
        ${DDK_LIB_PATH}/libhiai_ir.so)
add_library(hiai_ir_build SHARED IMPORTED)
set_target_properties(hiai_ir_build PROPERTIES IMPORTED_LOCATION
        ${DDK_LIB_PATH}/libhiai_ir_build.so)
add_library(npu_kernel_mid OBJECT ${NPU_RUNTIME_SRC})
add_dependencies(npu_kernel_mid fbs_src)
target_link_libraries(
        npu_kernel_mid
        hiai
        hiai_ir
        hiai_ir_build
)