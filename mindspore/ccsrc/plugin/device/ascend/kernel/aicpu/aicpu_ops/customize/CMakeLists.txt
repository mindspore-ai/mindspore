# spetcify output dir
set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/custom_aicpu_ops)
set(OP_IMPL_PATH ${OUTPUT_DIR}/op_impl/cpu/aicpu_kernel/impl)
set(OP_CFG_PATH ${OUTPUT_DIR}/op_impl/cpu/config)
set(OP_PROTO_PATH ${OUTPUT_DIR}/op_proto)
file(MAKE_DIRECTORY ${OP_IMPL_PATH})
file(MAKE_DIRECTORY ${OP_CFG_PATH})
file(MAKE_DIRECTORY ${OP_PROTO_PATH})
# =========================config json compile============================
set(AICPU_INI_2_JSON_PY "${CMAKE_CURRENT_SOURCE_DIR}/utils/aicpu_parser_ini.py")
function(GEN_OPS_INFO_FILE output_dir output_file_name_prefix)
    SET(INI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg)
    if(IS_DIRECTORY ${INI_PATH})
        execute_process(COMMAND find ${INI_PATH} -name "*.ini" OUTPUT_VARIABLE INI_FILES)
        if(NOT "x${INI_FILES}" STREQUAL "x")
            string(REPLACE "\n" ";" INI_FILES ${INI_FILES})
        endif()
        set(output_file_name ${output_file_name_prefix}_aicpu_kernel.json)
        if(NOT "x${INI_FILES}" STREQUAL "x")
            execute_process(
                COMMAND mkdir -p ${output_dir}
                COMMAND ${Python3_EXECUTABLE} ${AICPU_INI_2_JSON_PY} ${INI_FILES} ${output_dir}/${output_file_name})
        else()
            message(STATUS "ini file not exists.")
            execute_process(COMMAND mkdir -p ${output_dir})
        endif()
    endif()
endfunction()

GEN_OPS_INFO_FILE(${OP_CFG_PATH} cust)

# =========================proto compile============================
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
add_subdirectory(op_proto)

# =========================install pkg============================
set(INSTALL_LIBRARY_DIR lib/plugin/ascend)
install(DIRECTORY ${OUTPUT_DIR} DESTINATION ${INSTALL_LIBRARY_DIR})
set(CPU_KERNELS ${CMAKE_CURRENT_BINARY_DIR}/../cpu_kernel/libmindspore_cpu_kernels.so)
set(AICPU_KERNELS ${CMAKE_CURRENT_BINARY_DIR}/../libmindspore_aicpu_kernels.so)
set(IMPL_FILE_DIR ${INSTALL_LIBRARY_DIR}/custom_aicpu_ops/op_impl/cpu/aicpu_kernel/impl)
set(ENV{CMAKE_INSTALL_MODE} REL_SYMLINK)
# copy libmindspore_cpu_kernels.so to ${IMPL_FILE_DIR} and rename to "libcust_cpu_kernels.so"
install(FILES ${CPU_KERNELS} RENAME "libcust_cpu_kernels.so" DESTINATION ${IMPL_FILE_DIR})
