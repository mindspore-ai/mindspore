if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/Vulkan-Headers/archive/v1.2.144.zip")
    set(MD5 "8797a525aff953ea536ebe338a9f5ef6")
    set(PKG_GIT_TAG "")
    __download_pkg_with_git(Vulkan-Headers ${REQ_URL} ${PKG_GIT_TAG} ${MD5})
else()
    set(REQ_URL "https://github.com/KhronosGroup/Vulkan-Headers/archive/v1.2.144.zip")
    set(MD5 "91eae880a0ad9ad77c89d79b95b7399a")
    __download_pkg(Vulkan-Headers ${REQ_URL} ${MD5})
endif()

function(gene_spirv BASEPATH)
    string(CONCAT CL_SRC_DIR "${BASEPATH}" "/src/runtime/kernel/vulkan/glsl")
    message(STATUS "**********gene spirv*********base path: " "${BASEPATH}" ", glsl path: " "${CL_SRC_DIR}")
    if(NOT EXISTS ${CL_SRC_DIR})
        return()
    endif()
    file(GLOB_RECURSE CL_LIST ${CL_SRC_DIR}/*.cl)
    foreach(file_path ${CL_LIST})
        file(REMOVE ${file_path}.inc)
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" kernel_name "${file_path}")
        set(inc_file_ex "${kernel_name}.cl.inc")
        execute_process(
                COMMAND bash -c "sed 's/\\\\/\\\\\\\\/g' "
                COMMAND bash -c "sed 's/\\\"/\\\\\\\"/g' "
                COMMAND bash -c "sed 's/$/\\\\n\\\" \\\\/' "
                COMMAND bash -c "sed 's/^/\\\"/' "
                WORKING_DIRECTORY ${CL_SRC_DIR}
                INPUT_FILE ${file_path}
                OUTPUT_FILE ${inc_file_ex}
                RESULT_VARIABLE RESULT)
        if(NOT RESULT EQUAL "0")
            message(FATAL_ERROR "error! when generate ${inc_file_ex}")
        endif()
        __exec_cmd(COMMAND sed -i
                "1i\\static const char *${kernel_name}_source =\\\"\\\\n\\\" \\\\"
                ${inc_file_ex} WORKING_DIRECTORY ${CL_SRC_DIR}
                )
        __exec_cmd(COMMAND sed -i "$a\\\\\;" ${inc_file_ex} WORKING_DIRECTORY ${CL_SRC_DIR})
    endforeach()
endfunction()
