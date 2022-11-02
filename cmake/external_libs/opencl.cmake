if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "git@gitee.com:src-openeuler/opencl-clhpp.git")
    set(GIT_TAG "7347fa1bb52ebee9f3d6c44ff65ef3c4253cab79")
    set(SHA256 "d41d8cd98f00b204e9800998ecf8427e")

    __download_pkg_with_git(OpenCL-CLHPP ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    set(OPENCL_CLHPP_SRC "${TOP_DIR}/mindspore/lite/build/_deps/opencl-clhpp-src")
    execute_process(COMMAND tar -xf ${OPENCL_CLHPP_SRC}/v2.0.12.tar.gz --strip-components 1 -C ${OPENCL_CLHPP_SRC})

    set(OPENCL_HEADER_SRC "${TOP_DIR}/mindspore/lite/build/_deps/opencl-headers-src")
    execute_process(COMMAND mkdir -p ${OPENCL_HEADER_SRC})
    execute_process(COMMAND tar -xf ${OPENCL_CLHPP_SRC}/v2020.12.18.tar.gz --strip-components 1 -C ${OPENCL_HEADER_SRC})
elseif(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/OpenCL-Headers/repository/archive/v2020.12.18.tar.gz")
    set(SHA256 "076251b94284b931399ee525527bc9aef3f5f6f3f3b1964ae485218cc88956ba")
    __download_pkg(OpenCL-Headers ${REQ_URL} ${SHA256})
    set(REQ_URL "https://gitee.com/mirrors/OpenCL-CLHPP/repository/archive/v2.0.12.tar.gz")
    set(SHA256 "d5bdbfb614a6494de97abf7297db6d2c88a55a095b12949d797ce562f5d4fdce")
    __download_pkg(OpenCL-CLHPP ${REQ_URL} ${SHA256})
else()
    set(REQ_URL "https://github.com/KhronosGroup/OpenCL-Headers/archive/v2020.12.18.tar.gz")
    set(SHA256 "5dad6d436c0d7646ef62a39ef6cd1f3eba0a98fc9157808dfc1d808f3705ebc2")
    __download_pkg(OpenCL-Headers ${REQ_URL} ${SHA256})
    set(REQ_URL "https://github.com/KhronosGroup/OpenCL-CLHPP/archive/v2.0.12.tar.gz")
    set(SHA256 "20b28709ce74d3602f1a946d78a2024c1f6b0ef51358b9686612669897a58719")
    __download_pkg(OpenCL-CLHPP ${REQ_URL} ${SHA256})
endif()

function(gene_opencl CL_SRC_DIR)
    message(STATUS "**********gene opencl********* cl path: " "${CL_SRC_DIR}")
    if(NOT EXISTS ${CL_SRC_DIR})
        return()
    endif()
    file(GLOB_RECURSE CL_LIST ${CL_SRC_DIR}/*.cl)
    foreach(file_path ${CL_LIST})
        file(REMOVE ${file_path}.inc)
        string(REGEX REPLACE ".+/(.+)\\..*" "\\1" kernel_name "${file_path}")
        set(inc_file_ex "${file_path}.inc")
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
        __exec_cmd(COMMAND sed -i "1i\\static const char *${kernel_name}_source =\\\"\\\\n\\\" \\\\"
          ${inc_file_ex} WORKING_DIRECTORY ${CL_SRC_DIR})
        __exec_cmd(COMMAND sed -i "$a\\\\\;" ${inc_file_ex} WORKING_DIRECTORY ${CL_SRC_DIR})
    endforeach()
endfunction()
