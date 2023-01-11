set(cmsis_pkg_name cmsis)

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/CMSIS_5/repository/archive/5.7.0.tar.gz")
    set(SHA256 "1b4aa6d47c7d3a5032555049b95f4962a700e2022405f863781010606fe7f8f1")
else()
    set(REQ_URL "https://github.com/ARM-software/CMSIS_5/archive/5.7.0.tar.gz")
    set(SHA256 "1b4aa6d47c7d3a5032555049b95f4962a700e2022405f863781010606fe7f8f1")
endif()

set(INCLUDE "./")

mindspore_add_pkg(${cmsis_pkg_name}
        VER 5.7.0
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        SHA256 ${SHA256})

message("micro get ${cmsis_pkg_name} config hash: ${${cmsis_pkg_name}_CONFIG_HASH}")

file(GLOB cmsic_children RELATIVE ${_MS_LIB_CACHE} ${_MS_LIB_CACHE}/*)

foreach(child ${cmsic_children})
    string(FIND "${child}" "${cmsis_pkg_name}" position)
    if(NOT "${position}" EQUAL "-1")
        file(STRINGS ${_MS_LIB_CACHE}/${child}/options.txt cmsis_configs)
        foreach(cmsis_config ${cmsis_configs})
            string(FIND "${cmsis_config}" "${SHA256}" position_sha256)
            if(NOT "${position_sha256}" EQUAL "-1")
                if(NOT IS_DIRECTORY ${CMAKE_BINARY_DIR}/${cmsis_pkg_name})
                    MESSAGE("copy cmsis libaray: ${child} to ${CMAKE_BINARY_DIR}")
                    file(COPY ${_MS_LIB_CACHE}/${child}/CMSIS DESTINATION ${CMAKE_BINARY_DIR}/${cmsis_pkg_name})
                endif()
            endif()
        endforeach()
    endif()
endforeach()
