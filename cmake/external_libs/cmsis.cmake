set(cmsis_pkg_name cmsis)

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/CMSIS_5/repository/archive/5.7.0")
    set(MD5 "f8b5c3f0711feb9ebac0fb05c15f0306")
else()
    set(REQ_URL "https://github.com/ARM-software/CMSIS_5/archive/5.7.0.tar.gz")
    set(MD5 "0eaa594b0c62dd72e41ec181c4689842")
endif()

set(INCLUDE "./")

mindspore_add_pkg(${cmsis_pkg_name}
        VER 5.7.0
        HEAD_ONLY ${INCLUDE}
        URL ${REQ_URL}
        MD5 ${MD5})

message("micro get ${cmsis_pkg_name} config hash: ${${cmsis_pkg_name}_CONFIG_HASH}")

file(GLOB cmsic_children RELATIVE ${_MS_LIB_CACHE} ${_MS_LIB_CACHE}/*)

foreach(child ${cmsic_children})
    string(FIND "${child}" "${cmsis_pkg_name}" position)
    if(NOT "${position}" EQUAL "-1")
        file(STRINGS ${_MS_LIB_CACHE}/${child}/options.txt cmsis_configs)
        foreach(cmsis_config ${cmsis_configs})
            string(FIND "${cmsis_config}" "${MD5}" position_md5)
            if(NOT "${position_md5}" EQUAL "-1")
                if(NOT IS_DIRECTORY ${CMAKE_BINARY_DIR}/${cmsis_pkg_name})
                    MESSAGE("copy cmsis libaray: ${child} to ${CMAKE_BINARY_DIR}")
                    file(COPY ${_MS_LIB_CACHE}/${child}/CMSIS DESTINATION ${CMAKE_BINARY_DIR}/${cmsis_pkg_name})
                endif()
            endif()
        endforeach()
    endif()
endforeach()
