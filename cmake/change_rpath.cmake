function(changerpath target_so target_name link_sos)
    set(depend_so "")
    foreach(link_so ${link_sos})
        set(some-file "${CMAKE_SOURCE_DIR}/build/${target_name}_${link_so}.txt")
        set(some-file1 "${CMAKE_SOURCE_DIR}/build/${target_name}_${link_so}1.txt")
        set(some-file2 "${CMAKE_SOURCE_DIR}/build/${target_name}_${link_so}2.txt")
        add_custom_command(
                OUTPUT
                ${some-file}
                COMMAND
                otool -L ${target_so} | tail -n +2 | grep ${link_so} | head -n1 > ${some-file}
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
                VERBATIM
        )
        add_custom_command(
                OUTPUT
                ${some-file1}
                COMMAND
                cat ${some-file} | cut -d " " -f 1 | sed -E "s/^.//g" > ${some-file1}
                DEPENDS
                ${some-file}
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
                VERBATIM
        )
        add_custom_command(
                OUTPUT
                ${some-file2}
                COMMAND
                awk -F "/"  "{print $NF}"  ${some-file1} > ${some-file2}
                DEPENDS
                ${some-file1}
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
                VERBATIM
        )
        add_custom_target(
                link_${target_name}_${link_so} ALL
                COMMAND install_name_tool -change `cat ${some-file1}` @rpath/`cat ${some-file2}` ${target_so}
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
                DEPENDS ${target_so} ${some-file1} ${some-file2}
                COMMENT "install tool name"
        )
        add_custom_command(
                TARGET link_${target_name}_${link_so}
                POST_BUILD
                COMMAND rm ${some-file} ${some-file1} ${some-file2}
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
                VERBATIM
        )
        if(depend_so)
            add_dependencies(link_${target_name}_${link_so} link_${target_name}_${depend_so})
        endif()
        set(depend_so ${link_so})
    endforeach()
endfunction()
