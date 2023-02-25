set(CMAKE_OBJCOPY $ENV{CROSS_COMPILE}objcopy)
set(CMAKE_STRIP $ENV{CROSS_COMPILE}strip)

file(GLOB ALL_BINARIES
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/*.so
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/lib/*.so
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/lib/plugin/*.so*
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/lib/plugin/*/*.so
        )

foreach(item ${ALL_BINARIES})
    execute_process(
            COMMAND ${CMAKE_OBJCOPY} --only-keep-debug ${item} ${item}.sym
            WORKING_DIRECTORY ${MS_PACK_ROOT_DIR}
    )
    execute_process(
            COMMAND ${CMAKE_STRIP} ${item}
            WORKING_DIRECTORY ${MS_PACK_ROOT_DIR}
    )
endforeach()

file(GLOB DEBUG_SYM_FILE
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/*.sym
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/lib/*.sym
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/lib/plugin/*.sym
        ${MS_INSTALL_DIR}/${MS_PACKAGE_FILE_NAME}/lib/plugin/*/*.sym
        )

file(MAKE_DIRECTORY ${MS_PACK_ROOT_DIR}/debug_info)
file(COPY ${DEBUG_SYM_FILE} DESTINATION ${MS_PACK_ROOT_DIR}/debug_info/)
file(REMOVE_RECURSE ${DEBUG_SYM_FILE})
execute_process(COMMAND ${CMAKE_COMMAND} -E tar cfv ${MS_PACKAGE_FILE_NAME}.debuginfo.zip debug_info/
        --format=zip WORKING_DIRECTORY ${MS_PACK_ROOT_DIR})
file(REMOVE_RECURSE ${MS_PACK_ROOT_DIR}/debug_info)
