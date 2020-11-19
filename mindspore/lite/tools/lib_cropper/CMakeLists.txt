# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )

add_executable(lib_cropper
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/lib_cropper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cropper_flags.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cropper_utils.cc
        ${COMMON_SRC})

add_dependencies(lib_cropper fbs_src)

target_link_libraries(lib_cropper mindspore-lite_static)

add_custom_command(TARGET lib_cropper POST_BUILD COMMAND
        bash build_cropper_config.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
