# add shared link library
set(COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/common/utils.cc
        )

add_executable(net_train
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/net_train.cc
        ${COMMON_SRC})
if (WIN32)
    add_dependencies(net_train fbs_src mindspore-lite_static)
else ()
    add_dependencies(net_train fbs_src)
endif ()

if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    target_link_libraries(net_train mindspore-lite)
else()
    if (WIN32)
        target_link_libraries(net_train mindspore-lite_static pthread cpu_kernel_mid nnacl_mid)
    else ()
        target_link_libraries(net_train mindspore-lite pthread)
    endif ()
endif()
if (PLATFORM_ARM32 OR PLATFORM_ARM64)
    install(TARGETS net_train
                RUNTIME DESTINATION ${MAIN_DIR}-${COMPONENT_NAME}/net_train COMPONENT ${COMPONENT_NAME})
else()
    if (WIN32)
        install(TARGETS net_train
            RUNTIME DESTINATION ${MAIN_DIR}-${WIN_RUN_X86_NAME}/net_train COMPONENT ${WIN_RUN_X86_NAME})
    else ()
        install(TARGETS net_train
            RUNTIME DESTINATION ${MAIN_DIR}-${RUN_X86_COMPONENT_NAME}/net_train COMPONENT ${RUN_X86_COMPONENT_NAME})
    endif ()
endif()
