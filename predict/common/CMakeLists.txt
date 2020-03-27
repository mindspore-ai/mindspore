include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)

add_compile_options(-fPIC)

add_library(common_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/common.h
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/file_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/func_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/module_registry.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mslog.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/storage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc)
