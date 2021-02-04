cmake_minimum_required(VERSION 3.14)
project(micro)
string(REPLACE "/mindspore/lite/micro" "" MINDSPORE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
if(NOT ENABLE_CONVERTER)
    set(CMAKE_CXX_STANDARD 17)
    include(${MINDSPORE_ROOT_DIR}/cmake/utils.cmake)
    include(${MINDSPORE_ROOT_DIR}/cmake/dependency_utils.cmake)
    include(${MINDSPORE_ROOT_DIR}/cmake/dependency_securec.cmake)
    include(${MINDSPORE_ROOT_DIR}/cmake/external_libs/flatbuffers.cmake)
    set(FBS_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/../schema/model.fbs
            ${CMAKE_CURRENT_SOURCE_DIR}/../schema/ops.fbs
            )

    ms_build_flatbuffers_lite(FBS_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/../schema/
            fbs_src
            ${CMAKE_BINARY_DIR}/schema
            ""
            )

    ms_build_flatbuffers_lite(FBS_FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/../schema/
            fbs_inner_src
            ${CMAKE_BINARY_DIR}/schema/inner
            "inner"
            )
endif()

add_subdirectory(coder)
