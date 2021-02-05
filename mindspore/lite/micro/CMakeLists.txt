cmake_minimum_required(VERSION 3.14)
project(micro)
string(REPLACE "/mindspore/lite/micro" "" TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_BINARY_DIR})
include(${TOP_DIR}/cmake/utils.cmake)
include(${TOP_DIR}/cmake/dependency_utils.cmake)
include(${TOP_DIR}/cmake/dependency_securec.cmake)
include(${TOP_DIR}/cmake/external_libs/flatbuffers.cmake)
include(${TOP_DIR}/cmake/external_libs/cmsis.cmake)

set(FBS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../schema/model.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/../schema/ops.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/../schema/model_v0.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/../schema/ops_v0.fbs
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
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    MESSAGE(" ******Micro Debug********")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    MESSAGE(" ******Micro Release********")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -w -O2 -Werror \
    -fstack-protector-all -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -w -O2 -Werror \
    -fstack-protector-all -Wno-attributes -Wno-deprecated-declarations \
    -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
endif()
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-recover=address -lasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize-recover=address -lasan")
endif()

add_subdirectory(coder)
