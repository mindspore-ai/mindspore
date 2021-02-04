if(NOT ENABLE_CONVERTER)
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
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections ")
set(LITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(3RD_DIR ${LITE_DIR}/../../third_party)
set(NNACL_DIR ${LITE_DIR}/nnacl)

#include 3rd
include_directories(${3RD_DIR})
include_directories(${3RD_DIR}/flatbuffers/include)
include_directories(${3RD_DIR}/securec/include/)
#include ms
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MINDSPORE_ROOT_DIR})
include_directories(${LITE_DIR})
include_directories(${MINDSPORE_ROOT_DIR}/mindspore/core/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime)

#include coder
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
include_directories(${MINDSPORE_ROOT_DIR}/mindspore/lite/src/runtime/kernel/arm)

set(MS_SRC
        ${MINDSPORE_ROOT_DIR}/mindspore/lite/src/common/log_adapter.cc
        )

file(GLOB_RECURSE PRIMITIVE_OP_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITE_DIR}/src/ops/*.cc
        )
set(LITE_COMMON_SRC
        ${PRIMITIVE_OP_SRC}
        ${LITE_DIR}/tools/common/flag_parser.cc
        ${LITE_DIR}/src/common/file_utils.cc
        ${LITE_DIR}/src/common/graph_util.cc
        ${LITE_DIR}/src/common/string_util.cc
        ${LITE_DIR}/src/runtime/allocator.cc
        ${LITE_DIR}/src/lite_model.cc
        ${LITE_DIR}/src/tensorlist.cc
        ${LITE_DIR}/src/tensor.cc
        )

file(GLOB_RECURSE MICRO_ALLOCATOR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        allocator/*.cc
        )

file(GLOB_RECURSE MICRO_GENERATOR RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        generator/*.cc
        )

file(GLOB_RECURSE MICRO_UTILS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        utils/*.cc
        )

set(MICRO_CODER_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/coder_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session_coder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/debug.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/coder_graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/op_coder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/op_coder_register.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/op_coder_builder.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/opcoders/file_collector.cc
        )

list(APPEND MICRO_CODER_SRC
        ${MICRO_ALLOCATOR}
        ${MICRO_GENERATOR}
        )


add_executable(codegen main.cc
        ${MS_SRC}
        ${MICRO_UTILS}
        ${LITE_COMMON_SRC}
        ${MICRO_CODER_SRC}
        ${MICRO_RUNTIME_SRC})

add_dependencies(codegen fbs_src)
add_dependencies(codegen fbs_inner_src)

target_link_libraries(codegen ${SECUREC_LIBRARY} ${CMAKE_DL_LIBS})



