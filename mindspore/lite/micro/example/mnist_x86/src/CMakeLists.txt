
cmake_minimum_required(VERSION 3.14)
project(net)

if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

get_filename_component(PKG_PATH ${PKG_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(OP_LIB ${PKG_PATH}/inference/lib/libmindspore-lite.a)
set(WRAPPER_LIB ${PKG_PATH}/tools/codegen/lib/libwrapper.a)
set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)
set(HEADER_PATH ${PKG_PATH}/inference)

message("operator lib path: ${OP_LIB}")
message("operator header path: ${OP_HEADER_PATH}")

add_compile_definitions(NOT_USE_STL)

include_directories(${OP_HEADER_PATH})
include_directories(${HEADER_PATH})

if(NOT PLATFORM_ARM32 AND NOT PLATFORM_ARM64)
  include_directories(${PKG_PATH}/tools/codegen/third_party/include)
  include_directories(${PKG_PATH}/tools/codegen/third_party/include/CMSIS/Core/Include)
  include_directories(${PKG_PATH}/tools/codegen/third_party/include/CMSIS/DSP/Include)
  include_directories(${PKG_PATH}/tools/codegen/third_party/include/CMSIS/NN/Include)
endif()

include(net.cmake)

option(PLATFORM_ARM64 "build android arm64" OFF)
option(PLATFORM_ARM32 "build android arm32" OFF)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
  add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build net library with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    message(STATUS "build net library release version")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

function(create_library)
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND rm -rf tmp
            COMMAND mkdir tmp
            COMMAND cd tmp && ar -x ${OP_LIB}
            COMMAND cd tmp && ar -x ${WRAPPER_LIB}
            COMMAND echo "raw static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND mv ${library_name} ./tmp && cd tmp && ar -x ${library_name}
            COMMENT "unzip raw static library ${library_name}"
            )
    if(NOT PLATFORM_ARM32 AND NOT PLATFORM_ARM64)
        set(CMSIS_LIB ${PKG_PATH}/tools/codegen/third_party/lib/libcmsis_nn.a)
        add_custom_command(TARGET net POST_BUILD COMMAND cd tmp && ar -x ${CMSIS_LIB})
    endif()

    foreach(object_file ${OP_SRC})
        add_custom_command(TARGET net POST_BUILD COMMAND mv ./tmp/${object_file} .)
    endforeach()
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND ar cr ${library_name} *.o
            COMMAND ranlib ${library_name}
            COMMAND echo "new static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND rm -rf tmp && rm -rf *.o
            COMMENT "generate specified static library ${library_name}"
            )
endfunction(create_library)
string(CONCAT library_name "lib" net ".a")
create_library()
