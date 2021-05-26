cmake_minimum_required(VERSION 3.14.0)
project(MinddataCropper)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIE -fPIC -Wl,--allow-shlib-undefined -s")

file(GLOB minddata_OBJ CONFIGURE_DEPENDS "tmp/*.o")

if(NOT minddata_OBJ)
    message(FATAL_ERROR "Your code is not using any MindData functionality.\n \
    ... libminddata-lite_min.so is not needed\n... Terminating crop.sh")
endif()

message(STATUS ${CMAKE_CXX_COMPILER})

add_custom_command(
        OUTPUT libminddata-lite_min.so
        PRE_BUILD
        COMMAND ${CMAKE_CXX_COMPILER}
        -shared
        -o libminddata-lite_min.so
        ${minddata_OBJ}
        ${EXTERNAL_DEPS}
        -pthread
        -std=c++17
        -fPIE -fPIC
        -s
)

add_custom_target(
        minddata-lite ALL
        DEPENDS libminddata-lite_min.so
)
