cmake_minimum_required(VERSION 3.14.0)
project(MindSpore)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/check_requirements.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(NOT ENABLE_GLIBCXX)
        add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_SYSROOT "")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Winconsistent-missing-override -Wuser-defined-warnings \
        -Wno-return-std-move -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-sign-compare \
        -Wno-overloaded-virtual -Wno-unneeded-internal-declaration -Wno-unused-variable -Wno-pessimizing-move \
        -Wno-inconsistent-missing-override -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wl,--allow-shlib-undefined \
        -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
endif()

if(ENABLE_PYTHON)
    add_compile_definitions(ENABLE_PYTHON)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer \
    -Wl,--allow-shlib-undefined -D_LIBCPP_INLINE_VISIBILITY='' -D_LIBCPP_DISABLE_EXTERN_TEMPLATE=1 \
    -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2 -Wno-cpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include -std=c++17 \
    -Werror -Wall -Wno-deprecated-declarations -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

if(ENABLE_AKG AND (ENABLE_D OR ENABLE_GPU))
    add_subdirectory("${CMAKE_SOURCE_DIR}/akg")
endif()

include(${CMAKE_SOURCE_DIR}/cmake/mind_expression.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/securec/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include/flatbuffers)

include(${CMAKE_SOURCE_DIR}/cmake/dependency_utils.cmake)
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
    if(WIN32)
        if(Python3_DIR)
            message("Python3_DIR set already: " ${Python3_DIR})
        else()
            string(LENGTH ${PYTHON_LIBRARIES} PYTHON_LIBRARIES_LEN)
            string(LENGTH "libpythonxx.a" Python3_NAME_LEN)
            math(EXPR Python3_DIR_LEN  ${PYTHON_LIBRARIES_LEN}-${Python3_NAME_LEN})
            string(SUBSTRING ${Python3_LIBRARIES} 0 ${Python3_DIR_LEN} Python3_DIR)
            message("Python3_DIR: " ${Python3_DIR})
        endif()
        link_directories(${Python3_DIR})
    endif()
else()
    find_python_package(py_inc py_lib)
    set(PYTHON_INCLUDE_DIRS "${py_inc}")
    set(PYTHON_LIBRARIES "${py_lib}")
endif()
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})

set(MS_CCSRC_PATH ${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
set(MS_CCSRC_BUILD_PATH ${BUILD_PATH}/mindspore/mindspore/ccsrc)

if(ENABLE_D OR ENABLE_ACL OR ENABLE_TESTCASES)
    include(${CMAKE_SOURCE_DIR}/cmake/dependency_graphengine.cmake)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
add_subdirectory(mindspore/ccsrc)
add_subdirectory(mindspore/core)
if(ENABLE_TESTCASES OR ENABLE_CPP_ST)
    add_subdirectory(tests)
endif()

include(cmake/package.cmake)
