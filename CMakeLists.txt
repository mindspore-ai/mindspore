cmake_minimum_required(VERSION 3.14)
project (MindSpore)

include(${CMAKE_SOURCE_DIR}/cmake/options.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Werror -Wno-return-std-move -Wno-unused-private-field -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")    
else()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wl,--allow-shlib-undefined -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer -Wl,--allow-shlib-undefined -D_LIBCPP_INLINE_VISIBILITY='' -D'_LIBCPP_EXTERN_TEMPLATE(...)=' -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2 -Wno-cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include -std=c++17 -Werror -Wall -Wno-deprecated-declarations -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

find_package(Threads)
include(${CMAKE_SOURCE_DIR}/cmake/mind_expression.cmake)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/flatbuffers/include/flatbuffers)

include(${CMAKE_SOURCE_DIR}/cmake/dependency_utils.cmake)
find_package(Python3 3.7 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}")
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}")
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

if (ENABLE_GE)
    link_directories(${CMAKE_SOURCE_DIR}/third_party/ge/lib)
elseif(ENABLE_D OR ENABLE_TESTCASES)
    include(${CMAKE_SOURCE_DIR}/cmake/dependency_graphengine.cmake)
endif()

if (ENABLE_GE OR ENABLE_D OR ENABLE_TESTCASES)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graphengine/inc)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graphengine/inc/external)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graphengine/inc/framework)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graphengine/third_party/fwkacllib/inc)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graphengine/third_party/fwkacllib/inc/toolchain)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
add_subdirectory(mindspore/ccsrc)
if (ENABLE_TESTCASES)
    add_subdirectory(tests)
endif()

include(cmake/package.cmake)