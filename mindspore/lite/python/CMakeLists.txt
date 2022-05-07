cmake_minimum_required(VERSION 3.12)
project(MindSpore_Lite_Python_API)

# set(CMAKE_VERBOSE_MAKEFILE on)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -Wl,-rpath,$ORIGIN/")

find_package(Python3 COMPONENTS Interpreter Development)

if(Python3_FOUND)
  find_package(Python3 COMPONENTS NumPy Development)

  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${Python3_NumPy_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../core/)

  if(NOT ENABLE_CLOUD_AND_LITE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/utils.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/external_libs/pybind11.cmake)
  endif()

  file(GLOB_RECURSE PY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  pybind11_add_module(_c_lite_wrapper NO_EXTRAS ${PY_SRC_LIST})

  target_link_libraries(_c_lite_wrapper PRIVATE -Wl,--whole-archive mindspore-lite_static -Wl,--no-whole-archive)
  if(MSLITE_ENABLE_CONVERTER OR MSLITE_ENABLE_RUNTIME_GLOG)
    target_link_libraries(_c_lite_wrapper PRIVATE mindspore::glog)
  endif()
else()
  message(WARNING "Python3 not found, so Python API will not be compiled.")
endif()
