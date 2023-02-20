set(MINDSPORE_PROJECT_DIR ${TOP_DIR})

find_required_package(Patch)

if(MSLITE_DEPS_ROBIN_HOOD_HASHING)
    include(${TOP_DIR}/cmake/external_libs/robin.cmake)
endif()

if(MSLITE_DEPS_FLATBUFFERS)
    include(${TOP_DIR}/cmake/external_libs/flatbuffers.cmake)
endif()

if(MSLITE_DEPS_OPENCL)
    include(${TOP_DIR}/cmake/external_libs/opencl-headers.cmake)
    include(${TOP_DIR}/cmake/external_libs/opencl-clhpp.cmake)
endif()

if(MSLITE_DEPS_JSON)
    include(${TOP_DIR}/cmake/external_libs/json.cmake)
endif()

if(MSLITE_DEPS_GLOG)
    include(${TOP_DIR}/cmake/external_libs/glog.cmake)
endif()

if(MSLITE_DEPS_PROTOBUF)
    include(${TOP_DIR}/cmake/external_libs/protobuf.cmake)
endif()

if(MSLITE_DEPS_EIGEN)
    include(${TOP_DIR}/cmake/external_libs/eigen.cmake)
endif()

if(MSLITE_DEPS_OPENCV)
    include(${TOP_DIR}/cmake/external_libs/opencv.cmake)
endif()

if(MSLITE_DEPS_FAST_TRANSFORMERS)
    include(${TOP_DIR}/cmake/external_libs/fast_transformers.cmake)
endif()

if(MSLITE_DEPS_MKLDNN)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        set(USE_MS_THREADPOOL_FOR_DNNL ON)
    endif()
    if(USE_MS_THREADPOOL_FOR_DNNL)
        add_compile_definitions(USE_MS_THREADPOOL_FOR_DNNL)
    endif()
include(${TOP_DIR}/cmake/external_libs/mkl_dnn.cmake)
endif()

if(MSLITE_DEPS_LIBEVENT)
    include(${TOP_DIR}/cmake/external_libs/libevent.cmake)
endif()

if(MSLITE_DEPS_PYBIND11)
    find_package(Python3 COMPONENTS Interpreter Development)
    if(Python3_FOUND)
        find_package(Python3 COMPONENTS NumPy Development)
        if(Python3_NumPy_FOUND)
            include_directories(${Python3_INCLUDE_DIRS})
            include_directories(${Python3_NumPy_INCLUDE_DIRS})
            include_directories(${TOP_DIR})
            include_directories(${CORE_DIR})
            set(PYBIND11_CPP_STANDARD -std=c++17)
            include(${TOP_DIR}/cmake/external_libs/pybind11.cmake)
        endif()
    endif()
endif()

if(MSLITE_DEPS_OPENSSL)
    include(${TOP_DIR}/cmake/external_libs/openssl.cmake)
endif()