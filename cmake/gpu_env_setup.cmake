## Function for setting NVCC flag and cuda arch list
function(set_nvcc_flag CUDA_NVCC_FLAGS CUDA_ARCH_LIST)
    # Detect gpu archs by cudaGetDeviceProperties.
    message("Detect gpu arch on this device.")
    set(cu_file "${CMAKE_SOURCE_DIR}/build/mindspore/ccsrc/get_device_compute_capabilities.cu")
    file(WRITE ${cu_file} ""
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n"
            "int main () {\n"
            " int dev_num = 0;\n"
            " if (cudaGetDeviceCount(&dev_num) != cudaSuccess) return -1;\n"
            " if (dev_num < 1) return -1;\n"
            " for (int dev_id = 0; dev_id < dev_num; ++dev_id) {\n"
            "    cudaDeviceProp prop;"
            "    if (cudaGetDeviceProperties(&prop, dev_id) == cudaSuccess) {\n"
            "      printf(\"%d.%d \", prop.major, prop.minor);\n"
            "    }\n"
            "  }\n"
            "  return 0;\n"
            "}\n")
    # Build and run cu_file, get the result from properties.
    if(NOT MSVC)
        set(CUDA_LIB_PATH ${CUDA_PATH}/lib64/libcudart.so)
    else()
        set(CUDA_LIB_PATH ${CUDA_PATH}/lib/x64/cudart.lib)
    endif()
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR ${CMAKE_SOURCE_DIR}/build/mindspore/ccsrc/ ${cu_file}
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
            LINK_LIBRARIES ${CUDA_LIB_PATH}
            RUN_OUTPUT_VARIABLE compute_cap)
    set(cuda_archs_bin)
    if(RUN_RESULT_VAR EQUAL 0)
        string(REGEX REPLACE "[ \t]+" ";" compute_cap "${compute_cap}")
        list(REMOVE_DUPLICATES compute_cap)
        foreach(arch ${compute_cap})
            set(arch_bin)
            if(arch MATCHES "^([0-9]\\.[0-9](\\[0-9]\\.[0-9]\\))?)$")
                set(arch_bin ${CMAKE_MATCH_1})
            else()
                message(FATAL_ERROR "Unknown CUDA arch Name ${arch} !")
            endif()
            if(NOT arch_bin)
                message(FATAL_ERROR "arch_bin was not set !")
            endif()
            list(APPEND cuda_archs_bin ${arch_bin})
        endforeach()
    else()
        message("Failed to detect gpu arch automatically, build a base arch 6.0.")
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_60)
        list(APPEND CUDA_ARCH_LIST sm_60)
    endif()
    # Get build flag from env to choose common/auto build.
    set(NVCC_ARCH_FLAG_FROM_ENV $ENV{CUDA_ARCH})
    if(NVCC_ARCH_FLAG_FROM_ENV STREQUAL "common")
        message("Build common archs for release.")
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=sm_60
                -gencode=arch=compute_61,code=sm_61
                -gencode=arch=compute_70,code=sm_70)
        list(APPEND CUDA_ARCH_LIST sm_60 sm_61 sm_70)
        if(${CUDA_VERSION} VERSION_GREATER "9.5")
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
            list(APPEND CUDA_ARCH_LIST sm_75)
            if(${CUDA_VERSION} VERSION_LESS "11.0")
                list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=compute_75)
            endif()
        endif()
        if(${CUDA_VERSION} VERSION_GREATER "10.5")
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=sm_80)
            list(APPEND CUDA_ARCH_LIST sm_80)
            if(${CUDA_VERSION} VERSION_LESS "11.1")
                list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=compute_80)
            endif()
        endif()
        if(NOT ${CUDA_VERSION} VERSION_LESS "11.1")
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_86,code=compute_86)
        endif()
    elseif(NVCC_ARCH_FLAG_FROM_ENV STREQUAL "auto")
        message("Auto build for arch(s) " ${cuda_archs_bin})
        string(REGEX REPLACE "\\." "" cuda_archs_bin "${cuda_archs_bin}")
        string(REGEX MATCHALL "[0-9()]+" cuda_archs_bin "${cuda_archs_bin}")
        foreach(arch ${cuda_archs_bin})
            list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_${arch},code=sm_${arch})
            list(APPEND CUDA_ARCH_LIST sm_${arch})
        endforeach()
        # For auto build, it only generate the code for indeed arch, so add sm_60 as a default arch
        # to avoid error in different archs. It may increase the compilation time.
        list(APPEND CUDA_NVCC_FLAGS -arch=sm_60)
        list(APPEND CUDA_ARCH_LIST sm_60)
    else()
        message("Only build ptx to speed up compiling cuda ops.")
        set(CUDA_NVCC_FLAGS "CUDA_NVCC_FLAGS")
        list(APPEND CUDA_NVCC_FLAGS -arch=compute_60 -code=compute_60)
    endif()
    list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
    list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
    message("Final CUDA_NVCC_FLASG " ${CUDA_NVCC_FLAGS})

    list(APPEND CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
    if(MSVC AND ${CUDA_VERSION} VERSION_GREATER_EQUAL "11.6")
        list(APPEND CUDA_NVCC_FLAGS -t0)
    endif()
    set(${CUDA_NVCC_FLAGS} ${${CUDA_NVCC_FLAGS}} PARENT_SCOPE)
    set(${CUDA_ARCH_LIST} ${${CUDA_ARCH_LIST}} PARENT_SCOPE)
endfunction()

if(GPU_BACKEND_CUDA)
    find_package(CUDA REQUIRED)
    find_package(Threads)
    # We no longer carry an external CUB dependency. CUDA Toolkit (>= 11) bundles CUB headers,
    # so we require CUDA version strictly greater than 11 (i.e. >= 11.1).
    set(MS_REQUIRE_CUDA_VERSION "11.1")
    if(${CUDA_VERSION} VERSION_LESS ${MS_REQUIRE_CUDA_VERSION})
        message(FATAL_ERROR "The minimum CUDA version ${MS_REQUIRE_CUDA_VERSION} is required, \
              but only CUDA ${CUDA_VERSION} found.")
    endif()
    enable_language(CUDA)
    if(NOT CUDA_PATH OR CUDA_PATH STREQUAL "")
        if(DEFINED ENV{CUDA_HOME} AND NOT $ENV{CUDA_HOME} STREQUAL "")
            set(CUDA_PATH $ENV{CUDA_HOME})
        else()
            set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
        endif()
    endif()

    if(DEFINED ENV{CUDNN_HOME} AND NOT $ENV{CUDNN_HOME} STREQUAL "")
        set(CUDNN_INCLUDE_DIR $ENV{CUDNN_HOME}/include)
        if(WIN32)
            set(CUDNN_LIBRARY_DIR $ENV{CUDNN_HOME}/lib $ENV{CUDNN_HOME}/lib/x64)
        else()
            set(CUDNN_LIBRARY_DIR $ENV{CUDNN_HOME}/lib64)
        endif()
        find_path(CUDNN_INCLUDE_PATH cudnn.h HINTS ${CUDNN_INCLUDE_DIR} NO_DEFAULT_PATH)
        find_library(CUDNN_LIBRARY_PATH "cudnn" HINTS ${CUDNN_LIBRARY_DIR} NO_DEFAULT_PATH)
        if(WIN32)
            find_library(CUBLAS_LIBRARY_PATH "cublas" HINTS ${CUDA_PATH}/lib/x64)
        else()
            find_library(CUBLAS_LIBRARY_PATH "cublas" HINTS ${CUDNN_LIBRARY_DIR})
        endif()
        if(CUDNN_INCLUDE_PATH STREQUAL CUDNN_INCLUDE_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn header file, please set environment variable CUDNN_HOME to \
                    cudnn installation position.")
        endif()
        if(CUDNN_LIBRARY_PATH STREQUAL CUDNN_LIBRARY_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn library file, please set environment variable CUDNN_HOME to \
                    cudnn installation position.")
        endif()
    else()
        list(APPEND CMAKE_PREFIX_PATH ${CUDA_TOOLKIT_ROOT_DIR})
        find_path(CUDNN_INCLUDE_PATH cudnn.h PATH_SUFFIXES cuda/inclulde include cuda)
        find_library(CUDNN_LIBRARY_PATH "cudnn" PATH_SUFFIXES cuda/lib64 lib64 lib cuda/lib lib/x86_64-linux-gnu)
        find_library(CUBLAS_LIBRARY_PATH "cublas" PATH_SUFFIXES cuda/lib64 lib64 lib cuda/lib lib/x86_64-linux-gnu)
        if(CUDNN_INCLUDE_PATH STREQUAL CUDNN_INCLUDE_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn header file, if cudnn library is not installed, please put \
                    cudnn header file in cuda include path or user include path(eg. /usr/local/cuda/include; \
                    /usr/local/include; /usr/include), if cudnn library is installed in other position, please \
                    set environment variable CUDNN_HOME to cudnn installation position, there should be cudnn.h \
                    in {CUDNN_HOME}/include.")
        endif()
        if(CUDNN_LIBRARY_PATH STREQUAL CUDNN_LIBRARY_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn library file, if cudnn library is not installed, please put \
                    cudnn library file in cuda library path or user library path(eg. /usr/local/cuda/lib64; \
                    /usr/local/lib64; /usr/lib64; /usr/local/lib; /usr/lib), if cudnn library is installed in other \
                    position, please set environment variable CUDNN_HOME to cudnn installation position, there should \
                    be cudnn library file in {CUDNN_HOME}/lib64.")
        endif()
    endif()

    if(NOT CUPTI_INCLUDE_DIRS OR CUPTI_INCLUDE_DIRS STREQUAL "")
        set(CUPTI_INCLUDE_DIRS ${CUDA_PATH}/extras/CUPTI/include)
    endif()
    message("CUDA_PATH: ${CUDA_PATH}")
    message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message("CUDNN_INCLUDE_PATH: ${CUDNN_INCLUDE_PATH}")
    message("CUDNN_LIBRARY_PATH: ${CUDNN_LIBRARY_PATH}")
    message("CUBLAS_LIBRARY_PATH: ${CUBLAS_LIBRARY_PATH}")
    message("CUPTI_INCLUDE_DIRS: ${CUPTI_INCLUDE_DIRS}")
    include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})
    ## set NVCC ARCH FLAG and CUDA ARCH LIST
    set(CUDA_NVCC_FLAGS)
    set(CUDA_ARCH_LIST)
    set_nvcc_flag(CUDA_NVCC_FLAGS CUDA_ARCH_LIST)
    if(NOT MSVC)
        add_definitions(-Wno-unknown-pragmas) # Avoid compilation warnings from cuda/thrust
    endif()
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        list(APPEND CUDA_NVCC_FLAGS -G)
        message("CUDA_NVCC_FLAGS" ${CUDA_NVCC_FLAGS})
    endif()
    set(NVCC_TMP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_CXX_FLAGS ${NVCC_TMP_CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_GPU)

    # speed up cuda build with ccache if ccache is found on the environment
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_FOUND} CACHE PATH "CUDA compiler")
    endif()
    message(STATUS "CMAKE_CUDA_COMPILER_LAUNCHER:${CMAKE_CUDA_COMPILER_LAUNCHER}")

    foreach(arch ${CUDA_ARCH_LIST})
        string(APPEND CUDA_ARCH_LIST_STR "${arch} ")
    endforeach()
    message("Final CUDA_ARCH_LIST " ${CUDA_ARCH_LIST})
    add_compile_definitions(CUDA_ARCH_LIST=${CUDA_ARCH_LIST_STR})
endif()
