if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/nccl/repository/archive/v2.16.5-1.tar.gz")
    set(SHA256 "be162e5dee73be833cbec0a8e0a8edef6e4c48b71814efc09b7a2233599ab5eb")
else()
    set(REQ_URL "https://github.com/NVIDIA/nccl/archive/v2.16.5-1.tar.gz")
    set(SHA256 "0e3d7b6295beed81dc15002e88abf7a3b45b5c686b13b779ceac056f5612087f")
endif()

find_package(CUDA REQUIRED)
set(nccl_CFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fPIC -fstack-protector-all")

# without -I$ENV{CUDA_HOME}/targets/x86_64-linux/include, cuda_runtime.h will not be found
# "include_directories($ENV{CUDA_HOME}/targets/x86_64-linux/include)" does not help.
# without -fPIC, ld relocation error will be reported.
set(nccl_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 \
                    -I$ENV{CUDA_HOME}/targets/x86_64-linux/include -fPIC  -fstack-protector-all")
set(ENV{LDFLAGS} "-Wl,-z,relro,-z,now,-z,noexecstack,-s")

if(NOT BUILD_LITE)
    enable_language(CUDA)
    if(NOT CUDA_PATH OR CUDA_PATH STREQUAL "")
        if(DEFINED ENV{CUDA_HOME} AND NOT $ENV{CUDA_HOME} STREQUAL "")
            set(CUDA_PATH $ENV{CUDA_HOME})
        else()
            set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
        endif()
    endif()
    ## Function for setting NVCC flag
    function(set_nccl_arch NCCL_ARCH)
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
            message("Failed to detect gpu arch automatically.")
            list(APPEND ARCH_FLAGS "-arch=sm_60")
        endif()
        # Get build flag from env to choose common/auto build.
        set(NVCC_ARCH_FLAG_FROM_ENV $ENV{CUDA_ARCH})
        if(NVCC_ARCH_FLAG_FROM_ENV STREQUAL "common")
            message("Build common archs for release.")
            list(APPEND ARCH_FLAGS "-gencode=arch=compute_60,code=sm_60")
            list(APPEND ARCH_FLAGS "-gencode=arch=compute_61,code=sm_61")
            list(APPEND ARCH_FLAGS "-gencode=arch=compute_70,code=sm_70")
            if(${CUDA_VERSION} VERSION_GREATER "9.5")
                list(APPEND ARCH_FLAGS "-gencode=arch=compute_75,code=sm_75")
                if(${CUDA_VERSION} VERSION_LESS "11.0")
                    list(APPEND ARCH_FLAGS "-gencode=arch=compute_75,code=compute_75")
                endif()
            endif()
            if(${CUDA_VERSION} VERSION_GREATER "10.5")
                list(APPEND ARCH_FLAGS "-gencode=arch=compute_80,code=sm_80")
                if(${CUDA_VERSION} VERSION_LESS "11.1")
                    list(APPEND ARCH_FLAGS "-gencode=arch=compute_80,code=compute_80")
                endif()
            endif()
            if(NOT ${CUDA_VERSION} VERSION_LESS "11.1")
                list(APPEND ARCH_FLAGS "-gencode=arch=compute_86,code=compute_86")
            endif()
        else()
            message("Auto build for arch(s) " ${cuda_archs_bin})
            string(REGEX REPLACE "\\." "" cuda_archs_bin "${cuda_archs_bin}")
            string(REGEX MATCHALL "[0-9()]+" cuda_archs_bin "${cuda_archs_bin}")
            foreach(arch ${cuda_archs_bin})
                list(APPEND ARCH_FLAGS -gencode=arch=compute_${arch},code=sm_${arch})
            endforeach()
            list(APPEND ARCH_FLAGS "-arch=sm_60")
        endif()
        list(REMOVE_DUPLICATES ARCH_FLAGS)

        # Convert to string.
        list(LENGTH ARCH_FLAGS arch_flags_length)
        MATH(EXPR arch_flags_length "${arch_flags_length}-1")
        foreach(index RANGE ${arch_flags_length})
            list(GET ARCH_FLAGS ${index} item)
            if(${index} EQUAL 0)
                set(ARCH_FLAGS_STR "${item}")
            else()
                list(APPEND ARCH_FLAGS_STR " ${item}")
            endif()
        endforeach()
        message("Final NCCL_ARCH_FLAGS " ${ARCH_FLAGS_STR})
        set(${NCCL_ARCH} ${ARCH_FLAGS_STR} PARENT_SCOPE)
    endfunction()
    set_nccl_arch(NCCL_ARCH_FLAG)
else()
    set(NCCL_ARCH_FLAG "-arch=sm_60")
endif()

mindspore_add_pkg(nccl
        VER 2.16.5-1-${CUDA_VERSION}
        LIBS nccl
        URL ${REQ_URL}
        SHA256 ${SHA256}
        BUILD_OPTION src.build NVCC_GENCODE=""${NCCL_ARCH_FLAG}""
        INSTALL_INCS build/include/*
        INSTALL_LIBS build/lib/*)
set(ENV{LDFLAGS} "")
include_directories(${nccl_INC})
add_library(mindspore::nccl ALIAS nccl::nccl)
