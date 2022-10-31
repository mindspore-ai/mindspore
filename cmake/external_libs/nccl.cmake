if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/nccl/repository/archive/v2.7.6-1.tar.gz")
    set(MD5 "6884364c2b1cf229f0bdaf94efcb5782")
else()
    set(REQ_URL "https://github.com/NVIDIA/nccl/archive/v2.7.6-1.tar.gz")
    set(MD5 "073b19899f374c5ba07d2db02dc38f9f")
endif()

find_package(CUDA REQUIRED)
set(nccl_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

# without -I$ENV{CUDA_HOME}/targets/x86_64-linux/include, cuda_runtime.h will not be found
# "include_directories($ENV{CUDA_HOME}/targets/x86_64-linux/include)" does not help.
# without -fPIC, ld relocation error will be reported.
set(nccl_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -I$ENV{CUDA_HOME}/targets/x86_64-linux/include -fPIC")

mindspore_add_pkg(nccl
        VER 2.7.6-1-${CUDA_VERSION}
        LIBS nccl
        URL ${REQ_URL}
        MD5 ${MD5}
        BUILD_OPTION src.build
        INSTALL_INCS build/include/*
        INSTALL_LIBS build/lib/*)
include_directories(${nccl_INC})
add_library(mindspore::nccl ALIAS nccl::nccl)
