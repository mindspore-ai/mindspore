if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/nccl/repository/archive/v2.7.6-1.tar.gz")
    set(SHA256 "82f2eb496aa125414849868704579a1b89b88f1c0db5c9728ac84be0a9ed2a04")
else()
    set(REQ_URL "https://github.com/NVIDIA/nccl/archive/v2.7.6-1.tar.gz")
    set(SHA256 "60dd9b1743c2db6c05f60959edf98a4477f218115ef910d7ec2662f2fb5cf626")
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
        SHA256 ${SHA256}
        BUILD_OPTION src.build
        INSTALL_INCS build/include/*
        INSTALL_LIBS build/lib/*)
include_directories(${nccl_INC})
add_library(mindspore::nccl ALIAS nccl::nccl)
