
set(nccl_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(nccl
        VER 2.7.6-1
        LIBS nccl
        URL https://github.com/NVIDIA/nccl/archive/v2.7.6-1.tar.gz
        MD5 073b19899f374c5ba07d2db02dc38f9f
        BUILD_OPTION src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
        INSTALL_INCS build/include/*
        INSTALL_LIBS build/lib/*)
include_directories(${nccl_INC})
add_library(mindspore::nccl ALIAS nccl::nccl)