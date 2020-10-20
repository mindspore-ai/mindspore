if (ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/nccl/repository/archive/v2.4.8-1.tar.gz")
    set(MD5 "e3078a91635f6ac12927e9fa5a7248ec")
else()
    set(REQ_URL "https://github.com/NVIDIA/nccl/archive/v2.4.8-1.tar.gz")
    set(MD5 "f14b37d6af1c79db5f57cb029a753727")
endif ()

set(nccl_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(nccl
        VER 2.4.8-1
        LIBS nccl
        URL ${REQ_URL}
        MD5 ${MD5}
        BUILD_OPTION src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
        INSTALL_INCS build/include/*
        INSTALL_LIBS build/lib/*)
include_directories(${nccl_INC})
add_library(mindspore::nccl ALIAS nccl::nccl)