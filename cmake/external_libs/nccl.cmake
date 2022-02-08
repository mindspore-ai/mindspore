if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/nccl/repository/archive/v2.7.6-1.tar.gz")
    set(MD5 "6884364c2b1cf229f0bdaf94efcb5782")
else()
    set(REQ_URL "https://github.com/NVIDIA/nccl/archive/v2.7.6-1.tar.gz")
    set(MD5 "073b19899f374c5ba07d2db02dc38f9f")
endif()

set(nccl_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(nccl
        VER 2.7.6-1
        LIBS nccl
        URL ${REQ_URL}
        MD5 ${MD5}
        BUILD_OPTION src.build
        INSTALL_INCS build/include/*
        INSTALL_LIBS build/lib/*)
include_directories(${nccl_INC})
add_library(mindspore::nccl ALIAS nccl::nccl)
