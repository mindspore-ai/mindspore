set(REQ_URL "https://gitee.com/mindspore/akg/archive/refs/tags/v2.4.1.tar.gz")
set(SHA256 "05971f219a525d601c600a6cea994480984fc130b19391079101f756dbab61a2")

message("USE_LLVM: ${USE_LLVM}")
set(akg_cmake_option -DUSE_LLVM=${USE_LLVM} -DENABLE_AKG=ON)

if(USE_CUDA)
    set(akg_cmake_option -DUSE_CUDA=ON ${akg_cmake_option})
endif()


mindspore_add_pkg(akg
        LIBS akg
        VER 2.4.0
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${akg_cmake_option}
        )

add_library(mindspore::akg ALIAS akg::akg)