set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    mindspore_add_pkg(onednn
        VER 2.1
        LIBS dnnl mkldnn
        HEAD_ONLY ./include
        RELEASE on
        URL https://github.com/oneapi-src/oneDNN/releases/download/v2.1/dnnl_win_2.1.0_cpu_vcomp.zip
        MD5 b3111c4851dad06f7a796b27083dffa8)
else()
    if(ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/MKL-DNN/repository/archive/v2.1.tar.gz")
        set(MD5 "f4c10ad4197ce2358ad1a917e84c288c")
    else()
        set(REQ_URL "https://github.com/oneapi-src/oneDNN/archive/v2.1.tar.gz")
        set(MD5 "2ed85f2c0c3771a7618db04a9e08ae57")
    endif()
    mindspore_add_pkg(onednn
        VER 2.1
        LIBS dnnl mkldnn
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF)
endif()

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
