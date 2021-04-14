set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    mindspore_add_pkg(onednn
        VER 1.6
        LIBS dnnl mkldnn
        HEAD_ONLY ./include
        RELEASE on
        URL https://github.com/oneapi-src/oneDNN/releases/download/v1.6/dnnl_win_1.6.0_cpu_vcomp.zip
        MD5 fe660e34e9f73ab13a65987819a0712e)
else()
    if(ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/MKL-DNN/repository/archive/v2.1.2.tar.gz")
        set(MD5 "d98f171d7e66e252c79e2e167ba4a8e8")
    else()
        set(REQ_URL "https://github.com/oneapi-src/oneDNN/archive/v2.1.2.tar.gz")
        set(MD5 "1df4f16f650b7ea08610a10af013faa3")
    endif()
    mindspore_add_pkg(onednn
        VER 2.1.2
        LIBS dnnl mkldnn
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF)
endif()

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
