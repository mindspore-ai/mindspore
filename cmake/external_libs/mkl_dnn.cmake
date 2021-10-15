set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    mindspore_add_pkg(onednn
        VER 2.2
        LIBS dnnl mkldnn
        HEAD_ONLY ./include
        RELEASE on
        URL https://github.com/oneapi-src/oneDNN/releases/download/v2.2/dnnl_win_2.2.0_cpu_vcomp.zip
        MD5 fa12c693b2ec07700d174e1e99d60a7e)
else()
    if(ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/MKL-DNN/repository/archive/v2.2.tar.gz")
        set(MD5 "ac34c03a0ff31eb88dfe805967b9c351")
    else()
        set(REQ_URL "https://github.com/oneapi-src/oneDNN/archive/v2.2.tar.gz")
        set(MD5 "6a062e36ea1bee03ff55bf44ee243e27")
    endif()
    mindspore_add_pkg(onednn
        VER 2.2
        LIBS dnnl mkldnn
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF
            -DDNNL_ENABLE_CONCURRENT_EXEC=ON)
endif()

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
