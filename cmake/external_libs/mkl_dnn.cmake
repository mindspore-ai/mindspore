set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    mindspore_add_pkg(onednn
        VER 1.5
        LIBS dnnl mkldnn
        HEAD_ONLY ./include
        RELEASE on
        URL https://github.com/oneapi-src/oneDNN/releases/download/v1.5/dnnl_win_1.5.0_cpu_vcomp.zip
        MD5 17757c84f49edd42d34ae8c9288110a1)
else()
    mindspore_add_pkg(onednn
        VER 1.5
        LIBS dnnl mkldnn
        URL https://github.com/oneapi-src/oneDNN/archive/v1.5.tar.gz
        MD5 5d97e0e8f4c0b37da5f524533b7a644b
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_CPU_RUNTIME='SEQ' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF)
endif()

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
