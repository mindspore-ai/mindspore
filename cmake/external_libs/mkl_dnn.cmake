set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    mindspore_add_pkg(onednn
        VER 1.1.1
        LIBS dnnl mkldnn
        HEAD_ONLY ./include
        RELEASE on
        URL https://github.com/oneapi-src/oneDNN/releases/download/v1.1.1/dnnl_win_1.1.1_cpu_vcomp.zip
        MD5 ecaab9ed549643067699c80e5cea1c23)
else()
    mindspore_add_pkg(onednn
        VER 1.1.2
        LIBS dnnl mkldnn
        URL https://github.com/oneapi-src/oneDNN/archive/v1.1.2.tar.gz
        MD5 ab40d52230f3ad1d7a6f06ce0f6bc17a
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_CPU_RUNTIME='SEQ' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF)
endif()

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
