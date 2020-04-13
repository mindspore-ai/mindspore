set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(onednn
        VER 1.1.2
        LIBS dnnl mkldnn
        URL https://github.com/oneapi-src/oneDNN/archive/v1.1.2.tar.gz
        MD5 ab40d52230f3ad1d7a6f06ce0f6bc17a
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_CPU_RUNTIME='SEQ' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF)
include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
