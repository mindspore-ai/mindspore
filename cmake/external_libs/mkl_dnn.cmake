set(mkl_dnn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(mkl_dnn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(mkl_dnn
        VER 1.1.1
        LIBS dnnl mkldnn
        URL https://github.com/intel/mkl-dnn/archive/v1.1.1.tar.gz
        MD5 d6a422b00459600bdc22242590953f38
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_CPU_RUNTIME='SEQ' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF)
include_directories(${mkl_dnn_INC})
add_library(mindspore::dnnl ALIAS mkl_dnn::dnnl)
add_library(mindspore::mkldnn ALIAS mkl_dnn::mkldnn)
