set(incubator_tvm_gpu_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(incubator_tvm_gpu_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(incubator_tvm_gpu
        VER 0.6.0
        HEAD_ONLY ./
        URL https://github.com/apache/incubator-tvm/archive/v0.6.0.tar.gz
        MD5 9cbbd32545a776023acabbba270449fe)

