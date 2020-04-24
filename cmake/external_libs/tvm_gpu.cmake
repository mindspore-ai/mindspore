set(incubator_tvm_gpu_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(incubator_tvm_gpu_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(incubator_tvm_gpu
        VER 0.6.0
        LIBS tvm
        URL https://github.com/apache/incubator-tvm/archive/v0.6.0.tar.gz
        MD5 9cbbd32545a776023acabbba270449fe
        CUSTOM_CMAKE ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/
        SUBMODULES ${dlpack_DIRPATH} ${dmlc-core_DIRPATH} ${rang_DIRPATH}
        SOURCEMODULES topi/python/topi python/tvm
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/find_library.patch
        ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/include.patch
        ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/src_pass.patch
        CMAKE_OPTION " ")
add_library(mindspore::tvm ALIAS incubator_tvm_gpu::tvm)