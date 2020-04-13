set(incubator_tvm_gpu_CFLAGS "-pipe -Wall -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
set(incubator_tvm_gpu_CXXFLAGS "-std=c++11 -pipe -Wall -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
set(USE_CUDA "ON")
mindspore_add_pkg(incubator_tvm_gpu
        VER 0.6.0
        LIBS tvm
        URL https://github.com/apache/incubator-tvm/archive/v0.6.0.tar.gz
        MD5 9cbbd32545a776023acabbba270449fe
        SUBMODULES ${dlpack_DIRPATH} ${dmlc-core_DIRPATH} ${rang_DIRPATH}
        SOURCEMODULES topi/python/topi python/tvm
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/find_library.patch
                ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/include.patch
                ${CMAKE_SOURCE_DIR}/third_party/patch/incubator-tvm/src_pass.patch
        CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON)
include_directories(${incubator_tvm_gpu_INC})
add_library(mindspore::tvm ALIAS incubator_tvm_gpu::tvm)
