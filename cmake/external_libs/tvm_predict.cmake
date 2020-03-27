set(incubator_tvm_predict_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(incubator_tvm_predict_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(incubator_tvm_predict
        VER 0.6.0
        HEAD_ONLY ./
        URL https://github.com/apache/incubator-tvm/release/download/v0.6.0/apache-tvm-src-v0.6.0-incubating.tar.gz
        MD5 2d77a005f0046d937b99c67de82f6438
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/predict/0001-RetBugFix-CustomRuntime_v06.patch)
include_directories(${incubator_tvm_predict_INC})
add_library(mindspore::incubator_tvm_predict ALIAS incubator_tvm_predict)
