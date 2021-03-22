if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/incubator-tvm/repository/archive/v0.6.0.tar.gz")
    set(MD5 "7b22965745cf1c6208a4e367fb86a585")
else()
    set(REQ_URL
      "https://github.com/apache/incubator-tvm/release/download/v0.6.0/apache-tvm-src-v0.6.0-incubating.tar.gz")
    set(MD5 "2d77a005f0046d937b99c67de82f6438")
endif()
set(incubator_tvm_predict_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(incubator_tvm_predict_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(incubator_tvm_predict
        VER 0.6.0
        HEAD_ONLY ./
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/predict/0001-RetBugFix-CustomRuntime_v06.patch)
include_directories(${incubator_tvm_predict_INC})
add_library(mindspore::incubator_tvm_predict ALIAS incubator_tvm_predict)
