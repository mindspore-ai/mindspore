if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/ONNX/repository/archive/v1.6.0.tar.gz")
    set(MD5 "1bdbcecdd68ea8392630467646776e02")
else()
    set(REQ_URL "https://github.com/onnx/onnx/releases/download/v1.6.0/onnx-1.6.0.tar.gz")
    set(MD5 "512f2779d6215d4a36f366b6b9acdf1e")
endif()

mindspore_add_pkg(ms_onnx
        VER 1.6.0
        HEAD_ONLY ./
        URL ${REQ_URL}
        MD5 ${MD5})
