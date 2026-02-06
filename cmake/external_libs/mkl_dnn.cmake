set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_LDFLAGS "-s")

if(NOT MINDSPORE_PROJECT_DIR)
set(MINDSPORE_PROJECT_DIR ${CMAKE_SOURCE_DIR})
endif()

if(USE_MS_THREADPOOL_FOR_DNNL)
    set(USE_MS_THREADPOOL "-DDNNL_CPU_RUNTIME=THREADPOOL")
else()
    set(USE_MS_THREADPOOL "")
endif()
if(ENABLE_GITEE_EULER)
        message(FATAL_ERROR
            "ENABLE_GITEE_EULER path for oneDNN is not adapted for v3.10.1 yet. "
            "Please disable ENABLE_GITEE_EULER or update the onednn GIT_REPOSITORY/GIT_TAG/SHA256 and extraction logic "
            "to match oneDNN 3.10.1.")
        set(GIT_REPOSITORY "git@gitee.com:src-openeuler/onednn.git")
        set(GIT_TAG "0d726f1")
        set(SHA256 "4d655c0751ee6439584ef5e3d465953fe0c2f4ee2700bc02699bdc1d1572af0d")
    __download_pkg_with_git(ONEDNN ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    set(ONE_DNN_SRC "${CMAKE_BINARY_DIR}/_deps/onednn-src")
    execute_process(COMMAND tar -xf ${ONE_DNN_SRC}/v2.2.tar.gz --strip-components 1 -C ${ONE_DNN_SRC})
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/MKL-DNN/repository/archive/v3.10.1.tar.gz")
    set(SHA256 "e08b045582eec7a16a2d6c93b4551b41e15fb9215d596850ddaf6cffb2d2ac07")
else()
    set(REQ_URL "https://github.com/uxlfoundation/oneDNN/archive/v3.10.1.tar.gz")
    set(SHA256 "d89998bfae2f33c209e216e3d3a8d280ed6eb65a57971dfb4dcfa56beac9571b")
endif()
mindspore_add_pkg(onednn
    VER 3.10.1
    LIBS dnnl
    URL ${REQ_URL}
    SHA256 ${SHA256}
    PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0001-aarch64-matmul-guard-ranges-include.patch
            ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0002-x64-force-ncsp-bnorm-before-jit.patch
            ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0003-x64-force-gemm-ref-conv-bwdw-before-jit-ncsp.patch
    CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF
        ${USE_MS_THREADPOOL} -DDNNL_ENABLE_CONCURRENT_EXEC=ON)

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
