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
        set(GIT_REPOSITORY "git@gitee.com:src-openeuler/onednn.git")
        set(GIT_TAG "0d726f1")
        set(SHA256 "4d655c0751ee6439584ef5e3d465953fe0c2f4ee2700bc02699bdc1d1572af0d")
    __download_pkg_with_git(ONEDNN ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    set(ONE_DNN_SRC "${CMAKE_BINARY_DIR}/_deps/onednn-src")
    execute_process(COMMAND tar -xf ${ONE_DNN_SRC}/v2.2.tar.gz --strip-components 1 -C ${ONE_DNN_SRC})
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/MKL-DNN/repository/archive/v2.2.tar.gz")
    set(SHA256 "2e809b11727af9d10784a5481b445a14387297161b5cc7f9c969c57fe40752bc")
else()
    set(REQ_URL "https://github.com/oneapi-src/oneDNN/archive/v2.2.tar.gz")
    set(SHA256 "4d655c0751ee6439584ef5e3d465953fe0c2f4ee2700bc02699bdc1d1572af0d")
endif()
mindspore_add_pkg(onednn
    VER 2.2
    LIBS dnnl mkldnn
    URL ${REQ_URL}
    SHA256 ${SHA256}
    PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0001-fix-user-threadpool-bug.patch
    PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0002-fix-pool-nthr-bug.patch
    PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0003-fix-zero-threads-identified-on-AMD.patch
    PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/onednn/0004-fix-dnnl-limits.patch
    CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF
        ${USE_MS_THREADPOOL} -DDNNL_ENABLE_CONCURRENT_EXEC=ON)

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
