set(onednn_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(onednn_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(USE_MS_THREADPOOL_FOR_DNNL)
    set(USE_MS_THREADPOOL "-DDNNL_CPU_RUNTIME=THREADPOOL")
else()
    set(USE_MS_THREADPOOL "")
endif()
if(ENABLE_GITEE_EULER)
        set(GIT_REPOSITORY "git@gitee.com:src-openeuler/onednn.git")
        set(GIT_TAG "0d726f1")
        set(MD5 "6a062e36ea1bee03ff55bf44ee243e27")
    __download_pkg_with_git(ONEDNN ${GIT_REPOSITORY} ${GIT_TAG} ${MD5})
    set(ONE_DNN_SRC "${TOP_DIR}/mindspore/lite/build/_deps/onednn-src")
    execute_process(COMMAND tar -xf ${ONE_DNN_SRC}/v2.2.tar.gz --strip-components 1 -C ${ONE_DNN_SRC})
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    mindspore_add_pkg(onednn
        VER 2.2
        LIBS dnnl mkldnn
        HEAD_ONLY ./include
        RELEASE on
        URL http://tools.mindspore.cn/libs/dnnl/dnnl_win_2.2.0_cpu_vcomp.zip
        MD5 139fcdbd601a970fb86dd15b30ba5ae3)
else()
    if(ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/MKL-DNN/repository/archive/v2.2.tar.gz")
        set(MD5 "49c650e0cc24ef9ae7033d4cb22ebfad")
    else()
        set(REQ_URL "https://github.com/oneapi-src/oneDNN/archive/v2.2.tar.gz")
        set(MD5 "6a062e36ea1bee03ff55bf44ee243e27")
    endif()
    mindspore_add_pkg(onednn
        VER 2.2
        LIBS dnnl mkldnn
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/onednn/0001-fix-user-threadpool-bug.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/onednn/0002-fix-pool-nthr-bug.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/onednn/0003-fix-zero-threads-identified-on-AMD.patch
        CMAKE_OPTION -DDNNL_ARCH_OPT_FLAGS='' -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_TESTS=OFF
            ${USE_MS_THREADPOOL} -DDNNL_ENABLE_CONCURRENT_EXEC=ON)
endif()

include_directories(${onednn_INC})
add_library(mindspore::dnnl ALIAS onednn::dnnl)
add_library(mindspore::mkldnn ALIAS onednn::mkldnn)
