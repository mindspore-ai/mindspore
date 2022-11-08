set(cppjieba_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cppjieba_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/cppjieba/repository/archive/v5.0.3.tar.gz")
    set(SHA256 "c7049e059af2420f9151ecfba5d534b801fced23b48319485d2a9e790a926b72")
else()
    set(REQ_URL "https://github.com/yanyiwu/cppjieba/archive/v5.0.3.tar.gz")
    set(SHA256 "b40848a553dab24d7fcdb6dbdea2486102212baf58466d1c3c3481381af91248")
endif()

mindspore_add_pkg(cppjieba
        VER 5.0.3
        HEAD_ONLY ./
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/cppjieba/cppjieba.patch001)
include_directories(${cppjieba_INC}include)
include_directories(${cppjieba_INC}deps)
add_library(mindspore::cppjieba ALIAS cppjieba)

