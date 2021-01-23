set(cppjieba_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cppjieba_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/cppjieba/repository/archive/v5.0.3.tar.gz")
    set(MD5 "ea0bdd5a654a376e2c2077daae23b376")
else()
    set(REQ_URL "https://github.com/yanyiwu/cppjieba/archive/v5.0.3.tar.gz")
    set(MD5 "b8b3f7a73032c9ce9daafa4f67196c8c")
endif()

mindspore_add_pkg(cppjieba
        VER 5.0.3
        HEAD_ONLY ./
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/cppjieba/cppjieba.patch001)
include_directories(${cppjieba_INC}include)
include_directories(${cppjieba_INC}deps)
add_library(mindspore::cppjieba ALIAS cppjieba)