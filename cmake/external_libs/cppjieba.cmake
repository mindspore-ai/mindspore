set(cppjieba_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cppjieba_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(cppjieba
        VER 5.0.3
        HEAD_ONLY ./
        URL https://codeload.github.com/yanyiwu/cppjieba/tar.gz/v5.0.3
        MD5 b8b3f7a73032c9ce9daafa4f67196c8c
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/cppjieba/cppjieba.patch001)
include_directories(${cppjieba_INC}include)
include_directories(${cppjieba_INC}deps)
add_library(mindspore::cppjieba ALIAS cppjieba)