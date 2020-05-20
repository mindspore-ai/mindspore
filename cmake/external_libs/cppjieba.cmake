set(cppjieba_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cppjieba_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(cppjieba
        VER 5.0.3
        HEAD_ONLY ./
        URL https://codeload.github.com/yanyiwu/cppjieba/zip/v5.0.3
        MD5 0dfef44bd32328c221f128b401e1a45c
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/cppjieba/cppjieba.patch001)
include_directories(${cppjieba_INC}include)
include_directories(${cppjieba_INC}deps)
add_library(mindspore::cppjieba ALIAS cppjieba)