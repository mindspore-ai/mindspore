if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/c-ares/repository/archive/cares-1_15_0.tar.gz")
    set(SHA256 "7deb7872cbd876c29036d5f37e30c4cbc3cc068d59d8b749ef85bb0736649f04")
else()
    set(REQ_URL "https://github.com/c-ares/c-ares/releases/download/cares-1_15_0/c-ares-1.15.0.tar.gz")
    set(SHA256 "6cdb97871f2930530c97deb7cf5c8fa4be5a0b02c7cea6e7c7667672a39d6852")
endif()

mindspore_add_pkg(c-ares
        VER 1.15.0
        LIBS cares
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON
        -DHAVE_LIBNSL:BOOL=OFF
        PATCHES ${TOP_DIR}/third_party/patch/c-ares/CVE-2021-3672.patch)

include_directories(${c-ares_INC})
add_library(mindspore::cares ALIAS c-ares::cares)
