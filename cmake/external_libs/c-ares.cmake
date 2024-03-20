if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/c-ares/repository/archive/cares-1_19_1.tar.gz")
    set(SHA256 "9eadec0b34015941abdf3eb6aead694c8d96a192a792131186a7e0a86f2ad6d9")
else()
    set(REQ_URL "https://github.com/c-ares/c-ares/releases/download/cares-1_19_1/c-ares-1.19.1.tar.gz")
    set(SHA256 "321700399b72ed0e037d0074c629e7741f6b2ec2dda92956abe3e9671d3e268e")
endif()

mindspore_add_pkg(c-ares
        VER 1_19_1
        LIBS cares
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON
        -DHAVE_LIBNSL:BOOL=OFF
        PATCHES ${TOP_DIR}/third_party/patch/c-ares/CVE-2024-25629.patch)

include_directories(${c-ares_INC})
add_library(mindspore::cares ALIAS c-ares::cares)
