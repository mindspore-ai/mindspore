if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/zlib/repository/archive/v1.2.11.tar.gz")
    set(MD5 "be6d144068d8835e86a81b3f36b66a42")
else()
    set(REQ_URL "https://github.com/madler/zlib/archive/v1.2.11.tar.gz")
    set(MD5 "0095d2d2d1f3442ce1318336637b695f")
endif()

mindspore_add_pkg(zlib
        VER 1.2.11
        LIBS z
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release)

include_directories(${zlib_INC})
add_library(mindspore::z ALIAS zlib::z)
