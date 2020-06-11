mindspore_add_pkg(zlib
        VER 1.2.11
        LIBS z
        URL https://github.com/madler/zlib/archive/v1.2.11.tar.gz
        MD5 0095d2d2d1f3442ce1318336637b695f
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release)

include_directories(${zlib_INC})
add_library(mindspore::z ALIAS zlib::z)
