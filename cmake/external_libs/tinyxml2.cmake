set(tinyxml2_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result")
set(tinyxml2_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(tinyxml2
        VER 8.0.0
        LIBS tinyxml2
        URL https://github.com/leethomason/tinyxml2/archive/8.0.0.tar.gz
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release
        MD5 5dc535c8b34ee621fe2128f072d275b5)
include_directories(${tinyxml2_INC})
add_library(mindspore::tinyxml2 ALIAS tinyxml2::tinyxml2)
