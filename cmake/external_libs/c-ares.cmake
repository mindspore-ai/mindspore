mindspore_add_pkg(c-ares
        VER 1.15.0
        LIBS cares 
        URL https://github.com/c-ares/c-ares/releases/download/cares-1_15_0/c-ares-1.15.0.tar.gz
        MD5 d2391da274653f7643270623e822dff7
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
        -DCARES_SHARED:BOOL=OFF
        -DCARES_STATIC:BOOL=ON
        -DCARES_STATIC_PIC:BOOL=ON)

include_directories(${c-ares_INC})
add_library(mindspore::cares ALIAS c-ares::cares)
