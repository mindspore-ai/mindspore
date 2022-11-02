if(MSVC)
    set(tinyxml2_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(tinyxml2_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    if(DEBUG_MODE)
        set(tinyxml2_Debug ON)
    endif()
else()
    set(tinyxml2_CXXFLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result")
    set(tinyxml2_CFLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -O2")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/tinyxml2/repository/archive/8.0.0.tar.gz")
    set(SHA256 "6ce574fbb46751842d23089485ae73d3db12c1b6639cda7721bf3a7ee862012c")
else()
    set(REQ_URL "https://github.com/leethomason/tinyxml2/archive/8.0.0.tar.gz")
    set(SHA256 "6ce574fbb46751842d23089485ae73d3db12c1b6639cda7721bf3a7ee862012c")
endif()


if(NOT WIN32 AND NOT APPLE)
    set(tinyxml2_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

mindspore_add_pkg(tinyxml2
        VER 8.0.0
        LIBS tinyxml2
        URL ${REQ_URL}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release
        SHA256 ${SHA256})
include_directories(${tinyxml2_INC})
add_library(mindspore::tinyxml2 ALIAS tinyxml2::tinyxml2)
