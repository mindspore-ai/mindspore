if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/re2/repository/archive/2019-12-01.tar.gz")
    set(SHA256 "7268e1b4254d9ffa5ccf010fee954150dbb788fd9705234442e7d9f0ee5a42d3")
else()
    set(REQ_URL "https://github.com/google/re2/archive/2019-12-01.tar.gz")
    set(SHA256 "7268e1b4254d9ffa5ccf010fee954150dbb788fd9705234442e7d9f0ee5a42d3")
endif()

if(NOT ENABLE_GLIBCXX)
    set(re2_CXXFLAGS "${re2_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mindspore_add_pkg(re2
        VER 20191201
        LIBS re2
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE)

include_directories(${re2_INC})
add_library(mindspore::re2 ALIAS re2::re2)

