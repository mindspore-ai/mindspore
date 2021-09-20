if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/re2/repository/archive/2019-12-01.tar.gz")
    set(MD5 "f581b6356d8e8e3debfc230197f91b54")
else()
    set(REQ_URL "https://github.com/google/re2/archive/2019-12-01.tar.gz")
    set(MD5 "527eab0c75d6a1a0044c6eefd816b2fb")
endif()

if(NOT ENABLE_GLIBCXX)
    set(re2_CXXFLAGS "${re2_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mindspore_add_pkg(re2
        VER 20191201
        LIBS re2
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE)

include_directories(${re2_INC})
add_library(mindspore::re2 ALIAS re2::re2)