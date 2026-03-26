if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/re2/repository/archive/2022-04-01.tar.gz")
    set(SHA256 "1ae8ccfdb1066a731bba6ee0881baad5efd2cd661acd9569b689f2586e1a50e9")
else()
    set(REQ_URL "https://github.com/google/re2/archive/2022-04-01.tar.gz")
    set(SHA256 "1ae8ccfdb1066a731bba6ee0881baad5efd2cd661acd9569b689f2586e1a50e9")
endif()

if(NOT ENABLE_GLIBCXX)
    set(re2_CXXFLAGS "${re2_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mindspore_add_pkg(re2
        VER 20220401
        LIBS re2
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE)

include_directories(${re2_INC})
add_library(mindspore::re2 ALIAS re2::re2)
