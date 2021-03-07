set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS} -Dgoogle=mindspore_private")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
if(NOT ENABLE_GLIBCXX)
    set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
if(BUILD_LITE)
    set(glog_patch "")
    set(glog_lib glog)
else()
    set(glog_patch ${CMAKE_SOURCE_DIR}/third_party/patch/glog/glog.patch001)
    set(glog_lib mindspore_glog)
endif()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.4.0.tar.gz")
    set(MD5 "22fe340ddc231e6c8e46bc295320f8ee")
else()
    set(REQ_URL "https://github.com/google/glog/archive/v0.4.0.tar.gz")
    set(MD5 "0daea8785e6df922d7887755c3d100d0")
endif()

mindspore_add_pkg(glog
        VER 0.4.0
        LIBS ${glog_lib}
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${glog_patch}
        CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=OFF)
include_directories(${glog_INC})
add_library(mindspore::glog ALIAS glog::${glog_lib})
