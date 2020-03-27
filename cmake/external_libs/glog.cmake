set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS}")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(glog
        VER 0.4.0
        LIBS glog
        URL https://github.com/google/glog/archive/v0.4.0.tar.gz
        MD5 0daea8785e6df922d7887755c3d100d0
        CMAKE_OPTION -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON)
include_directories(${glog_INC})
add_library(mindspore::glog ALIAS glog::glog)
