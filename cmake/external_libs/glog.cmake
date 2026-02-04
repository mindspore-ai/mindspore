if(MSVC)
    # add "/EHsc", for vs2019 warning C4530 about glog
    set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS} -Dgoogle=mindspore_private /EHsc")
    if(DEBUG_MODE)
        set(glog_Debug ON)
    endif()
else()
    set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS} -Dgoogle=mindspore_private")
endif()
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(glog_lib glog)

if(NOT ENABLE_GLIBCXX)
    set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.7.1.tar.gz")
    set(SHA256 "54854d52a4a0f12a7a57f43d22457477281ef373b6487c5ac422e6303d7ff3e8")
else()
    set(REQ_URL "https://github.com/google/glog/archive/v0.7.1.tar.gz")
    set(SHA256 "00e4a87e87b7e7612f519a41e491f16623b12423620006f59f5688bfd8d13b08")
endif()

set(glog_option -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON -DWITH_GFLAGS=OFF
        -DCMAKE_BUILD_TYPE=Release)

if(WIN32 AND NOT MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(glog_option ${glog_option} -DHAVE_DBGHELP=ON)
    endif()
endif()

if(ANDROID_NDK)
    if(PLATFORM_ARM64)
        set(glog_option -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_NATIVE_API_LEVEL=19
                -DANDROID_NDK=$ENV{ANDROID_NDK}
                -DANDROID_ABI=arm64-v8a
                -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                -DANDROID_STL=${ANDROID_STR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} ${glog_option})
    elseif(PLATFORM_ARM32)
        set(glog_option -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_NATIVE_API_LEVEL=19
                -DANDROID_NDK=$ENV{ANDROID_NDK}
                -DANDROID_ABI=armeabi-v7a
                -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                -DANDROID_STL=${ANDROID_STR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} ${glog_option})
    endif()
endif()

mindspore_add_pkg(glog
        VER 0.7.1
        LIBS ${glog_lib}
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${glog_option})
include_directories(${glog_INC})
add_library(mindspore::glog ALIAS glog::${glog_lib})
