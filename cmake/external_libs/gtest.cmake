set(gtest_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(gtest_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

set(CMAKE_OPTION
        -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON
        -DCMAKE_MACOSX_RPATH=TRUE -Dgtest_disable_pthreads=ON)
if(BUILD_LITE)
    if(PLATFORM_ARM64)
        set(CMAKE_OPTION -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_NATIVE_API_LEVEL=19
                -DANDROID_NDK=$ENV{ANDROID_NDK}
                -DANDROID_ABI=arm64-v8a
                -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                ${CMAKE_OPTION})
    endif()
    if(PLATFORM_ARM32)
        set(CMAKE_OPTION -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_NATIVE_API_LEVEL=19
                -DANDROID_NDK=$ENV{ANDROID_NDK}
                -DANDROID_ABI=armeabi-v7a
                -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                ${CMAKE_OPTION})
    endif()
else()
    set(gtest_CXXFLAGS "${gtest_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.8.0.tar.gz")
    set(MD5 "89e13ca1aa48d370719d58010b83f62c")
else()
    set(REQ_URL "https://github.com/google/googletest/archive/release-1.8.0.tar.gz")
    set(MD5 "16877098823401d1bf2ed7891d7dce36")
endif()

mindspore_add_pkg(gtest
        VER 1.8.0
        LIBS gtest
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION ${CMAKE_OPTION})
include_directories(${gtest_INC})
add_library(mindspore::gtest ALIAS gtest::gtest)
file(COPY ${gtest_LIBPATH}/libgtest${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
file(COPY ${gtest_LIBPATH}/libgtest_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
