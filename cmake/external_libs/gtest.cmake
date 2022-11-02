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
                -DANDROID_STL=${ANDROID_STL}
                ${CMAKE_OPTION})
    endif()
    if(PLATFORM_ARM32)
        set(CMAKE_OPTION -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_NATIVE_API_LEVEL=19
                -DANDROID_NDK=$ENV{ANDROID_NDK}
                -DANDROID_ABI=armeabi-v7a
                -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                -DANDROID_STL=${ANDROID_STL}
                ${CMAKE_OPTION})
    endif()
endif()

if(NOT ENABLE_GLIBCXX)
    set(gtest_CXXFLAGS "${gtest_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.8.1.tar.gz")
    set(SHA256 "9bf1fe5182a604b4135edc1a425ae356c9ad15e9b23f9f12a02e80184c3a249c")
else()
    set(REQ_URL "https://github.com/google/googletest/archive/release-1.8.1.tar.gz")
    set(SHA256 "9bf1fe5182a604b4135edc1a425ae356c9ad15e9b23f9f12a02e80184c3a249c")
endif()

mindspore_add_pkg(gtest
        VER 1.8.1
        LIBS gtest gmock
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION})
include_directories(${gtest_INC})
add_library(mindspore::gtest ALIAS gtest::gtest)
add_library(mindspore::gmock ALIAS gtest::gmock)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${gtest_DIRPATH}/bin/libgtest${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
    file(COPY ${gtest_DIRPATH}/bin/libgtest_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
    file(COPY ${gtest_DIRPATH}/bin/libgmock_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
    file(COPY ${gtest_DIRPATH}/bin/libgmock${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
else()
    file(COPY ${gtest_LIBPATH}/libgtest${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
    file(COPY ${gtest_LIBPATH}/libgtest_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
    file(COPY ${gtest_LIBPATH}/libgmock${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
    file(COPY ${gtest_LIBPATH}/libgmock_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
endif()