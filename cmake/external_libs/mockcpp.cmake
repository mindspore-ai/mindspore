set(mockcpp_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(mockcpp_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

set(CMAKE_OPTION
        -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON
        -DCMAKE_MACOSX_RPATH=TRUE)
if(BUILD_LITE)
    if(PLATFORM_ARM64 AND CMAKE_SYSTEM_NAME MATCHES "Android")
        set(CMAKE_OPTION -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                -DANDROID_NATIVE_API_LEVEL=19
                -DANDROID_NDK=$ENV{ANDROID_NDK}
                -DANDROID_ABI=arm64-v8a
                -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                -DANDROID_STL=${ANDROID_STL}
                ${CMAKE_OPTION})
    endif()
    if(PLATFORM_ARM32 AND CMAKE_SYSTEM_NAME MATCHES "Android")
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
    set(mockcpp_CXXFLAGS "${mockcpp_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(BUILD_LITE)
  set(MOCKCPP_PATCH_ROOT ${TOP_DIR}/third_party/patch/mockcpp)
else()
  set(MOCKCPP_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/mockcpp)
endif()

# No Gitee mirror repo yet. just use Github repo
set(REQ_URL "https://github.com/sinojelly/mockcpp/archive/refs/tags/v2.7.tar.gz")
set(SHA256 "73ab0a8b6d1052361c2cebd85e022c0396f928d2e077bf132790ae3be766f603")

mindspore_add_pkg(mockcpp
        VER 2.7
        LIBS mockcpp
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION}
        PATCHES ${MOCKCPP_PATCH_ROOT}/mockcpp_support_arm64.patch)

include_directories(${mockcpp_INC})
add_library(mindspore::mockcpp ALIAS mockcpp::mockcpp)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${mockcpp_DIRPATH}/lib/libmockcpp${CMAKE_STATIC_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/mockcpp/ FOLLOW_SYMLINK_CHAIN)
else()
    file(COPY ${mockcpp_DIRPATH}/lib/libmockcpp${CMAKE_STATIC_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/mockcpp/ FOLLOW_SYMLINK_CHAIN)
endif()