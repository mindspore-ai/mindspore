
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/libjpeg-turbo/repository/archive/2.0.4.tar.gz")
    set(MD5 "44c43e4a9fb352f47090804529317c88")
else()
    set(REQ_URL "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.4.tar.gz")
    set(MD5 "44c43e4a9fb352f47090804529317c88")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 \
        -O2")
else()
    if(MSVC)
        set(jpeg_turbo_CFLAGS "-O2")
    else()
        set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC \
            -D_FORTIFY_SOURCE=2 -O2")
    endif()
endif()

set(jpeg_turbo_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack,-s")


set(jpeg_turbo_USE_STATIC_LIBS ON)
set(JPEG_TURBO_PATCHE ${CMAKE_SOURCE_DIR}/third_party/patch/jpeg_turbo/jpeg_turbo.patch001)
set(CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_RPATH=TRUE -DWITH_SIMD=ON)
set(CVE_2020_35538 ${CMAKE_SOURCE_DIR}/third_party/patch/jpeg_turbo/CVE-2020-35538.patch)
set(CVE_2021_46822 ${CMAKE_SOURCE_DIR}/third_party/patch/jpeg_turbo/CVE-2021-46822.patch)
if(BUILD_LITE)
    set(jpeg_turbo_USE_STATIC_LIBS OFF)
    set(JPEG_TURBO_PATCHE ${TOP_DIR}/third_party/patch/jpeg_turbo/jpeg_turbo.patch001)
    set(CVE_2020_35538 ${TOP_DIR}/third_party/patch/jpeg_turbo/CVE-2020-35538.patch)
    set(CVE_2021_46822 ${TOP_DIR}/third_party/patch/jpeg_turbo/CVE-2021-46822.patch)
    if(ANDROID_NDK)  #  compile android on x86_64 env
        if(PLATFORM_ARM64)
            set(CMAKE_OPTION  -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                              -DANDROID_NATIVE_API_LEVEL=19
                              -DANDROID_NDK=$ENV{ANDROID_NDK}
                              -DANDROID_ABI=arm64-v8a
                              -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                              -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()
        if(PLATFORM_ARM32)
            set(CMAKE_OPTION  -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                              -DANDROID_NATIVE_API_LEVEL=19
                              -DANDROID_NDK=$ENV{ANDROID_NDK}
                              -DANDROID_ABI=armeabi-v7a
                              -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                              -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()
    endif()
endif()

mindspore_add_pkg(jpeg_turbo
        VER 2.0.4
        LIBS jpeg turbojpeg
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION ${CMAKE_OPTION}
        PATCHES ${JPEG_TURBO_PATCHE}
        PATCHES ${CVE_2020_35538}
        PATCHES ${CVE_2021_46822}
        )
include_directories(${jpeg_turbo_INC})
add_library(mindspore::jpeg_turbo ALIAS jpeg_turbo::jpeg)
add_library(mindspore::turbojpeg ALIAS jpeg_turbo::turbojpeg)
