if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/OpenSSL_1_1_1k.tar.gz")
  set(MD5 "bdd51a68ad74618dd2519da8e0bcc759")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1k.tar.gz")
  set(MD5 "bdd51a68ad74618dd2519da8e0bcc759")
endif()

if(BUILD_LITE)
  set(OPENSSL_PATCH_ROOT ${TOP_DIR}/third_party/patch/openssl)
else()
  set(OPENSSL_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/openssl)
endif()

if(BUILD_LITE)
    if(PLATFORM_ARM64 AND ANDROID_NDK_TOOLCHAIN_INCLUDED)
        set(openssl_USE_STATIC_LIBS OFF)
        set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK})
        set(PATH
            ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:
            ${ANDROID_NDK_ROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:
            $ENV{PATH})
        mindspore_add_pkg(openssl
                VER 1.1.1k
                LIBS ssl crypto
                URL ${REQ_URL}
                MD5 ${MD5}
                CONFIGURE_COMMAND ./Configure android-arm64 -D__ANDROID_API__=29 no-zlib
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3711.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3712.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-4160.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-0778.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-1292.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2068.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2097.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4304.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4450.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0215.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0286.patch
                )
    elseif(PLATFORM_ARM32 AND ANDROID_NDK_TOOLCHAIN_INCLUDED)
        set(openssl_USE_STATIC_LIBS OFF)
        set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK})
        set(PATH
            ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:
            ${ANDROID_NDK_ROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:
            $ENV{PATH})
        mindspore_add_pkg(openssl
                VER 1.1.1k
                LIBS ssl crypto
                URL ${REQ_URL}
                MD5 ${MD5}
                CONFIGURE_COMMAND ./Configure android-arm -D__ANDROID_API__=19 no-zlib
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3711.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3712.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-4160.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-0778.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-1292.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2068.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2097.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4304.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4450.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0215.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0286.patch
                )
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
        set(openssl_CFLAGS -fvisibility=hidden)
        mindspore_add_pkg(openssl
                VER 1.1.1k
                LIBS ssl crypto
                URL ${REQ_URL}
                MD5 ${MD5}
                CONFIGURE_COMMAND ./config no-zlib no-shared
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3711.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3712.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-4160.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-0778.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-1292.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2068.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2097.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4304.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4450.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0215.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0286.patch
                )
    else()
        MESSAGE(FATAL_ERROR "openssl does not support compilation for the current environment.")
    endif()
    include_directories(${openssl_INC})
    add_library(mindspore::ssl ALIAS openssl::ssl)
    add_library(mindspore::crypto ALIAS openssl::crypto)
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
        set(openssl_CFLAGS -fvisibility=hidden)
        mindspore_add_pkg(openssl
                VER 1.1.1k
                LIBS ssl crypto
                URL ${REQ_URL}
                MD5 ${MD5}
                CONFIGURE_COMMAND ./config no-zlib no-shared
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3711.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3712.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-4160.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-0778.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-1292.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2068.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-2097.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4304.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2022-4450.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0215.patch
                PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2023-0286.patch
                )
        include_directories(${openssl_INC})
        add_library(mindspore::ssl ALIAS openssl::ssl)
        add_library(mindspore::crypto ALIAS openssl::crypto)
    endif()
endif()