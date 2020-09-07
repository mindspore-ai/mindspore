#!/usr/bin/env bash

set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
TOP_PATH="${BASE_PATH}/../../.."

get_version() {
    VERSION_MAJOR=`grep "const int ms_version_major =" ../include/version.h | tr -dc "[0-9]"`
    VERSION_MINOR=`grep "const int ms_version_minor =" ../include/version.h | tr -dc "[0-9]"`
    VERSION_REVISION=`grep "const int ms_version_revision =" ../include/version.h | tr -dc "[0-9]"`
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}

build_mslite_arm64() {
    # build mindspore-lite arm64
    cd ${TOP_PATH}
    bash build.sh -I arm64
    COMPILE_RET=$?

    if [[ "${COMPILE_RET}" -ne 0 ]]; then
        echo "---------------- mindspore lite arm64: build failed ----------------"
        exit
    fi
    # copy arm64 so
    cd ${TOP_PATH}/output/
    rm -rf mindspore-lite-${VERSION_STR}-runtime-arm64-cpu
    tar -zxvf mindspore-lite-${VERSION_STR}-runtime-arm64-cpu.tar.gz
    mkdir -p ${BASE_PATH}/java/app/libs/arm64-v8a/
    rm -rf ${BASE_PATH}/java/app/libs/arm64-v8a/*
    cp ${TOP_PATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm64-cpu/lib/libmindspore-lite.so ${BASE_PATH}/java/app/libs/arm64-v8a/
    cp ${TOP_PATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm64-cpu/lib/liboptimize.so ${BASE_PATH}/java/app/libs/arm64-v8a/
    cp ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so ${BASE_PATH}/java/app/libs/arm64-v8a/
}

build_mslite_arm32() {
    # build mindspore-lite arm64
    cd ${TOP_PATH}
    bash build.sh -I arm32
    COMPILE_RET=$?

    if [[ "${COMPILE_RET}" -ne 0 ]]; then
        echo "---------------- mindspore lite arm32: build failed ----------------"
        exit
    fi
    # copy arm32 so
    cd ${TOP_PATH}/output/
    rm -rf mindspore-lite-${VERSION_STR}runtime-arm32-cpu
    tar -zxvf mindspore-lite-${VERSION_STR}-runtime-arm32-cpu.tar.gz
    mkdir -p ${BASE_PATH}/java/app/libs/armeabi-v7a/
    rm -rf ${BASE_PATH}/java/app/libs/armeabi-v7a/*
    cp ${TOP_PATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm32-cpu/lib/libmindspore-lite.so ${BASE_PATH}/java/app/libs/armeabi-v7a/
    cp ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so ${BASE_PATH}/java/app/libs/armeabi-v7a/
}

build_jni_arm64() {
    # build jni so
    cd ${BASE_PATH}/java/app/src/main/native
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
                  -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
                  -DANDROID_STL="c++_shared" -DCMAKE_BUILD_TYPE=Debug ..
    VERBOSE=2 make -j8
    mkdir -p ${BASE_PATH}/java/app/libs/arm64-v8a/
    cp ${BASE_PATH}/java/app/src/main/native/build/libmindspore-lite-jni.so ${BASE_PATH}/java/app/libs/arm64-v8a/
}

build_jni_arm32() {
    # build jni so
    cd ${BASE_PATH}/java/app/src/main/native
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
                  -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
                  -DANDROID_STL="c++_shared" -DCMAKE_BUILD_TYPE=Debug ..
    VERBOSE=2 make -j8
    mkdir -p ${BASE_PATH}/java/app/libs/armeabi-v7a/
    cp ${BASE_PATH}/java/app/src/main/native/build/libmindspore-lite-jni.so ${BASE_PATH}/java/app/libs/armeabi-v7a/
}

build_aar() {
    # build aar
    ## check sdk gradle
    cd ${BASE_PATH}/java
    rm -rf .gradle build gradle gradlew gradlew.bat build app/build

    gradle init
    gradle wrapper
    ./gradlew build
}

copy_output() {
    # copy output
    cd ${BASE_PATH}/
    rm -rf output
    mkdir -pv output
    cp ${BASE_PATH}/java/app/build/outputs/aar/mindspore-lite.aar ${BASE_PATH}/output/
}

get_version
build_mslite_arm64
build_mslite_arm32
build_jni_arm64
build_jni_arm32
build_aar
copy_output
