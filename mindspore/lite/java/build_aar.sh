#!/usr/bin/env bash

set -e
BASE_PATH=$(cd "$(dirname $0)"; pwd)
TOP_PATH="${BASE_PATH}/../../.."
# build mindspore-lite arm64
cd ${TOP_PATH}
bash build.sh -I arm64
COMPILE_RET=$?

if [[ "${COMPILE_RET}" -ne 0 ]]; then
    echo "---------------- mindspore lite: build failed ----------------"
    exit
fi

# copy arm64 so
cd ${TOP_PATH}/output/
rm -rf MSLite-0.6.0-linux_arm64
tar -zxvf MSLite-0.6.0-linux_arm64.tar.gz
cp ${TOP_PATH}/output/MSLite-0.6.0-linux_arm64/lib/libmindspore-lite.so ${BASE_PATH}/lib/
cp ${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so ${BASE_PATH}/lib/

# build jni so
cd ${BASE_PATH}/native
rm -rf build
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
              -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
              -DANDROID_STL="c++_shared" -DCMAKE_BUILD_TYPE=Debug ..
VERBOSE=2 make -j8
cp ${BASE_PATH}/native/build/libmindspore-lite-jni.so ${BASE_PATH}/lib/

# build aar
## check sdk gradle
cd ${BASE_PATH}/java
rm -rf .gradle build gradle gradlew gradlew.bat build app/build
rm -rf ${BASE_PATH}/java/app/libs/arm64-v8a/*
cp ${BASE_PATH}/lib/*.so ${BASE_PATH}/java/app/libs/arm64-v8a/
gradle init
gradle wrapper
./gradlew build

# copy output
cd ${BASE_PATH}/
rm -rf output
mkdir -pv output
cp ${BASE_PATH}/java/app/build/outputs/aar/mindspore-lite.aar ${BASE_PATH}/output/
