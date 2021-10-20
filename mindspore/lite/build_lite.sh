#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e

checkndk() {
    if [ "${ANDROID_NDK}" ]; then
        echo -e "\e[31mANDROID_NDK=$ANDROID_NDK  \e[0m"
    else
        echo -e "\e[31mplease set ANDROID_NDK in environment variable for example: export ANDROID_NDK=/root/usr/android-ndk-r20b/ \e[0m"
        exit 1
    fi
}

get_version() {
    VERSION_MAJOR=$(grep "const int ms_version_major =" ${BASEPATH}/mindspore/lite/include/version.h | tr -dc "[0-9]")
    VERSION_MINOR=$(grep "const int ms_version_minor =" ${BASEPATH}/mindspore/lite/include/version.h | tr -dc "[0-9]")
    VERSION_REVISION=$(grep "const int ms_version_revision =" ${BASEPATH}/mindspore/lite/include/version.h | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}

write_commit_file() {
    COMMIT_STR=$(git log -1 | grep commit)
    echo ${COMMIT_STR} > "${BASEPATH}/mindspore/lite/build/.commit_id"
}

build_lite_x86_64_jni_and_jar() {
    # copy x86 so
    local is_train=on
    cd ${BASEPATH}/output/tmp
    local pkg_name=mindspore-lite-${VERSION_STR}-linux-x64

    cd ${BASEPATH}/output/tmp/
    rm -rf ${pkg_name}
    tar -zxf ${BASEPATH}/output/tmp/${pkg_name}.tar.gz
    rm -rf ${LITE_JAVA_PATH}/java/linux_x86/libs/   && mkdir -pv ${LITE_JAVA_PATH}/java/linux_x86/libs/
    rm -rf ${LITE_JAVA_PATH}/native/libs/linux_x86/ && mkdir -pv ${LITE_JAVA_PATH}/native/libs/linux_x86/
    cp ./${pkg_name}/runtime/lib/*.so* ${LITE_JAVA_PATH}/java/linux_x86/libs/
    cp ./${pkg_name}/runtime/lib/*.so* ${LITE_JAVA_PATH}/native/libs/linux_x86/
    local train_so=$pkg_name/runtime/lib/libmindspore-lite-train.so
    if [ ! -f "$train_so" ]; then
        echo "not exist"
        is_train=off
    fi
    if [[ "X$is_train" = "Xon" ]]; then
        cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/java/linux_x86/libs/
        cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/native/libs/linux_x86/
    fi
    # build jni so
    cd ${BASEPATH}/mindspore/lite/build
    rm -rf java/jni && mkdir -pv java/jni
    cd java/jni
    cmake -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DCMAKE_BUILD_TYPE=${LITE_BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} -DSUPPORT_TRAIN=${is_train} "${LITE_JAVA_PATH}/native/"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni x86_64 failed----------------"
        exit 1
    fi
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/java/linux_x86/libs/
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/native/libs/linux_x86/
    cp ./libmindspore-lite-jni.so ${BASEPATH}/output/tmp/${pkg_name}/runtime/lib/
    if [[ "X$is_train" = "Xon" ]]; then
        cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/java/linux_x86/libs/
        cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/native/libs/linux_x86/
        cp ./libmindspore-lite-train-jni.so ${BASEPATH}/output/tmp/${pkg_name}/runtime/lib/
    fi

    cd ${LITE_JAVA_PATH}/java
    rm -rf gradle .gradle gradlew gradlew.bat
    gradle wrapper --gradle-version 6.6.1 --distribution-type all
    # build java common
    ${LITE_JAVA_PATH}/java/gradlew clean -p ${LITE_JAVA_PATH}/java/common
    ${LITE_JAVA_PATH}/java/gradlew build -p ${LITE_JAVA_PATH}/java/common
    cp ${LITE_JAVA_PATH}/java/common/build/libs/mindspore-lite-java-common.jar ${LITE_JAVA_PATH}/java/linux_x86/libs/

    # build java fl_client
    if [[ "X$is_train" = "Xon" ]]; then
        ${LITE_JAVA_PATH}/java/gradlew clean -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew createFlatBuffers -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew build -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew clearJar -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew flReleaseJarX86 --rerun-tasks -p ${LITE_JAVA_PATH}/java/fl_client
        cp ${LITE_JAVA_PATH}/java/fl_client/build/libs/jarX86/mindspore-lite-java-flclient.jar ${BASEPATH}/output/tmp/${pkg_name}/runtime/lib/
        rm -rf ${LITE_JAVA_PATH}/java/fl_client/.gradle ${LITE_JAVA_PATH}/java/fl_client/src/main/java/mindspore
    fi

    # build jar
    ${LITE_JAVA_PATH}/java/gradlew clean -p ${LITE_JAVA_PATH}/java/linux_x86/
    ${LITE_JAVA_PATH}/java/gradlew releaseJar -p ${LITE_JAVA_PATH}/java/linux_x86/
    cp ${LITE_JAVA_PATH}/java/linux_x86/build/lib/jar/*.jar ${BASEPATH}/output/tmp/${pkg_name}/runtime/lib/

    # package
    cd ${BASEPATH}/output/tmp
    rm -rf ${pkg_name}.tar.gz ${pkg_name}.tar.gz.sha256
    tar czf ${pkg_name}.tar.gz ${pkg_name}
    sha256sum ${pkg_name}.tar.gz > ${pkg_name}.tar.gz.sha256
    rm -rf ${LITE_JAVA_PATH}/java/linux_x86/libs/
    rm -rf ${LITE_JAVA_PATH}/native/libs/linux_x86/
}

build_lite() {
    [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output
    get_version
    echo "============ Start building MindSpore Lite ${VERSION_STR} ============"
    local local_lite_platform=${LITE_PLATFORM}
    if [[ "${LITE_ENABLE_AAR}" == "on" ]]; then
        local_lite_platform=$1
        mkdir -pv ${BASEPATH}/mindspore/lite/build/java
        cd ${BASEPATH}/mindspore/lite/build/
        [ -n "${BASEPATH}" ] && find . -maxdepth 1 | grep -v java | grep '/' | xargs -I {} rm -rf {}
    else
        if [[ "${INC_BUILD}" == "off" ]]; then
            [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/mindspore/lite/build
        fi
        mkdir -pv ${BASEPATH}/mindspore/lite/build
    fi
    cd ${BASEPATH}/mindspore/lite/build
    write_commit_file

    if [[ "${local_lite_platform}" == "arm32" ]]; then
      if [[ "${TOOLCHAIN_NAME}" == "ohos-lite" ]]; then
        COMPILE_MINDDATA_LITE="off"
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/ohos-lite.toolchain.cmake
        CMAKE_TOOLCHAIN_NAME="ohos-lite"
      elif [[ "${TOOLCHAIN_NAME}" == "himix200" ]]; then
        COMPILE_MINDDATA_LITE="off"
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/himix200.toolchain.cmake
        CMAKE_TOOLCHAIN_NAME="himix200"
      else
        CMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        ANDROID_NATIVE_API_LEVEL="19"
        CMAKE_ANDROID_NDK=${ANDROID_NDK}
        CMAKE_ANDROID_ABI="armeabi-v7a"
        CMAKE_ANDROID_TOOLCHAIN_NAME="clang"
        CMAKE_ANDROID_STL=${MSLITE_ANDROID_STL}
        MSLITE_ENABLE_FP16="on"
      fi
    fi

    if [[ "${local_lite_platform}" == "arm64" ]]; then
      if [ "$(uname)" == "Darwin" ]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ios-aarch64
        cmake -DCMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake -DARCHS="arm64" -DENABLE_BITCODE=0                   \
              -DCMAKE_BUILD_TYPE="Release" -DBUILD_MINDDATA="" -DPLATFORM_ARM64="on" -DENABLE_NEON="on" -DMSLITE_ENABLE_FP16="on" \
              -DMSLITE_ENABLE_TRAIN="off" -DMSLITE_GPU_BACKEND="off" -DMSLITE_ENABLE_NPU="off"        \
              -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_INSTALL_PREFIX=${BUILD_PATH}/output/tmp -G Xcode ..
      else
        checkndk
        echo "default link libc++_static.a, export MSLITE_ANDROID_STL=c++_shared to link libc++_shared.so"
        cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"         \
              -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"     \
              -DANDROID_STL=${MSLITE_ANDROID_STL} -DCMAKE_BUILD_TYPE=${LITE_BUILD_TYPE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} \
              -DPLATFORM_ARM64="on" -DENABLE_NEON="on" -DMSLITE_ENABLE_FP16="on" -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp           \
              -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION}   \
              -DENABLE_ASAN=${ENABLE_ASAN} -DENABLE_VERBOSE=${ENABLE_VERBOSE} "${BASEPATH}/mindspore/lite"
      fi
    elif [[ "${local_lite_platform}" == "arm32" ]]; then
      if [ "$(uname)" == "Darwin" ]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ios-aarch32
        cmake -DCMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake -DARCHS="armv7;armv7s" -DENABLE_BITCODE=0     \
              -DCMAKE_BUILD_TYPE="Release" -DBUILD_MINDDATA="" -DPLATFORM_ARM32="on" -DENABLE_NEON="on"             \
              -DMSLITE_ENABLE_TRAIN="off" -DMSLITE_GPU_BACKEND="off" -DMSLITE_ENABLE_NPU="off" \
              -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_INSTALL_PREFIX=${BUILD_PATH}/output/tmp -G Xcode ..
      else
        checkndk
        echo "default link libc++_static.a, export MSLITE_ANDROID_STL=c++_shared to link libc++_shared.so"
        cmake -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DTOOLCHAIN_NAME=${CMAKE_TOOLCHAIN_NAME} -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}          \
              -DANDROID_NDK=${CMAKE_ANDROID_NDK} -DANDROID_ABI=${CMAKE_ANDROID_ABI} -DANDROID_TOOLCHAIN_NAME=${CMAKE_ANDROID_TOOLCHAIN_NAME}                    \
              -DANDROID_STL=${CMAKE_ANDROID_STL}  -DCMAKE_BUILD_TYPE=${LITE_BUILD_TYPE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} \
              -DPLATFORM_ARM32="on" -DENABLE_NEON="on"  -DMSLITE_ENABLE_FP16=${MSLITE_ENABLE_FP16} -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp           \
              -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION}    \
              -DENABLE_ASAN=${ENABLE_ASAN} -DENABLE_VERBOSE=${ENABLE_VERBOSE} "${BASEPATH}/mindspore/lite"
      fi
    else
      if [ "$(uname)" == "Darwin" ]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ios-simulator
        cmake -DCMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake -DPLATFORM=SIMULATOR64 -DMSLITE_ENABLE_MINDRT="on" \
              -DCMAKE_BUILD_TYPE="Release" -DPLATFORM_ARM64="off" -DENABLE_NEON="off" -DMSLITE_ENABLE_TOOLS="off" -DMSLITE_ENABLE_CONVERTER=off \
              -DMSLITE_ENABLE_TRAIN="off" -DMSLITE_GPU_BACKEND="off" -DMSLITE_ENABLE_NPU="off" -DBUILD_MINDDATA="" -DMSLITE_ENABLE_V0=on \
              -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp -DENABLE_VERBOSE=${ENABLE_VERBOSE} "${BASEPATH}/mindspore/lite" \
              -G Xcode ..
      else
        cmake -DPLATFORM_X86_64=on -DCMAKE_BUILD_TYPE=${LITE_BUILD_TYPE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE}              \
              -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
              -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp -DENABLE_VERBOSE=${ENABLE_VERBOSE} "${BASEPATH}/mindspore/lite"
      fi
    fi
    if [[ "$(uname)" == "Darwin" && "${local_lite_platform}" != "x86_64" ]]; then
        xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme mindspore-lite_static -target mindspore-lite_static -sdk iphoneos -quiet
    elif [[ "$(uname)" == "Darwin" && "${local_lite_platform}" == "x86_64" ]]; then
        xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme mindspore-lite_static -target mindspore-lite_static -sdk iphonesimulator -quiet
    else
      make -j$THREAD_NUM && make install && make package
      if [[ "${local_lite_platform}" == "x86_64" ]]; then
        if [ "${JAVA_HOME}" ]; then
            echo -e "\e[31mJAVA_HOME=$JAVA_HOME  \e[0m"
            build_lite_x86_64_jni_and_jar
        else
            echo -e "\e[31mJAVA_HOME is not set, so jni and jar packages will not be compiled \e[0m"
            echo -e "\e[31mIf you want to compile the JAR package, please set $JAVA_HOME. For example: export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64 \e[0m"
        fi
      fi
    fi
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build failed ----------------"
        exit 1
    else
        if [ "$(uname)" == "Darwin" ]; then
          mkdir -p ${BASEPATH}/output
          cp -r ${BASEPATH}/mindspore/lite/build/src/Release-*/mindspore-lite.framework ${BASEPATH}/output/mindspore-lite.framework
          cd ${BASEPATH}/output
          tar -zcvf ${pkg_name}.tar.gz mindspore-lite.framework/
          sha256sum ${pkg_name}.tar.gz > ${pkg_name}.tar.gz.sha256
          rm -r mindspore-lite.framework
        else
          mv ${BASEPATH}/output/tmp/*.tar.gz* ${BASEPATH}/output/
        fi

        if [[ "${local_lite_platform}" == "x86_64" ]]; then
          if [[ "${MSLITE_ENABLE_TESTCASES}" == "ON" || "${MSLITE_ENABLE_TESTCASES}" == "on" ]]; then
            mkdir -pv ${BASEPATH}/mindspore/lite/test/do_test || true
            if [[ ! "${MSLITE_ENABLE_CONVERTER}" || "${MSLITE_ENABLE_CONVERTER}"  == "ON" || "${MSLITE_ENABLE_CONVERTER}"  == "on" ]]; then
              cp ${BASEPATH}/output/tmp/mindspore-lite*/tools/converter/lib/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            fi
            cp ${BASEPATH}/output/tmp/mindspore-lite*/runtime/lib/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            if [[ ! "${MSLITE_ENABLE_TRAIN}" || "${MSLITE_ENABLE_TRAIN}"  == "ON" || "${MSLITE_ENABLE_TRAIN}"  == "on" ]]; then
              cp ${BASEPATH}/output/tmp/mindspore-lite*/runtime/third_party/libjpeg-turbo/lib/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            fi
          fi
        fi

        [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output/tmp/
        if [[ "${MSLITE_ENABLE_NNIE}" == "on" ]]; then
          compile_nnie_script=${BASEPATH}/mindspore/lite/tools/providers/NNIE/Hi3516D/compile_nnie.sh
          cd ${BASEPATH}/../
          if [[ "${local_lite_platform}" == "x86_64" ]]; then
            sh ${compile_nnie_script} -I x86_64 -b nnie_3516_r1.5 -j $THREAD_NUM
            if [[ $? -ne 0 ]]; then
              echo "compile x86_64 for nnie failed."
              exit 1
            fi
          elif [[ "${local_lite_platform}" == "arm32" ]]; then
            sh ${compile_nnie_script} -I arm32 -b nnie_3516_r1.5 -j $THREAD_NUM
            if [[ $? -ne 0 ]]; then
              echo "compile arm32 for nnie failed."
              exit 1
            fi
          fi
        fi
        echo "---------------- mindspore lite: build success ----------------"
    fi
}

build_lite_arm64_and_jni() {
    build_lite "arm64"
    # copy arm64 so
    local is_train=on
    local pkg_name=mindspore-lite-${VERSION_STR}-android-aarch64
    cd "${BASEPATH}/mindspore/lite/build"

    rm -rf ${pkg_name}
    tar -zxf ${BASEPATH}/output/${pkg_name}.tar.gz
    rm -rf ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/ && mkdir -p ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
    rm -rf ${LITE_JAVA_PATH}/native/libs/arm64-v8a/   && mkdir -p ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    cp ./${pkg_name}/runtime/lib/libmindspore-lite.so ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
    cp ./${pkg_name}/runtime/lib/libmindspore-lite.so ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    local train_so=$pkg_name/runtime/lib/libmindspore-lite-train.so
    if [ ! -f "$train_so" ]; then
        echo "not exist"
        is_train=off
    fi
    if [[ "X$is_train" = "Xon" ]]; then
        cp ./${pkg_name}/runtime/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
        cp ./${pkg_name}/runtime/lib/*.so* ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
        cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
        cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    fi
    # build jni so
    [ -n "${BASEPATH}" ] && rm -rf java/jni && mkdir -pv java/jni
    cd java/jni
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DANDROID_STL=${MSLITE_ANDROID_STL} -DCMAKE_BUILD_TYPE=${LITE_BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
          -DSUPPORT_TRAIN=${is_train} -DPLATFORM_ARM64=on "${LITE_JAVA_PATH}/native/"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni arm64 failed----------------"
        exit 1
    fi
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    if [[ "X$is_train" = "Xon" ]]; then
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    fi
}

build_lite_arm32_and_jni() {
    build_lite "arm32"
    # copy arm32 so
    local is_train=on
    local pkg_name=mindspore-lite-${VERSION_STR}-android-aarch32
    cd "${BASEPATH}/mindspore/lite/build"

    rm -rf ${pkg_name}
    tar -zxf ${BASEPATH}/output/${pkg_name}.tar.gz
    rm -rf ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/ && mkdir -pv ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
    rm -rf ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/   && mkdir -pv ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    cp ./${pkg_name}/runtime/lib/libmindspore-lite.so ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
    cp ./${pkg_name}/runtime/lib/libmindspore-lite.so ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    local train_so=$pkg_name/runtime/lib/libmindspore-lite-train.so
    if [ ! -f "$train_so" ]; then
        echo "not exist"
        is_train=off
    fi
    if [[ "X$is_train" = "Xon" ]]; then
        cp ./${pkg_name}/runtime/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
        cp ./${pkg_name}/runtime/lib/*.so* ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
        cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
        cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    fi

    # build jni so
    [ -n "${BASEPATH}" ] && rm -rf java/jni && mkdir -pv java/jni
    cd java/jni
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DANDROID_STL=${MSLITE_ANDROID_STL} -DCMAKE_BUILD_TYPE=${LITE_BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
          -DSUPPORT_TRAIN=${is_train} -DPLATFORM_ARM32=on "${LITE_JAVA_PATH}/native"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni arm32 failed----------------"
        exit 1
    fi
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    if [[ "X$is_train" = "Xon" ]]; then
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    fi
}

build_aar() {
    get_version
    if [[ "X${INC_BUILD}" == "Xoff" ]]; then
        [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/mindspore/lite/build
    fi
    cd ${LITE_JAVA_PATH}/java
    rm -rf gradle .gradle gradlew gradlew.bat
    gradle wrapper --gradle-version 6.6.1 --distribution-type all
    # build common module
    ${LITE_JAVA_PATH}/java/gradlew clean -p ${LITE_JAVA_PATH}/java/common
    ${LITE_JAVA_PATH}/java/gradlew build -p ${LITE_JAVA_PATH}/java/common

    # build aar
    local npu_bak=${MSLITE_ENABLE_NPU}
    export MSLITE_ENABLE_NPU="off"
    build_lite_arm64_and_jni
    build_lite_arm32_and_jni
    export MSLITE_ENABLE_NPU=${npu_bak}

    # build java fl_client
    local is_train=on
    local train_so=${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/libmindspore-lite-train.so
    if [ ! -f "$train_so" ]; then
        echo "not exist"
        is_train=off
    fi
    if [[ "X$is_train" = "Xon" ]]; then
        ${LITE_JAVA_PATH}/java/gradlew clean -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew createFlatBuffers -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew build -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew clearJar -p ${LITE_JAVA_PATH}/java/fl_client
        ${LITE_JAVA_PATH}/java/gradlew flReleaseJarAAR --rerun-tasks -p ${LITE_JAVA_PATH}/java/fl_client
        cp ${LITE_JAVA_PATH}/java/fl_client/build/libs/jarAAR/mindspore-lite-java-flclient.jar ${LITE_JAVA_PATH}/java/app/libs
        rm -rf ${LITE_JAVA_PATH}/java/fl_client/.gradle ${LITE_JAVA_PATH}/java/fl_client/src/main/java/mindspore
    fi

    cp ${LITE_JAVA_PATH}/java/common/build/libs/mindspore-lite-java-common.jar ${LITE_JAVA_PATH}/java/app/libs
    ${LITE_JAVA_PATH}/java/gradlew clean -p ${LITE_JAVA_PATH}/java/app
    ${LITE_JAVA_PATH}/java/gradlew assembleRelease  -p ${LITE_JAVA_PATH}/java/app
    ${LITE_JAVA_PATH}/java/gradlew publish -PLITE_VERSION=${VERSION_STR} -p ${LITE_JAVA_PATH}/java/app

    cd ${LITE_JAVA_PATH}/java/app/build
    [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output/*.tar.gz*
    zip -r ${BASEPATH}/output/mindspore-lite-maven-${VERSION_STR}.zip mindspore
    cd ${BASEPATH}/output
    sha256sum mindspore-lite-maven-${VERSION_STR}.zip > mindspore-lite-maven-${VERSION_STR}.zip.sha256
}

update_submodule()
{
  git submodule update --init graphengine
  cd "${BASEPATH}/graphengine"
  git submodule update --init metadef
}

LITE_JAVA_PATH=${BASEPATH}/mindspore/lite/java
LITE_BUILD_TYPE="Release"
if [[ "${MSLITE_ENABLE_ACL}" == "on" ]]; then
    update_submodule
fi
if [[ "${DEBUG_MODE}" == "on" ]]; then
    LITE_BUILD_TYPE="Debug"
fi
if [[ "X$LITE_ENABLE_AAR" = "Xon" ]]; then
    build_aar
elif [[ "X$LITE_PLATFORM" != "X" ]]; then
    build_lite
else
    echo "Invalid parameter"
fi
