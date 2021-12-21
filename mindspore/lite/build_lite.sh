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

check_Hi35xx() {
  if [[ "X${HI35XX_SDK_PATH}" == "X" ]]; then
    echo "error: to compile the runtime package of Hi35XX, you need to set HI35XX_SDK_PATH to declare the path of Hi35XX sdk."
    exit 1
  else
    cp -r ${HI35XX_SDK_PATH}/third_patry ${BASEPATH}/mindspore/lite/tools/benchmark/nnie/
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
    X86_JNI_CMAKE_ARGS=$1
    export MSLITE_ENABLE_RUNTIME_CONVERT=off
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
    cmake ${X86_JNI_CMAKE_ARGS} -DSUPPORT_TRAIN=${is_train} "${LITE_JAVA_PATH}/native/"
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
    local gradle_version=`gradle --version | grep Gradle | awk '{print$2}'`
    if [[ ${gradle_version} == '6.6.1' ]]; then
      gradle_command=gradle
    else
      gradle wrapper --gradle-version 6.6.1 --distribution-type all
      gradle_command=${LITE_JAVA_PATH}/java/gradlew
    fi
    # build java common
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/java/common
    ${gradle_command} build -p ${LITE_JAVA_PATH}/java/common
    cp ${LITE_JAVA_PATH}/java/common/build/libs/mindspore-lite-java-common.jar ${LITE_JAVA_PATH}/java/linux_x86/libs/

    # build java fl_client
    if [[ "X$is_train" = "Xon" ]]; then
        ${gradle_command} clean -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} createFlatBuffers -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} build -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} clearJar -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} flReleaseJarX86 --rerun-tasks -p ${LITE_JAVA_PATH}/java/fl_client
        cp ${LITE_JAVA_PATH}/java/fl_client/build/libs/jarX86/mindspore-lite-java-flclient.jar ${BASEPATH}/output/tmp/${pkg_name}/runtime/lib/
        rm -rf ${LITE_JAVA_PATH}/java/fl_client/.gradle ${LITE_JAVA_PATH}/java/fl_client/src/main/java/mindspore
    fi

    # build jar
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/
    if [[ "${ENABLE_ASAN}" == "ON" || "${ENABLE_ASAN}" == "on" ]] ; then
      ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ -x test
    else
      if [[ "${MSLITE_ENABLE_TESTCASES}" == "ON" || "${MSLITE_ENABLE_TESTCASES}" == "on" ]] ; then
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_JAVA_PATH}/native/libs/linux_x86/
          ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/
      else
           ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ -x test
      fi
    fi
    cp ${LITE_JAVA_PATH}/build/lib/jar/*.jar ${BASEPATH}/output/tmp/${pkg_name}/runtime/lib/

    # package
    cd ${BASEPATH}/output/tmp
    rm -rf ${pkg_name}.tar.gz ${pkg_name}.tar.gz.sha256
    tar czf ${pkg_name}.tar.gz ${pkg_name}
    sha256sum ${pkg_name}.tar.gz > ${pkg_name}.tar.gz.sha256
    rm -rf ${LITE_JAVA_PATH}/java/linux_x86/libs/
    rm -rf ${LITE_JAVA_PATH}/native/libs/linux_x86/
}

build_lite() {
    LITE_CMAKE_ARGS=${CMAKE_ARGS}
    [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output
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

    LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp"

    if [[ "$(uname)" == "Darwin" && "${local_lite_platform}" != "x86_64" ]]; then
      LITE_CMAKE_ARGS=`echo $LITE_CMAKE_ARGS | sed 's/-DCMAKE_BUILD_TYPE=Debug/-DCMAKE_BUILD_TYPE=Release/g'`
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off -DMSLITE_ENABLE_NPU=off"
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DENABLE_BITCODE=0 -G Xcode"
      CMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake
    fi

    BRANCH_NAME=nnie_3516_master_2
    if [[ ("${MSLITE_REGISTRY_DEVICE}" == "Hi3516D" || "${TOOLCHAIN_NAME}" == "himix200") && "${local_lite_platform}" == "arm32" ]]; then
      TOOLCHAIN_NAME="himix200"
      MSLITE_REGISTRY_DEVICE=Hi3516D
      check_Hi35xx
      MSLITE_COMPILE_TWICE=ON
    elif [[ "${MSLITE_REGISTRY_DEVICE}" == "Hi3559A" && "${local_lite_platform}" == "arm64" ]]; then
      TOOLCHAIN_NAME="himix100"
      check_Hi35xx
      MSLITE_COMPILE_TWICE=ON
    elif [[ "${MSLITE_REGISTRY_DEVICE}" == "SD3403" && "${local_lite_platform}" == "arm64" ]]; then
      TOOLCHAIN_NAME="mix210"
      MSLITE_COMPILE_TWICE=ON
    elif [[ "${MSLITE_REGISTRY_DEVICE}" == "Hi3519A" && "${local_lite_platform}" == "arm32" ]]; then
      TOOLCHAIN_NAME="himix200"
      check_Hi35xx
      MSLITE_COMPILE_TWICE=ON
    elif [[ ("${MSLITE_ENABLE_NNIE}" == "on" || "${MSLITE_REGISTRY_DEVICE}" == "Hi3516D") && "${local_lite_platform}" == "x86_64" ]]; then
      MSLITE_REGISTRY_DEVICE=Hi3516D
    fi

    machine=`uname -m`
    echo "machine:${machine}."
    if [[ "${local_lite_platform}" == "arm32" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM_ARM32=on -DENABLE_NEON=on"
      if [ "$(uname)" == "Darwin" ]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ios-aarch32
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DARCHS=armv7;armv7s"
      elif [[ "${TOOLCHAIN_NAME}" == "ohos-lite" ]]; then
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/ohos-lite.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=ohos-lite"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off"
      elif [[ "${TOOLCHAIN_NAME}" == "himix200" ]]; then
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/himix200.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=himix200"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      else
        checkndk
        CMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=lite_cv"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=clang -DANDROID_STL=${MSLITE_ANDROID_STL}"
      fi
    elif [[ "${local_lite_platform}" == "arm64" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM_ARM64=on -DENABLE_NEON=on"
      if [ "$(uname)" == "Darwin" ]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ios-aarch64
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DARCHS=arm64"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on"
      elif [[ "${TOOLCHAIN_NAME}" == "himix100" ]]; then
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/himix100.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=himix100"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      elif [[ "${TOOLCHAIN_NAME}" == "mix210" ]]; then
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/mix210.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=mix210"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      else
        if [[ "${machine}" == "aarch64" ]]; then
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMACHINE_LINUX_ARM64=on"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_TRAIN=off"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_GPU_BACKEND=off"
        else
          checkndk
          CMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang -DANDROID_STL=${MSLITE_ANDROID_STL}"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=lite_cv"
        fi
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on"
      fi
    else
      if [ "$(uname)" == "Darwin" ]; then
         pkg_name=mindspore-lite-${VERSION_STR}-ios-simulator
         CMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake
         LITE_CMAKE_ARGS=`echo $LITE_CMAKE_ARGS | sed 's/-DCMAKE_BUILD_TYPE=Debug/-DCMAKE_BUILD_TYPE=Release/g'`
         LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM=SIMULATOR64 -DPLATFORM_ARM64=off -DENABLE_NEON=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off -DMSLITE_ENABLE_NPU=off -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_V0=on"
         LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_TOOLS=off -DMSLITE_ENABLE_CONVERTER=off"
         LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -G Xcode .."
      else
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=lite_cv"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM_X86_64=on"
      fi
    fi

    if [[ "X$CMAKE_TOOLCHAIN_FILE" != "X" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    fi
    if [[ "X$MSLITE_REGISTRY_DEVICE" != "X" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_REGISTRY_DEVICE=${MSLITE_REGISTRY_DEVICE}"
    fi
    if [[ "X$MSLITE_COMPILE_TWICE" != "X" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_COMPILE_TWICE=${MSLITE_COMPILE_TWICE}"
    fi
    if [[ "${local_lite_platform}" == "arm64" || "${local_lite_platform}" == "arm32" ]]; then
      echo "default link libc++_static.a, export MSLITE_ANDROID_STL=c++_shared to link libc++_shared.so"
    fi

    echo "cmake ${LITE_CMAKE_ARGS} ${BASEPATH}/mindspore/lite"
    cmake ${LITE_CMAKE_ARGS} "${BASEPATH}/mindspore/lite"

    if [[ "$(uname)" == "Darwin" && "${local_lite_platform}" != "x86_64" ]]; then
        xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme mindspore-lite_static -target mindspore-lite_static -sdk iphoneos -quiet -UseModernBuildSystem=YES
    elif [[ "$(uname)" == "Darwin" && "${local_lite_platform}" == "x86_64" ]]; then
        xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme mindspore-lite_static -target mindspore-lite_static -sdk iphonesimulator -quiet -UseModernBuildSystem=YES
    else
      make -j$THREAD_NUM && make install
      if [[ "X$MSLITE_COMPILE_TWICE" == "XON" ]]; then
        if [[ "X$MSLITE_ENABLE_TOOLS" != "X" ]]; then
          MSLITE_ENABLE_TOOLS=$(echo $MSLITE_ENABLE_TOOLS | tr '[a-z]' '[A-Z]')
        fi
        if [[ "X$MSLITE_ENABLE_TOOLS" != "XOFF" ]]; then
          LITE_CMAKE_ARGS=`echo $LITE_CMAKE_ARGS | sed 's/-DMSLITE_COMPILE_TWICE=ON/-DMSLITE_COMPILE_TWICE=OFF/g'`
          cp -r ${BASEPATH}/output/tmp/mindspore*/runtime ${BASEPATH}/mindspore/lite/tools/benchmark
          echo "cmake ${LITE_CMAKE_ARGS} ${BASEPATH}/mindspore/lite"
          cmake ${LITE_CMAKE_ARGS} "${BASEPATH}/mindspore/lite"
          cmake --build "${BASEPATH}/mindspore/lite/build" --target benchmark -j$THREAD_NUM
          make install
        fi
      fi
      make package
      if [[ "${local_lite_platform}" == "x86_64" ]]; then
        if [ "${JAVA_HOME}" ]; then
            echo -e "\e[31mJAVA_HOME=$JAVA_HOME  \e[0m"
            build_lite_x86_64_jni_and_jar $1
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
        if [[ "X$MSLITE_REGISTRY_DEVICE" != "X" ]] && [[ "${MSLITE_REGISTRY_DEVICE}" != "SD3403" ]]; then
          local compile_nnie_script=${BASEPATH}/mindspore/lite/tools/providers/NNIE/Hi3516D/compile_nnie.sh
          cd ${BASEPATH}/../
          if [[ "${local_lite_platform}" == "x86_64" ]]; then
            bash ${compile_nnie_script} -I ${local_lite_platform} -b ${BRANCH_NAME} -j $THREAD_NUM
          fi
          if [[ $? -ne 0 ]]; then
            echo "compile ${local_lite_platform} for nnie failed."
            exit 1
          fi
        fi
        echo "---------------- mindspore lite: build success ----------------"
    fi
}

build_lite_arm64_and_jni() {
    local ARM64_CMAKE_ARGS=${CMAKE_ARGS}
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
    cmake ${ARM64_CMAKE_ARGS} -DSUPPORT_TRAIN=${is_train} -DPLATFORM_ARM64=on  \
          -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DANDROID_STL=${MSLITE_ANDROID_STL} "${LITE_JAVA_PATH}/native/"
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
    local ARM32_CMAKE_ARGS=${CMAKE_ARGS}
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
    cmake ${ARM32_CMAKE_ARGS} -DSUPPORT_TRAIN=${is_train} -DPLATFORM_ARM32=on \
          -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DANDROID_STL=${MSLITE_ANDROID_STL} "${LITE_JAVA_PATH}/native"
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
    if [[ "X${INC_BUILD}" == "Xoff" ]]; then
        [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/mindspore/lite/build
    fi
    cd ${LITE_JAVA_PATH}/java
    rm -rf gradle .gradle gradlew gradlew.bat
    local gradle_version=`gradle --version | grep Gradle | awk '{print$2}'`
    if [[ ${gradle_version} == '6.6.1' ]]; then
      gradle_command=gradle
    else
      gradle wrapper --gradle-version 6.6.1 --distribution-type all
      gradle_command=${LITE_JAVA_PATH}/java/gradlew
    fi
    # build common module
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/java/common
    ${gradle_command} build -p ${LITE_JAVA_PATH}/java/common
    # build new java api module
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/
    ${gradle_command} build -p ${LITE_JAVA_PATH}/ -x test

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
        ${gradle_command} clean -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} createFlatBuffers -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} build -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} clearJar -p ${LITE_JAVA_PATH}/java/fl_client
        ${gradle_command} flReleaseJarAAR --rerun-tasks -p ${LITE_JAVA_PATH}/java/fl_client
        cp ${LITE_JAVA_PATH}/java/fl_client/build/libs/jarAAR/mindspore-lite-java-flclient.jar ${LITE_JAVA_PATH}/java/app/libs
        rm -rf ${LITE_JAVA_PATH}/java/fl_client/.gradle ${LITE_JAVA_PATH}/java/fl_client/src/main/java/mindspore
    fi

    cp ${LITE_JAVA_PATH}/java/common/build/libs/mindspore-lite-java-common.jar ${LITE_JAVA_PATH}/java/app/libs
    cp ${LITE_JAVA_PATH}/build/libs/mindspore-lite-java.jar ${LITE_JAVA_PATH}/java/app/libs
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/java/app
    ${gradle_command} assembleRelease  -p ${LITE_JAVA_PATH}/java/app
    ${gradle_command} publish -PLITE_VERSION=${VERSION_STR} -p ${LITE_JAVA_PATH}/java/app

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
if [[ "${MSLITE_ENABLE_ACL}" == "on" ]]; then
    update_submodule
fi

CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_VERBOSE=${ENABLE_VERBOSE}"
if [[ "${DEBUG_MODE}" == "on" ]]; then
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug "
else
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release "
fi
if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
fi

get_version
CMAKE_ARGS="${CMAKE_ARGS} -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION}"

if [[ "X$LITE_ENABLE_AAR" = "Xon" ]]; then
    build_aar
elif [[ "X$LITE_PLATFORM" != "X" ]]; then
    build_lite
else
    echo "Invalid parameter"
fi
