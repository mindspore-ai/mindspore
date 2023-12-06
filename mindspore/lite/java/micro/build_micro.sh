#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

usage() {
  echo "Usage:"
  echo "    bash build_micro.sh [-p path] [-s path] [-A on|off] [-t on|off]"
  echo ""
  echo "Options:"
  echo "    -p set the path of mindspore lite package"
  echo "    -s set the path of micro static lib file. This file is not needed when -t is on"
  echo "    -A whether build Android lib, on for arm64, off for x84_64"
  echo "    -t whether build and run testcases, only support x84_64"
}

checkndk() {
  if [ "${ANDROID_NDK}" ]; then
    echo -e "\e[31mANDROID_NDK=$ANDROID_NDK  \e[0m"
  else
    echo -e "\e[31mplease set ANDROID_NDK in environment variable for example: export ANDROID_NDK=/root/usr/android-ndk-r20b/ \e[0m"
    exit 1
  fi
}

checkopts() {
  while getopts 'p:s:A:t:' opt
  do
    LOW_OPT_ARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')

    case "${opt}" in
      p)
        echo "user opt: -p ""${OPTARG}"
        LITE_PKG_PATH=${OPTARG}
        ;;
      s)
        echo "user opt: -s ""${OPTARG}"
        MICRO_STATIC_LIB=${OPTARG}
        ;;
      A)
        echo "user opt: -A ""${OPTARG}"
        ANDROID_PLATFORM=${LOW_OPT_ARG}
        ;;
      t)
        echo "user opt: -t ""${OPTARG}"
        RUN_TESTCASES=${LOW_OPT_ARG}
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

build_micro_so() {
  cd ${MICRO_BUILD_PATH}
  MICRO_CMAKE_ARGS="-DPKG_PATH=${LITE_PKG_PATH}"
  MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DMICRO_STATIC_LIB=${MICRO_STATIC_LIB}"
  MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DMICRO_BUILD_PATH=${MICRO_BUILD_PATH}"
  MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DMICRO_JNI_PATH=${MICRO_JNI_PATH}"

  if [[ "${ANDROID_PLATFORM}" == "on" ]]; then
    checkndk
    export PATH=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin:${ANDROID_NDK}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:${PATH}
    MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
    MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_NDK=${ANDROID_NDK}"
    MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DANDROID_ABI=arm64-v8a -DPLATFORM_ARM64=ON"
    MICRO_CMAKE_ARGS="${MICRO_CMAKE_ARGS} -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang"
    cmake ${MICRO_JNI_PATH} ${MICRO_CMAKE_ARGS} && make
    ${ANDROID_NDK}/toolchains/aarch64-linux-android-4.9/prebuilt/linux-x86_64/bin/aarch64-linux-android-strip ./*.so
    mkdir -pv output/arm64-v8a && mv ${MICRO_BUILD_PATH}/*.so output/arm64-v8a
  else
    cmake ${MICRO_JNI_PATH} ${MICRO_CMAKE_ARGS} && make
    mkdir -pv output/x86_64 && mv ${MICRO_BUILD_PATH}/*.so output/x86_64
  fi
}

build_micro_jar() {
  cd ${BASEPATH}
  rm -rf gradle .gradle gradlew gradlew.bat
  local gradle_version=""
  gradle_version=`gradle --version | grep Gradle | awk '{print$2}'`
  if [[ ${gradle_version} == '6.6.1' ]]; then
    gradle_command=gradle
  else
    gradle wrapper --gradle-version 6.6.1 --distribution-type all
    gradle_command=${BASEPATH}/gradlew
  fi

  if [[ "${RUN_TESTCASES}" == "on" ]] ; then
    export LD_LIBRARY_PATH=${MICRO_BUILD_PATH}/output/x86_64/:${LD_LIBRARY_PATH}
    echo "start micro Java test (x86_64), 3 SEVERE and 4 ERROR messages will be print"
    ${gradle_command} releaseJar
  else
    ${gradle_command} releaseJar -x test
  fi
}

test_prepare() {
  TEST_MODEL_PATH=${BASEPATH}/../../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms
  MICRO_CFG_PATH=${BASEPATH}/../../test/config_level0/micro/micro_x86.cfg

  # convert .ms to micro C code, and build libnet.a
  export LD_LIBRARY_PATH=${LITE_PKG_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
  cd ${LITE_PKG_PATH}/tools/converter/converter
  ./converter_lite --fmk=MSLITE --modelFile=${TEST_MODEL_PATH} --outputFile=${MICRO_BUILD_PATH}/lenet --configFile=${MICRO_CFG_PATH}
  cd ${MICRO_BUILD_PATH}/lenet/src
  mkdir build && cd build
  cmake -DPKG_PATH=${LITE_PKG_PATH} -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && make

  MICRO_STATIC_LIB=$(pwd)/libnet.a
}

BASEPATH=$(cd "$(dirname $0)"; pwd)
MICRO_BUILD_PATH="${BASEPATH}/build"
MICRO_JNI_PATH="${BASEPATH}/src/main/native"
mkdir -pv "${MICRO_BUILD_PATH}"

# Init default values of build options
LITE_PKG_PATH=""
MICRO_STATIC_LIB=""
ANDROID_PLATFORM="off"
RUN_TESTCASES="off"

checkopts "$@"

if [[ "${RUN_TESTCASES}" == "on" ]] ; then
  test_prepare
fi

build_micro_so
build_micro_jar
