#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
    cp -r ${HI35XX_SDK_PATH}/third_patry ${BASEPATH}/mindspore/lite/providers/nnie
  fi
}

get_version() {
    VERSION_STR=$(cat ${BASEPATH}/version.txt)
}

write_commit_file() {
    COMMIT_STR=$(git log -1 | grep commit)
    echo ${COMMIT_STR} > "${BASEPATH}/mindspore/lite/build/.commit_id"
}

build_lite_jni_and_jar() {
    local JNI_CMAKE_ARGS=$1
    local local_lite_platform=$2
    echo "============ mindspore lite: start building jni and jar ${VERSION_STR} ${local_lite_platform} ============"
    export MSLITE_ENABLE_RUNTIME_CONVERT=off

    local PKG_NAME
    local NATIVE_PATH_ARCH
    local RESOURCE_PATH_ARCH
    if [[ "${local_lite_platform}" == "x86_64" ]]; then
      PKG_NAME=mindspore-lite-${VERSION_STR}-linux-x64
      NATIVE_PATH_ARCH=linux_x86
      RESOURCE_PATH_ARCH=linux_x86_64
    elif [[ "${local_lite_platform}" == "aarch64" ]]; then
      PKG_NAME=mindspore-lite-${VERSION_STR}-linux-aarch64
      NATIVE_PATH_ARCH=linux_aarch64
      RESOURCE_PATH_ARCH=linux_aarch64
      JNI_CMAKE_ARGS="${JNI_CMAKE_ARGS} -DMACHINE_LINUX_ARM64=on"
    else
      echo "platform: ${local_lite_platform} not support building jni."
      exit 0
    fi

    # copy so
    local is_train=on
    cd ${INSTALL_PREFIX}/

    rm -rf ${PKG_NAME}
    tar -zxf ${INSTALL_PREFIX}/${PKG_NAME}.tar.gz
    rm -rf ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/   && mkdir -pv ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
    rm -rf ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/ && mkdir -pv ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
    cp ./${PKG_NAME}/runtime/lib/*.so* ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
    cp ./${PKG_NAME}/runtime/lib/*.so* ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
    local train_so=${PKG_NAME}/runtime/lib/libmindspore-lite-train.so
    if [ ! -f "$train_so" ]; then
        echo "libmindspore-lite-train.so so not exist"
        is_train=off
    fi
    if [[ "X$is_train" = "Xon" ]]; then
        cp ./${PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
        cp ./${PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
    fi
    # prepare
    cd ${BASEPATH}/mindspore/lite/build
    rm -rf java/jni && mkdir -pv java/jni
    cd java/jni
    # copy glog lib and headers
    LIB_GLOG="libmindspore_glog.so*"
    if [[ ${MSLITE_ENABLE_RUNTIME_GLOG} == "on" || ${MSLITE_ENABLE_RUNTIME_GLOG} == "ON" || ${MSLITE_ENABLE_SERVER_INFERENCE} == "on" || ${MSLITE_ENABLE_SERVER_INFERENCE} == "ON" ]]; then
      if [ -f "`echo ${INSTALL_PREFIX}/${PKG_NAME}/runtime/third_party/glog/${LIB_GLOG}`" ]; then
        cp ${INSTALL_PREFIX}/${PKG_NAME}/runtime/third_party/glog/*.so* ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
        cp ${INSTALL_PREFIX}/${PKG_NAME}/runtime/third_party/glog/*.so* ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
      else
        echo "no glog lib found, exit."
        exit 1
      fi
      if [ -d "${BASEPATH}/output/tmp/${PKG_NAME}/runtime/include/third_party/glog" ]; then
          rm -rf jni_include && mkdir jni_include
          cp ${BASEPATH}/output/tmp/${PKG_NAME}/runtime/include/third_party/glog  ./jni_include -r
      else
          echo "no glog hesders found, exit."
          exit 1
      fi
    fi
    # build jni so
    echo "cmake ${JNI_CMAKE_ARGS} -DSUPPORT_TRAIN=${is_train} ${LITE_JAVA_PATH}/native/"
    cmake ${JNI_CMAKE_ARGS} -DSUPPORT_TRAIN=${is_train} "${LITE_JAVA_PATH}/native/"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni ${local_lite_platform} failed----------------"
        exit 1
    fi
    rm -f ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/*.so*
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
    cp ./libmindspore-lite-jni.so ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
    cp ./libmindspore-lite-jni.so ${INSTALL_PREFIX}/${PKG_NAME}/runtime/lib/

    RUNTIME_LIB_DIR="${BASEPATH}/output/tmp/${PKG_NAME}/runtime/lib"
    if [ -d ${RUNTIME_LIB_DIR} ]; then
      if [ "$(ls -A ${RUNTIME_LIB_DIR})" ]; then
        cp ${RUNTIME_LIB_DIR}/*.so ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/
      fi
    fi
    CONVERTER_LIB_DIR="${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib"
    if [ -d ${CONVERTER_LIB_DIR} ]; then
      if [ "$(ls -A ${CONVERTER_LIB_DIR})" ]; then
        cp ${CONVERTER_LIB_DIR}/*.so ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/
      fi
    fi

    if [ -f "`echo ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_GLOG}`" ]; then
      cp ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/libmindspore_glog.so* ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/libmindspore_glog.so
    fi
    LIB_JPEG="libjpeg.so*"
    if [ -f "`echo ${BASEPATH}/output/tmp/${PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/${LIB_JPEG}`" ]; then
      cp ${BASEPATH}/output/tmp/${PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/${LIB_JPEG} ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/libjpeg.so
    fi
    LIB_TURBOJPEG="libturbojpeg.so*"
    if [ -f "`echo ${BASEPATH}/output/tmp/${PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/${LIB_TURBOJPEG}`" ]; then
      cp ${BASEPATH}/output/tmp/${PKG_NAME}/runtime/third_party/libjpeg-turbo/lib/${LIB_TURBOJPEG} ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/libturbojpeg.so
    fi
    LIB_OPENCV_IMGPROC="libopencv_imgproc.so*"
    if [ -f "`echo ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_OPENCV_IMGPROC}`" ]; then
      cp ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_OPENCV_IMGPROC} ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/libopencv_imgproc.so
    fi
    LIB_OPENCV_CORE="libopencv_core.so*"
    if [ -f "`echo ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_OPENCV_CORE}`" ]; then
      cp ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_OPENCV_CORE} ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/libopencv_core.so
    fi
    LIB_OPENCV_IMGCODECS="libopencv_imgcodecs.so*"
    if [ -f "`echo ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_OPENCV_IMGCODECS}`" ]; then
      cp ${BASEPATH}/output/tmp/${PKG_NAME}/tools/converter/lib/${LIB_OPENCV_IMGCODECS} ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/libopencv_imgcodecs.so
    fi

    if [[ "X$is_train" = "Xon" ]]; then
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
      cp ./libmindspore-lite-train-jni.so ${INSTALL_PREFIX}/${PKG_NAME}/runtime/lib/
      cp ./libmindspore-lite-train-jni.so ${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite/${RESOURCE_PATH_ARCH}/
    fi

    cd ${LITE_JAVA_PATH}/java
    rm -rf gradle .gradle gradlew gradlew.bat
    local gradle_version=""
    gradle_version=`gradle --version | grep Gradle | awk '{print$2}'`
    if [[ ${gradle_version} == '6.6.1' ]]; then
      gradle_command=gradle
    else
      gradle wrapper --gradle-version 6.6.1 --distribution-type all
      gradle_command=${LITE_JAVA_PATH}/java/gradlew
    fi

    # build jar
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/
    if [[ "${ENABLE_ASAN}" == "ON" || "${ENABLE_ASAN}" == "on" ]] ; then
      ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ -x test --info
    else
      if [[ "${MSLITE_ENABLE_TESTCASES}" == "ON" || "${MSLITE_ENABLE_TESTCASES}" == "on" ]] ; then
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
          ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ --info
      else
           ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ -x test --info
      fi
    fi
    cp ${LITE_JAVA_PATH}/build/lib/jar/*.jar ${INSTALL_PREFIX}/${PKG_NAME}/runtime/lib/

    # package
    cd ${INSTALL_PREFIX}
    rm -rf ${PKG_NAME}.tar.gz ${PKG_NAME}.tar.gz.sha256
    tar czf ${PKG_NAME}.tar.gz ${PKG_NAME}
    sha256sum ${PKG_NAME}.tar.gz > ${PKG_NAME}.tar.gz.sha256
    rm -rf ${LITE_JAVA_PATH}/java/${NATIVE_PATH_ARCH}/libs/
    rm -rf ${LITE_JAVA_PATH}/native/libs/${NATIVE_PATH_ARCH}/
    echo "---------------- mindspore lite jni and jar: build success ----------------"
}

build_python_wheel_package() {
  local python_version=`python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'` || true
  if [[ "${python_version}" == "3" ]]; then
    cd ${BASEPATH}/mindspore/lite/build/
    local lite_wrapper_so=`ls python/_c_lite_wrapper*.so` || true
    if [ ! -f "${lite_wrapper_so}" ]; then
      echo "error: can not find python so."
      return 0
    fi
    mkdir -pv package/mindspore_lite/lib/
    cp ../python/api/* package/mindspore_lite/
    local pkg_name=mindspore-lite-${VERSION_STR}-linux-$1
    if [[ "$1" == "x86_64" ]]; then
      local pkg_name=mindspore-lite-${VERSION_STR}-linux-x64
    fi
    if [[ ("${MSLITE_ENABLE_CLOUD_FUSION_INFERENCE}" == "on") || ("${MSLITE_ENABLE_CLOUD_INFERENCE}" == "on") ]]; then
      cp src/extendrt/*.so package/mindspore_lite/lib/
      find src/extendrt/delegate/graph_executor/litert/ -name "*.so" -exec cp '{}' package/mindspore_lite/lib/ \;
      find src/extendrt/convert/ -name "*.so" -exec cp '{}' package/mindspore_lite/lib/ \;
      if [[ "${MSLITE_ENABLE_ACL}" ]]; then
        cp src/extendrt/kernel/ascend/*.so package/mindspore_lite/lib/
        local dvpp_utils=minddata/kernels-dvpp-image/utils/libdvpp_utils.so
        if [ -f ${dvpp_utils} ]; then
          cp ${dvpp_utils} package/mindspore_lite/lib/
        fi
      fi
      if [ -f "${INSTALL_PREFIX}/${pkg_name}/runtime/lib/libtransformer-shared.so" ]; then
        cp ${INSTALL_PREFIX}/${pkg_name}/runtime/lib/libtransformer-shared.so package/mindspore_lite/lib/
      fi
      if [ -f "${INSTALL_PREFIX}/${pkg_name}/runtime/lib/libtensorrt_plugin.so" ]; then
        cp ${INSTALL_PREFIX}/${pkg_name}/runtime/lib/libtensorrt_plugin.so package/mindspore_lite/lib/
      fi
    else
      if [[ "${MSLITE_ENABLE_ACL}" ]]; then
        cp src/litert/kernel/ascend/*.so package/mindspore_lite/lib/
      fi
      cp src/*.so package/mindspore_lite/lib/
    fi
    if [ -d "${INSTALL_PREFIX}/${pkg_name}/runtime/third_party/glog" ]; then
      cp ${INSTALL_PREFIX}/${pkg_name}/runtime/third_party/glog/*.so* package/mindspore_lite/lib/
    fi
    if [ -d "${INSTALL_PREFIX}/${pkg_name}/runtime/third_party/dnnl" ]; then
      cp ${INSTALL_PREFIX}/${pkg_name}/runtime/third_party/dnnl/*.so* package/mindspore_lite/lib/
    fi
    if [ -d "${INSTALL_PREFIX}/${pkg_name}/tools/converter/lib" ]; then
      cp ${INSTALL_PREFIX}/${pkg_name}/tools/converter/lib/*.so* package/mindspore_lite/lib/
    fi
    cp python/*.so package/mindspore_lite/lib/
    mkdir -pv package/mindspore_lite/include/registry/
    if [ -d "${INSTALL_PREFIX}/${pkg_name}/runtime/include/api" ]; then
      cp -rf ${INSTALL_PREFIX}/${pkg_name}/runtime/include/api package/mindspore_lite/include/
    fi
    if [ -d "${INSTALL_PREFIX}/${pkg_name}/runtime/include/mindapi" ]; then
      cp -rf ${INSTALL_PREFIX}/${pkg_name}/runtime/include/mindapi package/mindspore_lite/include/
    fi
    if [ -f "${INSTALL_PREFIX}/${pkg_name}/tools/converter/include/registry/converter_context.h" ]; then
      cp ${INSTALL_PREFIX}/${pkg_name}/tools/converter/include/registry/converter_context.h package/mindspore_lite/include/registry/
    fi
    if [ -f "${INSTALL_PREFIX}/${pkg_name}/tools/converter/include/converter.h" ]; then
      cp ${INSTALL_PREFIX}/${pkg_name}/tools/converter/include/converter.h package/mindspore_lite/include/
    fi
    cp .commit_id package/mindspore_lite/
    echo "__version__ = '${VERSION_STR}'" > package/mindspore_lite/version.py
    cp ../python/setup.py  package/
    export TOP_DIR=${BASEPATH}
    cd package
    rm -rf dist/mindspore_lite-*.whl
    python setup.py bdist_wheel
    local minor_version=`python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'` || true
    local py_tags="cp${python_version}${minor_version}-cp${python_version}${minor_version}"
    if [[ "${minor_version}" == "7" ]]; then
      py_tags="cp37-cp37m"
    fi
    local whl_name=mindspore_lite-${VERSION_STR}-${py_tags}-linux_$1.whl
    cp dist/mindspore_lite-*.whl ${BASEPATH}/output/${whl_name}
    cd ${BASEPATH}/output/
    sha256sum ${whl_name} > ${whl_name}.sha256
  else
    echo -e "\e[31mPython3 not found, so Python API will not be compiled. \e[0m"
  fi
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

    LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DENABLE_ASAN=${ENABLE_ASAN} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"

    if [[ "$(uname)" == "Darwin" && "${local_lite_platform}" != "x86_64" ]]; then
      LITE_CMAKE_ARGS=`echo $LITE_CMAKE_ARGS | sed 's/-DCMAKE_BUILD_TYPE=Debug/-DCMAKE_BUILD_TYPE=Release/g'`
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off -DMSLITE_ENABLE_NPU=off"
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DENABLE_BITCODE=0 -G Xcode"
      CMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake
    fi

    BRANCH_NAME=nnie_3516_master_dev1
    if [[ ("${MSLITE_REGISTRY_DEVICE}" == "Hi3516D" || "${TOOLCHAIN_NAME}" == "himix200") && "${local_lite_platform}" == "arm32" ]]; then
      TOOLCHAIN_NAME="himix200"
      MSLITE_REGISTRY_DEVICE=Hi3516D
      check_Hi35xx
      MSLITE_COMPILE_TWICE=ON
      twice_target='benchmark micro_nnie'
    elif [[ "${MSLITE_REGISTRY_DEVICE}" == "Hi3559A" && "${local_lite_platform}" == "arm64" ]]; then
      TOOLCHAIN_NAME="himix100"
      check_Hi35xx
      MSLITE_COMPILE_TWICE=ON
      twice_target='benchmark micro_nnie'
    elif [[ "${MSLITE_REGISTRY_DEVICE}" == "SD3403" && "${local_lite_platform}" == "arm64" ]]; then
      TOOLCHAIN_NAME="mix210"
      MSLITE_COMPILE_TWICE=ON
      twice_target=benchmark
    elif [[ "${MSLITE_REGISTRY_DEVICE}" == "Hi3519A" && "${local_lite_platform}" == "arm32" ]]; then
      TOOLCHAIN_NAME="himix200"
      check_Hi35xx
      MSLITE_COMPILE_TWICE=ON
      twice_target='benchmark micro_nnie'
    elif [[ ("${MSLITE_ENABLE_NNIE}" == "on" || "${MSLITE_REGISTRY_DEVICE}" == "Hi3516D") && "${local_lite_platform}" == "x86_64" ]]; then
      MSLITE_REGISTRY_DEVICE=Hi3516D
    elif [[ "${MSLITE_MICRO_PLATFORM}" == cortex-m* && "${local_lite_platform}" == "x86_64" ]]; then
      TOOLCHAIN_NAME=${MSLITE_MICRO_PLATFORM}
    fi

    machine=`uname -m`
    echo "machine:${machine}."
    if [[ "${local_lite_platform}" == "arm32" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM_ARM32=on -DENABLE_NEON=on"
      if [ "$(uname)" == "Darwin" ]; then
        # CPU : iOS-aarch32
        pkg_name=mindspore-lite-${VERSION_STR}-ios-aarch32
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DARCHS=armv7;armv7s"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
      elif [[ "${TOOLCHAIN_NAME}" == "ohos-lite" ]]; then
        # Hi3516D : OpenHarmony-aarch32
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/ohos-lite.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=ohos-lite"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off"
      elif [[ "${TOOLCHAIN_NAME}" == "himix200" ]]; then
        # Hi3516D : Linux-aarch32
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/himix200.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=himix200"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      elif [[ "${TOOLCHAIN_NAME}" == "ohos" ]]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ohos-arm32
        CMAKE_TOOLCHAIN_FILE=${OHOS_NDK}/build/cmake/ohos.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DCMAKE_OHOS_NDK=${OHOS_NDK} -DOHOS_ARCH=armeabi-v7a -DOHOS_STL=c++_static -DTOOLCHAIN_NAME=${TOOLCHAIN_NAME}"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      else
        # CPU : Android-aarch32
        checkndk
        export PATH=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin:${ANDROID_NDK}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:${PATH}
        CMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=full"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=clang -DANDROID_STL=${MSLITE_ANDROID_STL}"
      fi
    elif [[ "${local_lite_platform}" == "arm64" ]]; then
      LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM_ARM64=on -DENABLE_NEON=on"
      if [ "$(uname)" == "Darwin" ]; then
        # CPU : iOS-aarch64
        pkg_name=mindspore-lite-${VERSION_STR}-ios-aarch64
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DARCHS=arm64"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
      elif [[ "${TOOLCHAIN_NAME}" == "himix100" ]]; then
        # Hi3559A :
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/himix100.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=himix100"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      elif [[ "${TOOLCHAIN_NAME}" == "mix210" ]]; then
        # SD3403 :
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/mix210.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=mix210"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      elif [[ "${TOOLCHAIN_NAME}" == "ohos" ]]; then
        pkg_name=mindspore-lite-${VERSION_STR}-ohos-aarch64
        CMAKE_TOOLCHAIN_FILE=${OHOS_NDK}/build/cmake/ohos.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DCMAKE_OHOS_NDK=${OHOS_NDK} -DOHOS_ARCH=arm64-v8a -DOHOS_STL=c++_static -DTOOLCHAIN_NAME=${TOOLCHAIN_NAME}"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off"
      else
        if [[ "${machine}" == "aarch64" ]]; then
          # CPU : Linux-aarch64
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMACHINE_LINUX_ARM64=on"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=full"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_TRAIN=off"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_GPU_BACKEND=off"
        else
          # CPU/GPU : Android-aarch64
          checkndk
          export PATH=${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin:${ANDROID_NDK}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:${PATH}
          CMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DANDROID_NATIVE_API_LEVEL=19 -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang -DANDROID_STL=${MSLITE_ANDROID_STL}"
          LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=full"
        fi
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_FP16=on"
      fi
    elif [[ "$(uname)" == "Darwin" ]]; then
       pkg_name=mindspore-lite-${VERSION_STR}-ios-simulator
       CMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake
       LITE_CMAKE_ARGS=`echo $LITE_CMAKE_ARGS | sed 's/-DCMAKE_BUILD_TYPE=Debug/-DCMAKE_BUILD_TYPE=Release/g'`
       LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM=SIMULATOR64 -DPLATFORM_ARM64=off -DENABLE_NEON=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off -DMSLITE_ENABLE_NPU=off"
       LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off"
       LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_ENABLE_TOOLS=off -DMSLITE_ENABLE_CONVERTER=off"
       LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -G Xcode .."
    else
      if [[ "${machine}" == "aarch64" ]]; then
        echo "Use the '-I arm64' command when compiling MindSpore Lite on an aarch64 architecture system."
        exit 1
      fi
      if [[ "${TOOLCHAIN_NAME}" == cortex-m* ]]; then
        CMAKE_TOOLCHAIN_FILE=${BASEPATH}/mindspore/lite/cmake/${TOOLCHAIN_NAME}.toolchain.cmake
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DTOOLCHAIN_NAME=${TOOLCHAIN_NAME}"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DPLATFORM_MCU=on"
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=off -DMSLITE_ENABLE_TRAIN=off -DMSLITE_GPU_BACKEND=off -DMSLITE_ENABLE_TOOLS=off"
      else
        # CPU : Linux-x86_64
        LITE_CMAKE_ARGS="${LITE_CMAKE_ARGS} -DMSLITE_MINDDATA_IMPLEMENT=full"
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
    if [[ ("${local_lite_platform}" == "arm64" || "${local_lite_platform}" == "arm32") && "${TOOLCHAIN_NAME}" != "ohos" ]]; then
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
          cp -r ${INSTALL_PREFIX}/mindspore*/runtime ${BASEPATH}/mindspore/lite/providers
          PKG_PATH=${INSTALL_PREFIX}/`ls ${INSTALL_PREFIX}/`
          echo "cmake ${LITE_CMAKE_ARGS} -DPKG_PATH=${PKG_PATH} ${BASEPATH}/mindspore/lite"
          cmake ${LITE_CMAKE_ARGS} -DPKG_PATH=${PKG_PATH} "${BASEPATH}/mindspore/lite"
          cmake --build "${BASEPATH}/mindspore/lite/build" --target ${twice_target} -j$THREAD_NUM
          make install
        fi
      fi
      make package
      if [[ "${local_lite_platform}" == "x86_64" && "X$CMAKE_TOOLCHAIN_FILE" == "X" ]]; then
        build_python_wheel_package "x86_64"
        if [ "${JAVA_HOME}" ]; then
          echo -e "\e[31mJAVA_HOME=$JAVA_HOME  \e[0m"
          build_lite_jni_and_jar "${CMAKE_ARGS}" "x86_64"
        else
            echo -e "\e[31mJAVA_HOME is not set, so jni and jar packages will not be compiled \e[0m"
            echo -e "\e[31mIf you want to compile the JAR package, please set $JAVA_HOME. For example: export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64 \e[0m"
        fi
      elif [[ "${local_lite_platform}" == "arm64" ]] && [[ "${machine}" == "aarch64" ]]; then
        build_python_wheel_package "aarch64"
        if [ "${JAVA_HOME}" ]; then
          echo -e "\e[31mJAVA_HOME=$JAVA_HOME  \e[0m"
          build_lite_jni_and_jar "${CMAKE_ARGS}" "aarch64"
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
          # schema files are copied in shell script since they will not generated until xcodebuild is executed.
          local schema_path=${BASEPATH}/output/mindspore-lite.framework/Headers/include/schema
          mkdir -p ${schema_path}
          cp ${BASEPATH}/mindspore/lite/build/schema/model_generated.h ${schema_path}
          cp ${BASEPATH}/mindspore/lite/build/schema/ops_generated.h ${schema_path}
          cp ${BASEPATH}/mindspore/lite/build/schema/ops_types_generated.h ${schema_path}
          local protobuf_arm_lib=${BASEPATH}/mindspore/lite/build/_deps/protobuf_arm-src/_build/libprotobuf-lite.a
          if [ -e "$protobuf_arm_lib" ]; then
            mkdir -p ${BASEPATH}/output/mindspore-lite.framework/third_party/protobuf
            cp $protobuf_arm_lib ${BASEPATH}/output/mindspore-lite.framework/third_party/protobuf/
          fi
          cd ${BASEPATH}/output
          tar -zcvf ${pkg_name}.tar.gz mindspore-lite.framework/
          sha256sum ${pkg_name}.tar.gz > ${pkg_name}.tar.gz.sha256
          rm -r mindspore-lite.framework
        else
          mv ${INSTALL_PREFIX}/*.tar.gz* ${BASEPATH}/output/
        fi

        if [[ "${local_lite_platform}" == "x86_64" ]]; then
          if [[ "${MSLITE_ENABLE_TESTCASES}" == "ON" || "${MSLITE_ENABLE_TESTCASES}" == "on" ]]; then
            mkdir -pv ${BASEPATH}/mindspore/lite/test/do_test || true
            if [[ ! "${MSLITE_ENABLE_CONVERTER}" || "${MSLITE_ENABLE_CONVERTER}" == "ON" || "${MSLITE_ENABLE_CONVERTER}" == "on" ]]; then
              cp ${INSTALL_PREFIX}/mindspore-lite*/tools/converter/lib/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            fi
            cp ${INSTALL_PREFIX}/mindspore-lite*/runtime/lib/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            if [[ "${MSLITE_ENABLE_SERVER_INFERENCE}" == "ON" || "${MSLITE_ENABLE_SERVER_INFERENCE}" == "on" ]]; then
              cp ${INSTALL_PREFIX}/mindspore-lite*/runtime/third_party/glog/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            fi
            if [[ ! "${MSLITE_ENABLE_TRAIN}" || "${MSLITE_ENABLE_TRAIN}" == "ON" || "${MSLITE_ENABLE_TRAIN}" == "on" ]]; then
              cp ${INSTALL_PREFIX}/mindspore-lite*/runtime/third_party/libjpeg-turbo/lib/*.so* ${BASEPATH}/mindspore/lite/test/do_test || true
            fi
          fi
        fi

        rm -rf ${INSTALL_PREFIX:?}/
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
        cp ./${pkg_name}/runtime/lib/libmindspore-lite*so* ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
        cp ./${pkg_name}/runtime/lib/libmindspore-lite*so* ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    fi
    local minddata_so=${pkg_name}/runtime/lib/libminddata-lite.so
    if [ -e "${minddata_so}" ]; then
       cp ./${pkg_name}/runtime/lib/libminddata-lite.so  ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
       cp ./${pkg_name}/runtime/lib/libminddata-lite.so  ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    fi
    local jpeg_turbo_dir=${pkg_name}/runtime/third_party/libjpeg-turbo/lib
    if [ -e "$jpeg_turbo_dir" ]; then
       cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
       cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
    fi
    local npu_so=${pkg_name}/runtime/third_party/hiai_ddk/lib/libhiai.so
    if [ -e "${npu_so}" ]; then
      cp ./${pkg_name}/runtime/third_party/hiai_ddk/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ./${pkg_name}/runtime/third_party/hiai_ddk/lib/*.so* ${LITE_JAVA_PATH}/native/libs/arm64-v8a/
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
        cp ./${pkg_name}/runtime/lib/libmindspore-lite*so* ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
        cp ./${pkg_name}/runtime/lib/libmindspore-lite*so* ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    fi
    local minddata_so=${pkg_name}/runtime/lib/libminddata-lite.so
    if [ -e "${minddata_so}" ]; then
       cp ./${pkg_name}/runtime/lib/libminddata-lite.so  ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
       cp ./${pkg_name}/runtime/lib/libminddata-lite.so  ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    fi
    local jpeg_turbo_dir=${pkg_name}/runtime/third_party/libjpeg-turbo/lib
    if [ -e "$jpeg_turbo_dir" ]; then
       cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
       cp ./${pkg_name}/runtime/third_party/libjpeg-turbo/lib/*.so* ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
    fi
    local npu_so=$pkg_name/runtime/third_party/hiai_ddk/lib/libhiai.so
    if [ -e "$npu_so" ]; then
      cp ./${pkg_name}/runtime/third_party/hiai_ddk/lib/*.so* ${LITE_JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ./${pkg_name}/runtime/third_party/hiai_ddk/lib/*.so* ${LITE_JAVA_PATH}/native/libs/armeabi-v7a/
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
    local gradle_version=""
    gradle_version=`gradle --version | grep Gradle | awk '{print$2}'`
    if [[ ${gradle_version} == '6.6.1' ]]; then
      gradle_command=gradle
    else
      gradle wrapper --gradle-version 6.6.1 --distribution-type all
      gradle_command=${LITE_JAVA_PATH}/java/gradlew
    fi
    # build new java api module
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/
    ${gradle_command} build -p ${LITE_JAVA_PATH}/ -x test

    # build aar
    build_lite_arm64_and_jni
    build_lite_arm32_and_jni

    cp ${LITE_JAVA_PATH}/build/libs/mindspore-lite-java.jar ${LITE_JAVA_PATH}/java/app/libs
    # build aar
    ${gradle_command} clean -p ${LITE_JAVA_PATH}/java/app
    ${gradle_command} assembleRelease  -p ${LITE_JAVA_PATH}/java/app

    cd ${LITE_JAVA_PATH}/java/app/build
    [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output/*.tar.gz*
    local minddata_so=${LITE_JAVA_PATH}/java/app/libs/arm64-v8a/libminddata-lite.so
    if [ -e "${minddata_so}" ]; then
      cp ${LITE_JAVA_PATH}/java/app/build/outputs/aar/mindspore-lite.aar ${BASEPATH}/output/mindspore-lite-full-${VERSION_STR}.aar
      sha256sum ${BASEPATH}/output/mindspore-lite-full-${VERSION_STR}.aar > ${BASEPATH}/output/mindspore-lite-full-${VERSION_STR}.aar.sha256
    else
      cp ${LITE_JAVA_PATH}/java/app/build/outputs/aar/mindspore-lite.aar ${BASEPATH}/output/mindspore-lite-${VERSION_STR}.aar
      sha256sum ${BASEPATH}/output/mindspore-lite-${VERSION_STR}.aar > ${BASEPATH}/output/mindspore-lite-${VERSION_STR}.aar.sha256
    fi
}

update_submodule()
{
  git submodule update --init graphengine
  cd "${BASEPATH}/graphengine"
  git submodule update --init metadef
  cd "${BASEPATH}"
}

build_lite_x86_64_aarch64_jar()
{
  echo "build_lite_x86_64_aarch64_jar start."
  if [[ ! -f ${SERVER_X86_64_PACKAGE_FILE} ]] || [[ ! -f ${SERVER_AARCH64_PACKAGE_FILE} ]]; then
    echo "x86_64_package aarch64_package file not exist."
    exit 1
  fi

  local x86_64_base_path=""
  x86_64_base_path=${SERVER_X86_64_PACKAGE_FILE%/*}
  local aarch64_base_path=""
  aarch64_base_path=${SERVER_AARCH64_PACKAGE_FILE%/*}
  echo "x86_64_base_path: "${x86_64_base_path}
  echo "aarch64_base_path: "${aarch64_base_path}

  local x86_64_package_name=""
  local aarch64_package_name=""
  x86_64_package_name=`basename ${SERVER_X86_64_PACKAGE_FILE} '.tar.gz'`
  aarch64_package_name=`basename ${SERVER_AARCH64_PACKAGE_FILE} '.tar.gz'`
  echo "x86_64_package_name: "${x86_64_package_name}
  echo "aarch64_package_name: "${aarch64_package_name}

  # unzip tar.gz, extract native libs(libmindspore-lite.so,libmindspore-lite-jni.so)
  rm -rf ${x86_64_base_path}/tmp ${aarch64_base_path}/tmp
  mkdir ${x86_64_base_path}/tmp ${aarch64_base_path}/tmp
  tar -zxvf ${SERVER_X86_64_PACKAGE_FILE} -C ${x86_64_base_path}/tmp
  tar -zxvf ${SERVER_AARCH64_PACKAGE_FILE} -C ${aarch64_base_path}/tmp

  LITE_JAVA_PATH=${LITE_BASEPATH}/java
  local LITE_JAVA_NATIVE_RESOURCE_PATH=${LITE_JAVA_PATH}/src/main/resources/com/mindspore/lite
  rm -f ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_x86_64/*.so*
  rm -f ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_aarch64/*.so*
  cp ${x86_64_base_path}/tmp/${x86_64_package_name}/runtime/lib/libmindspore-lite.so ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_x86_64/
  cp ${x86_64_base_path}/tmp/${x86_64_package_name}/runtime/lib/libmindspore-lite-jni.so ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_x86_64/
  cp ${aarch64_base_path}/tmp/${aarch64_package_name}/runtime/lib/libmindspore-lite.so ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_aarch64/
  cp ${aarch64_base_path}/tmp/${aarch64_package_name}/runtime/lib/libmindspore-lite-jni.so ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_aarch64/

  if [ -f "${x86_64_base_path}/tmp/${x86_64_package_name}/runtime/third_party/glog/libmindspore_glog.so.0" ]; then
    cp ${x86_64_base_path}/tmp/${x86_64_package_name}/runtime/third_party/glog/libmindspore_glog.so* ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_x86_64/libmindspore_glog.so
  fi

  if [ -f "${aarch64_base_path}/tmp/${aarch64_package_name}/runtime/third_party/glog/libmindspore_glog.so.0" ]; then
    cp ${aarch64_base_path}/tmp/${aarch64_package_name}/runtime/third_party/glog/libmindspore_glog.so* ${LITE_JAVA_NATIVE_RESOURCE_PATH}/linux_aarch64/libmindspore_glog.so
  fi
  # compile jar package
  rm -rf ${LITE_JAVA_PATH}/build
  # build jar
  local gradle_version=""
  gradle_version=`gradle --version | grep Gradle | awk '{print$2}'`
  if [[ ${gradle_version} == '6.6.1' ]]; then
    gradle_command=gradle
  else
    gradle wrapper --gradle-version 6.6.1 --distribution-type all
    gradle_command=${LITE_JAVA_PATH}/java/gradlew
  fi

  ${gradle_command} clean -p ${LITE_JAVA_PATH}/
  if [[ "${ENABLE_ASAN}" == "ON" || "${ENABLE_ASAN}" == "on" ]] ; then
    ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ -x test --info
  else
    if [[ "${MSLITE_ENABLE_TESTCASES}" == "ON" || "${MSLITE_ENABLE_TESTCASES}" == "on" ]] ; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${LITE_JAVA_PATH}/native/libs/linux_x86/
        ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ --info
    else
         ${gradle_command} releaseJar -p ${LITE_JAVA_PATH}/ -x test --info
    fi
  fi
  echo "compile jar success."

  # update jar package, compress tar.gz, update tar.gz
  cp ${LITE_JAVA_PATH}/build/lib/jar/mindspore-lite-java.jar ${x86_64_base_path}/tmp/${x86_64_package_name}/runtime/lib/ -f
  cp ${LITE_JAVA_PATH}/build/lib/jar/mindspore-lite-java.jar ${aarch64_base_path}/tmp/${aarch64_package_name}/runtime/lib/ -f

  cd ${x86_64_base_path}/tmp
  tar -zcvf ${x86_64_package_name}.tar.gz ${x86_64_package_name}/
  sha256sum ${x86_64_package_name}.tar.gz > ${x86_64_package_name}.tar.gz.sha256
  rm -f ../${x86_64_package_name}.tar.gz ../${x86_64_package_name}.tar.gz.sha256
  mv ${x86_64_package_name}.tar.gz ../
  mv ${x86_64_package_name}.tar.gz.sha256 ../

  cd ${aarch64_base_path}/tmp
  tar -zcvf ${aarch64_package_name}.tar.gz ${aarch64_package_name}/
  sha256sum ${aarch64_package_name}.tar.gz > ${aarch64_package_name}.tar.gz.sha256
  rm -f ../${aarch64_package_name}.tar.gz ../${aarch64_package_name}.tar.gz.sha256
  mv ${aarch64_package_name}.tar.gz ../
  mv ${aarch64_package_name}.tar.gz.sha256 ../

  cd ${LITE_BASEPATH}
  rm -rf ${x86_64_base_path}/tmp
  rm -rf ${aarch64_base_path}/tmp
  java -version
}

LITE_BASEPATH=$(cd "$(dirname $0)"; pwd)
echo "LITE_BASEPATH="${LITE_BASEPATH}
if [[ -z "${BASEPATH}" ]]; then
  BASEPATH=${LITE_BASEPATH}/../..
fi

INSTALL_PREFIX=${BASEPATH}/output/tmp
LITE_JAVA_PATH=${BASEPATH}/mindspore/lite/java
if [[ "${MSLITE_ENABLE_ACL}" == "on" ]]; then
    update_submodule
fi

CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_VERBOSE=${ENABLE_VERBOSE}"
if [[ "${DEBUG_MODE}" == "on" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Debug "
else
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release "
fi

if [[ "X$ENABLE_FAST_HASH_TABLE" == "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_FAST_HASH_TABLE=ON"
else
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_FAST_HASH_TABLE=OFF"
fi

get_version
CMAKE_ARGS="${CMAKE_ARGS} -DVERSION_STR=${VERSION_STR}"

if [[ "X$LITE_PLATFORM" != "Xx86_64" ]]; then
    export ENABLE_AKG="off"
fi

if [[ "X$LITE_ENABLE_AAR" = "Xon" ]]; then
    build_aar
elif [[ "X$LITE_PLATFORM" != "X" ]]; then
    build_lite
else
    echo "Invalid parameter"
fi

if [[ -n "${SERVER_X86_64_PACKAGE_FILE}" ]] && [[ -n "${SERVER_AARCH64_PACKAGE_FILE}" ]]; then
    build_lite_x86_64_aarch64_jar
fi
