#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
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
BASEPATH=$(cd "$(dirname $0)"; pwd)
PROJECT_PATH="${BASEPATH}"
CUDA_PATH=""
CUDNN_PATH=""
export BUILD_PATH="${BASEPATH}/build/"
# print usage message
usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-v] [-c on|off] [-t on|off] [-g on|off] [-h] [-s] [-b ge|cpu] [-m infer|train] \\"
  echo "              [-a on|off] [-g on|off] [-p on|off] [-i] [-L] [-R] [-D on|off] [-j[n]] [-e gpu|d|cpu] \\"
  echo "              [-P on|off] [-z [on|off]] [-M on|off] [-V 9.2|10.1] [-I] [-K]"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -r Release mode, default mode"
  echo "    -v Display build command"
  echo "    -c Enable code coverage switch, default off"
  echo "    -t Run testcases switch, default on"
  echo "    -g Use glog to output log, default on"
  echo "    -h Print usage"
  echo "    -s Install or setup"
  echo "    -b Select other backend, available: \\"
  echo "           ge:graph engine, cpu"
  echo "    -m Select mode, available: infer, train, default is infer "
  echo "    -a Enable ASAN, default off"
  echo "    -p Enable pipeline profile, default off"
  echo "    -i Enable increment building, default off"
  echo "    -L Enable load ANF-IR as input of 'infer', default off"
  echo "    -R Enable the time_line record, default off"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -e Use gpu, d or cpu"
  echo "    -P Enable dump anf graph to file in ProtoBuffer format, default on"
  echo "    -Q Enable dump end to end, default off"
  echo "    -D Enable dumping of function graph ir, default on"
  echo "    -z Compile dataset & mindrecord, default on"
  echo "    -M Enable MPI and NCCL for GPU training, default on"
  echo "    -V Specify the minimum required cuda version, default CUDA 9.2"
  echo "    -I Compile predict, default off"
  echo "    -K Compile with AKG, default off"
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

# check and set options
checkopts()
{
  # Init default values of build options
  THREAD_NUM=8
  DEBUG_MODE="off"
  VERBOSE=""
  ENABLE_COVERAGE="off"
  RUN_TESTCASES="off"
  EXECUTE_SETUP="off"
  ENABLE_BACKEND=""
  TRAIN_MODE="INFER"
  ENABLE_ASAN="off"
  ENABLE_PROFILE="off"
  INC_BUILD="off"
  ENABLE_LOAD_IR="off"
  ENABLE_TIMELINE="off"
  ENABLE_DUMP2PROTO="on"
  ENABLE_DUMPE2E="off"
  ENABLE_DUMP_IR="on"
  COMPILE_MINDDATA="on"
  ENABLE_MPI="on"
  CUDA_VERSION="9.2"
  COMPILE_PREDICT="off"
  USE_GLOG="on"
  PREDICT_PLATFORM=""
  ENABLE_AKG="off"

  # Process the options
  while getopts 'drvj:c:t:hsb:a:g:p:ie:m:I:LRP:Q:D:zM:V:K' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      d)
        DEBUG_MODE="on"
        ;;
      r)
        DEBUG_MODE="off"
        ;;
      v)
        VERBOSE="VERBOSE=1"
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      c)
        check_on_off $OPTARG c
        ENABLE_COVERAGE="$OPTARG"
        ;;
      t)
        check_on_off $OPTARG t
        RUN_TESTCASES="$OPTARG"
        ;;
      g)
        check_on_off $OPTARG g
        USE_GLOG="$OPTARG"
        ;;
      h)
        usage
        exit 0
        ;;
      s)
        EXECUTE_SETUP="on"
        ;;
      b)
        if [[ "X$OPTARG" != "Xge" && "X$OPTARG" != "Xcpu" ]]; then
          echo "Invalid value ${OPTARG} for option -b"
          usage
          exit 1
        fi
        ENABLE_BACKEND=$(echo "$OPTARG" | tr '[a-z]' '[A-Z]')
        if [[ "X$ENABLE_BACKEND" == "XGE" ]]; then
          ENABLE_GE="on"
        fi
        if [[ "X$ENABLE_BACKEND" != "XCPU" ]]; then
          ENABLE_CPU="on"
        fi
        ;;
      a)
        check_on_off $OPTARG a
        ENABLE_ASAN="$OPTARG"
        ;;
      p)
        check_on_off $OPTARG p
        ENABLE_PROFILE="$OPTARG"
        ;;
      i)
        INC_BUILD="on"
        ;;
      m)
        if [[ "X$OPTARG" != "Xinfer" && "X$OPTARG" != "Xtrain" ]]; then
          echo "Invalid value ${OPTARG} for option -m"
          usage
          exit 1
        fi
        TRAIN_MODE=$(echo "$OPTARG" | tr '[a-z]' '[A-Z]')
        ;;
      L)
        ENABLE_LOAD_IR="on"
        echo "build with enable load anf ir"
        ;;
      R)
        ENABLE_TIMELINE="on"
        echo "enable time_line record"
        ;;
      e)
        if [[ "X$OPTARG" == "Xgpu" ]]; then
          ENABLE_GPU="on"
          ENABLE_CPU="on"
        elif [[ "X$OPTARG" == "Xd" || "X$OPTARG" == "Xascend" ]]; then
          ENABLE_D="on"
          ENABLE_CPU="on"
        elif [[ "X$OPTARG" == "Xcpu" ]]; then
          ENABLE_CPU="on"
        else
          echo "Invalid value ${OPTARG} for option -e"
          usage
          exit 1
        fi
        ;;
      M)
        check_on_off $OPTARG M
        ENABLE_MPI="$OPTARG"
        ;;
      V)
        if [[ "X$OPTARG" != "X9.2" && "X$OPTARG" != "X10.1" ]]; then
          echo "Invalid value ${OPTARG} for option -V"
          usage
          exit 1
        fi
        CUDA_VERSION="$OPTARG"
        ;;
      P)
        check_on_off $OPTARG p
        ENABLE_DUMP2PROTO="$OPTARG"
        echo "enable dump anf graph to proto file"
        ;;
      Q)
        check_on_off $OPTARG Q
        ENABLE_DUMPE2E="$OPTARG"
        echo "enable dump end to end"
        ;;
      D)
        check_on_off $OPTARG D
        ENABLE_DUMP_IR="$OPTARG"
        echo "enable dump function graph ir"
        ;;
      z)
        eval ARG=\$\{$OPTIND\}
        if [[ -n $ARG && $ARG != -* ]]; then
          OPTARG=$ARG
          check_on_off $OPTARG z
          OPTIND=$((OPTIND + 1))
        else
          OPTARG=""
        fi
        if [[ "X$OPTARG" == "Xoff" ]]; then
          COMPILE_MINDDATA="off"
        fi
        ;;
      I)
        COMPILE_PREDICT="on"
        if [[ "$OPTARG" == "arm64" ]]; then
          PREDICT_PLATFORM="arm64"
        elif [[ "$OPTARG" == "x86_64" ]]; then
          PREDICT_PLATFORM="x86_64"
        else
          echo "-I parameter must be arm64 or x86_64"
          exit 1
        fi
        ;;
      K)
        ENABLE_AKG="on"
        echo "enable compile with akg"
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}
checkopts "$@"
echo "---------------- mindspore: build start ----------------"
mkdir -pv "${BUILD_PATH}/package/mindspore/lib"
git submodule update --init graphengine

build_exit()
{
    echo "$@" >&2
    stty echo
    exit 1
}

# Create building path
build_mindspore()
{
    echo "start build mindspore project."
    mkdir -pv "${BUILD_PATH}/mindspore"
    cd "${BUILD_PATH}/mindspore"
    CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH"
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_LOAD_ANF_IR=$ENABLE_LOAD_IR"
    if [[ "X$ENABLE_COVERAGE" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_COVERAGE=ON"
    fi
    if [[ "X$RUN_TESTCASES" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TESTCASES=ON"
    fi
    if [[ -n "$ENABLE_BACKEND" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_${ENABLE_BACKEND}=ON"
    fi
    if [[ -n "$TRAIN_MODE" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_${TRAIN_MODE}=ON"
    fi
    if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
    fi
    if [[ "X$ENABLE_PROFILE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PROFILE=ON"
    fi
    if [[ "X$ENABLE_TIMELINE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TIMELINE=ON"
    fi
    if [[ "X$ENABLE_DUMP2PROTO" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_PROTO=ON"
    fi
    if [[ "X$ENABLE_DUMPE2E" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_E2E=ON"
    fi
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_IR=${ENABLE_DUMP_IR}"
    if [[ "X$ENABLE_MPI" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MPI=ON"
    fi
    if [[ "X$ENABLE_D" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON"
    fi
    if [[ "X$ENABLE_GPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON -DCUDA_PATH=$CUDA_PATH -DCUDNN_PATH=$CUDNN_PATH -DMS_REQUIRE_CUDA_VERSION=${CUDA_VERSION}"
    fi
    if [[ "X$ENABLE_CPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CPU=ON"
    fi
    if [[ "X$COMPILE_MINDDATA" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MINDDATA=ON"
    fi
    if [[ "X$USE_GLOG" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DUSE_GLOG=ON"
    fi
    if [[ "X$ENABLE_AKG" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_AKG=ON"
    fi
    echo "${CMAKE_ARGS}"
    if [[ "X$INC_BUILD" = "Xoff" ]]; then
      cmake ${CMAKE_ARGS} ../..
    fi
    make ${VERBOSE} -j$THREAD_NUM
    if [[ "X$EXECUTE_SETUP" = "Xon" ]]; then
      make install
    fi
    echo "success to build mindspore project!"
}

build_predict()
{
    git submodule update --init --recursive third_party/incubator-tvm
    echo "start build predict project"

    git submodule update --init --recursive third_party/flatbuffers
    git submodule update --init --recursive third_party/googletest
    git submodule update --init --recursive third_party/protobuf

    rm -rf "${BASEPATH}/predict/build"
    mkdir -pv "${BASEPATH}/predict/build"
    rm -rf "${BASEPATH}/predict/output"
    mkdir -pv "${BASEPATH}/predict/output"

    if [[ "$PREDICT_PLATFORM" == "arm64" ]]; then
      if [ "${ANDROID_NDK}" ]; then
          echo -e "\e[31mANDROID_NDK_PATH=$ANDROID_NDK  \e[0m"
      else
          echo -e "\e[31mplease set ANDROID_NDK_PATH in environment variable for example: export ANDROID_NDK=/root/usr/android-ndk-r16b/ \e[0m"
          exit 1
      fi
    fi

    #build flatbuf
    cd "${BASEPATH}/third_party/flatbuffers"
    rm -rf build && mkdir -p build && cd build && cmake .. && make -j$THREAD_NUM
    FLATC="${BASEPATH}"/third_party/flatbuffers/build/flatc
    cd "${BASEPATH}"/predict/schema && mkdir -p "${BASEPATH}"/predict/schema/inner
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o ${BASEPATH}/predict/schema/inner

    # check LLVM_PATH
    if [ "${LLVM_PATH}" == "" ]; then
        echo "Please set LLVM_PATH in env for example export LLVM_PATH=/xxxx/bin/llvm-config"
        exit
    fi

    #build tvm
    tvm_open_source="${BASEPATH}/third_party/incubator-tvm"
    tvm_kernel_build="${BASEPATH}/predict/module/tvm_kernel"
    if [ ! -f "${tvm_kernel_build}"/incubator-tvm/build/libtvm.so ]; then
        rm -fr "${tvm_kernel_build}"/incubator-tvm
        cp -fr "${tvm_open_source}" "${tvm_kernel_build}"
        mkdir -p "${tvm_kernel_build}"/incubator-tvm/build
        patch -d "${tvm_kernel_build}"/incubator-tvm -p1 < "${BASEPATH}"/third_party/patch/predict/0001-RetBugFix-CustomRuntime_v06.patch
        cp "${tvm_kernel_build}"/lite/src/codegen/llvm/lite_rtfunc_reset.cc "${tvm_kernel_build}"/incubator-tvm/src/codegen/llvm/
        cp "${tvm_open_source}"/cmake/config.cmake "${tvm_kernel_build}"/incubator-tvm
        if [ "${LLVM_PATH}" ]; then
            sed -i "s#set(USE_LLVM .*)#set(USE_LLVM \"${LLVM_PATH}\")#g"  "${tvm_kernel_build}"/incubator-tvm/config.cmake
        else
            echo "need set LLVM_PATH in env for example export LLVM_PATH=/xxxx/bin/llvm-config"
        fi
        cd "${tvm_kernel_build}"/incubator-tvm/build
        cmake ..
        make -j$THREAD_NUM
    else
        cd "${tvm_kernel_build}"/incubator-tvm/build
        make -j$THREAD_NUM
    fi

    #gen op
    predict_tvm_op_lib_path="${BASEPATH}/predict/module/tvm_kernel/build/lib_x86"
    predict_platform="x86"
    if [[ "$PREDICT_PLATFORM" == "arm64" ]]; then
      predict_tvm_op_lib_path="${BASEPATH}/predict/module/tvm_kernel/build/lib_arm64"
      predict_platform="arm64"
    fi

    need_get_libs=true
    if [ -d "${predict_tvm_op_lib_path}" ]; then
      file_list=$(ls "${predict_tvm_op_lib_path}")
      if [ -n "${file_list}" ]; then
        libstime=$(stat -c %Y "${predict_tvm_op_lib_path}"/* | sort -u | tail -n1)
        pythontime=$(find "${BASEPATH}"/predict/module/tvm_kernel/lite/python/ -name "*.py" -exec stat -c %Y {} \; |
        sort -u | tail -n1)
        if [ "${libstime}" -ge "${pythontime}" ]; then
          need_get_libs=false
        else
          rm -fr "${predict_tvm_op_lib_path}"
        fi
      fi
    fi

    if $need_get_libs; then
       PYTHONPATH_OLD=${PYTHONPATH}
       export PYTHONPATH="${tvm_kernel_build}/incubator-tvm/python:${tvm_kernel_build}/incubator-tvm/topi/python:${tvm_kernel_build}/incubator-tvm/nnvm/python:${tvm_kernel_build}/lite/python:"
       cd "${BASEPATH}"/predict/module/tvm_kernel/lite/python/at_ops
       python3 at_gen_strip.py ${predict_platform}
       export PYTHONPATH=${PYTHONPATH_OLD}
    fi

    cd "${BASEPATH}/predict/build"
    if [[ "$PREDICT_PLATFORM" == "arm64" ]]; then
      cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
            -DANDROID_NATIVE_API_LEVEL=android-19 -DANDROID_NDK="${ANDROID_NDK}" \
            -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang" -DANDROID_STL="c++_shared" \
            -DANDROID_ABI="arm64-v8a" -DENABLE_PREDICT_ARM64=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE ..
    elif [[ "$PREDICT_PLATFORM" == "x86_64" ]]; then
      cmake ..
    fi

    make ${VERBOSE} -j$THREAD_NUM
    if [[ "$PREDICT_PLATFORM" == "x86_64" ]]; then
      cd "${BASEPATH}/predict/build/test" && ./run_tests.sh
    fi

    # copy securec include files
    mkdir -p "${BASEPATH}/predict/output/include/securec/include"
    cp "${BASEPATH}"/third_party/securec/include/* "${BASEPATH}"/predict/output/include/securec/include

    cd "${BASEPATH}/predict/output/"
    if [[ "$PREDICT_PLATFORM" == "x86_64" ]]; then
      tar -cf MSPredict-0.1.0-linux_x86_64.tar.gz include/ lib/ --warning=no-file-changed
    elif [[ "$PREDICT_PLATFORM" == "arm64" ]]; then
      tar -cf MSPredict-0.1.0-linux_aarch64.tar.gz include/ lib/ --warning=no-file-changed
    fi
    echo "success to build predict project!"
}

if [[ "X$COMPILE_PREDICT" = "Xon" ]]; then
    build_predict
    echo "---------------- mindspore: build end   ----------------"
    exit
else
    build_mindspore
fi

if [[ "X$INC_BUILD" = "Xoff" ]]; then
    if [[ "X$ENABLE_GE" = "Xon" ]]; then
        bash "${PROJECT_PATH}/package.sh" ge
    elif [[ "X$ENABLE_GPU" = "Xon" ]]; then
        bash "${PROJECT_PATH}/package.sh" ms gpu
    elif [[ "X$ENABLE_D" = "Xon" ]]; then
        bash "${PROJECT_PATH}/package.sh" ms ascend
    elif [[ "X$ENABLE_CPU" = "Xon" ]]; then
        bash "${PROJECT_PATH}/package.sh" ms cpu
    else
        bash "${PROJECT_PATH}/package.sh" debug
    fi
fi

cp -rf ${BUILD_PATH}/package/mindspore/lib ${BUILD_PATH}/../mindspore
cp -rf ${BUILD_PATH}/package/mindspore/*.so ${BUILD_PATH}/../mindspore

if [[ -d "${BUILD_PATH}/package/build" ]]; then
    rm -rf "${BUILD_PATH}/package/build"
fi
echo "---------------- mindspore: build end   ----------------"
