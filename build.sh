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
CUDA_PATH=""
export BUILD_PATH="${BASEPATH}/build/"
# print usage message
usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-v] [-c on|off] [-t ut|st] [-g on|off] [-h] [-b ge] [-m infer|train] \\"
  echo "              [-a on|off] [-p on|off] [-i] [-R] [-D on|off] [-j[n]] [-e gpu|ascend|cpu|npu] \\"
  echo "              [-P on|off] [-z [on|off]] [-M on|off] [-V 10.1|11.1|310|910] [-I arm64|arm32|x86_64] [-K] \\"
  echo "              [-B on|off] [-E] [-l on|off] [-n full|lite|off] [-T on|off] [-H on|off] \\"
  echo "              [-A [cpp|java|object-c] [-C on|off] [-o on|off] [-S on|off] [-k on|off] [-W sse|neon|avx|off] \\"
  echo "              [-L Tensor-RT path]  \\"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -r Release mode, default mode"
  echo "    -v Display build command"
  echo "    -c Enable code coverage, default off"
  echo "    -t Run testcases, default off"
  echo "    -g Use glog to output log, default on"
  echo "    -h Print usage"
  echo "    -b Select other backend, available: \\"
  echo "           ge:graph engine"
  echo "    -m Select graph engine backend mode, available: infer, train, default is infer"
  echo "    -a Enable ASAN, default off"
  echo "    -p Enable pipeline profile, print to stdout, default off"
  echo "    -R Enable pipeline profile, record to json, default off"
  echo "    -i Enable increment building, default off"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -e Use cpu, gpu, npu or ascend"
  echo "    -P Enable dump anf graph to file in ProtoBuffer format, default on"
  echo "    -D Enable dumping of function graph ir, default on"
  echo "    -z Compile dataset & mindrecord, default on"
  echo "    -n Compile minddata with mindspore lite, available: off, lite, full, lite_cv, full mode in lite train and lite_cv, wrapper mode in lite predict"
  echo "    -M Enable MPI and NCCL for GPU training, gpu default on"
  echo "    -V Specify the device version, if -e gpu, default CUDA 10.1, if -e ascend, default Ascend 910"
  echo "    -I Enable compiling mindspore lite for arm64, arm32 or x86_64, default disable mindspore lite compilation"
  echo "    -K Compile with AKG, default on"
  echo "    -B Enable debugger, default on"
  echo "    -E Enable IBVERBS for parameter server, default off"
  echo "    -l Compile with python dependency, default on"
  echo "    -A Language used by mindspore lite, default cpp"
  echo "    -T Enable on-device training, default off"
  echo "    -C Enable mindspore lite converter compilation, enabled when -I is specified, default on"
  echo "    -o Enable mindspore lite tools compilation, enabled when -I is specified, default on"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "    -k Enable make clean, clean up compilation generated cache "
  echo "    -W Enable x86_64 SSE or AVX instruction set, use [sse|avx|neon|off], default off"
  echo "    -H Enable hidden"
  echo "    -L Link and specify Tensor-RT library path, default disable Tensor-RT lib linking"
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
  RUN_CPP_ST_TESTS="off"
  ENABLE_BACKEND=""
  TRAIN_MODE="INFER"
  ENABLE_ASAN="off"
  ENABLE_PROFILE="off"
  INC_BUILD="off"
  ENABLE_TIMELINE="off"
  ENABLE_DUMP2PROTO="on"
  ENABLE_DUMP_IR="on"
  COMPILE_MINDDATA="on"
  COMPILE_MINDDATA_LITE="lite_cv"
  ENABLE_MPI="off"
  CUDA_VERSION="10.1"
  COMPILE_LITE="off"
  LITE_PLATFORM=""
  SUPPORT_TRAIN="off"
  USE_GLOG="on"
  ENABLE_AKG="on"
  ENABLE_ACL="off"
  ENABLE_DEBUGGER="on"
  ENABLE_IBVERBS="off"
  ENABLE_PYTHON="on"
  ENABLE_GPU="off"
  ENABLE_VERBOSE="off"
  ENABLE_TOOLS="on"
  ENABLE_CONVERTER="on"
  LITE_LANGUAGE="cpp"
  ENABLE_GITEE="off"
  ANDROID_STL="c++_shared"
  ENABLE_MAKE_CLEAN="off"
  X86_64_SIMD="off"
  ARM_SIMD="off"
  DEVICE_VERSION=""
  DEVICE=""
  ENABLE_NPU="off"
  ENABLE_HIDDEN="on"
  LITE_ENABLE_GPU=""
  TENSORRT_HOME=""
  # Process the options
  while getopts 'drvj:c:t:hsb:a:g:p:ie:m:l:I:RP:D:zM:V:K:B:En:T:A:C:o:S:k:W:H:L:' opt
  do
    CASE_SENSIVE_ARG=${OPTARG}
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      d)
        DEBUG_MODE="on"
        ;;
      n)
        if [[ "X$OPTARG" == "Xoff" || "X$OPTARG" == "Xlite" || "X$OPTARG" == "Xfull" || "X$OPTARG" == "Xlite_cv"  || "X$OPTARG" == "Xwrapper" ]]; then
          COMPILE_MINDDATA_LITE="$OPTARG"
        else
          echo "Invalid value ${OPTARG} for option -n"
          usage
          exit 1
        fi
        ;;
      r)
        DEBUG_MODE="off"
        ;;
      v)
        ENABLE_VERBOSE="on"
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
        if [[ "X$OPTARG" == "Xon" || "X$OPTARG" == "Xut" ]]; then
          RUN_TESTCASES="on"
        elif [[ "X$OPTARG" == "Xoff" ]]; then
          RUN_TESTCASES="off"
        elif [[ "X$OPTARG" == "Xst" ]]; then
          RUN_CPP_ST_TESTS="on"
        else
          echo "Invalid value ${OPTARG} for option -t"
          usage
          exit 1
        fi
        ;;
      g)
        check_on_off $OPTARG g
        USE_GLOG="$OPTARG"
        ;;
      h)
        usage
        exit 0
        ;;
      b)
        if [[ "X$OPTARG" != "Xge" && "X$OPTARG" != "Xcpu" ]]; then
          echo "Invalid value ${OPTARG} for option -b"
          usage
          exit 1
        fi
        ENABLE_BACKEND=$(echo "$OPTARG" | tr '[a-z]' '[A-Z]')
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
      l)
        check_on_off $OPTARG l
        ENABLE_PYTHON="$OPTARG"
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
      R)
        ENABLE_TIMELINE="on"
        echo "enable time_line record"
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      k)
        check_on_off $OPTARG k
        ENABLE_MAKE_CLEAN="$OPTARG"
        echo "enable make clean"
        ;;
      e)
        DEVICE=$OPTARG
        ;;
      M)
        check_on_off $OPTARG M
        ENABLE_MPI="$OPTARG"
        ;;
      V)
        DEVICE_VERSION=$OPTARG
        ;;
      P)
        check_on_off $OPTARG p
        ENABLE_DUMP2PROTO="$OPTARG"
        echo "enable dump anf graph to proto file"
        ;;
      D)
        check_on_off $OPTARG D
        ENABLE_DUMP_IR="$OPTARG"
        echo "enable dump function graph ir"
        ;;
      z)
        eval ARG=\$\{$OPTIND\}
        if [[ -n "$ARG" && "$ARG" != -* ]]; then
          OPTARG="$ARG"
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
        COMPILE_LITE="on"
        if [[ "$OPTARG" == "arm64" ]]; then
          ENABLE_CONVERTER="off"
          RUN_TESTCASES="on"
          LITE_PLATFORM="arm64"
        elif [[ "$OPTARG" == "arm32" ]]; then
          ENABLE_CONVERTER="off"
          RUN_TESTCASES="on"
          LITE_PLATFORM="arm32"
        elif [[ "$OPTARG" == "x86_64" ]]; then
          ENABLE_CONVERTER="on"
          RUN_TESTCASES="on"
          LITE_PLATFORM="x86_64"
        else
          echo "-I parameter must be arm64、arm32 or x86_64"
          exit 1
        fi
        ;;
      K)
        ENABLE_AKG="on"
        echo "enable compile with akg"
        ;;
      B)
        check_on_off $OPTARG B
        ENABLE_DEBUGGER="$OPTARG"
        ;;
      E)
        ENABLE_IBVERBS="on"
        echo "enable IBVERBS for parameter server"
        ;;
      T)
        check_on_off $OPTARG T
        SUPPORT_TRAIN=$OPTARG
        COMPILE_MINDDATA_LITE="full"
        echo "support train on device "
        ;;
      A)
        COMPILE_LITE="on"
        if [[ "$OPTARG" == "cpp" ]]; then
          LITE_LANGUAGE="cpp"
          ANDROID_STL="c++_shared"
        elif [[ "$OPTARG" == "java" ]]; then
          LITE_LANGUAGE="java"
          ENABLE_CONVERTER="off"
          ANDROID_STL="c++_static"
          RUN_TESTCASES="off"
          ENABLE_TOOLS="off"
        elif [[ "$OPTARG" == "object-c" ]]; then
          LITE_LANGUAGE="object-c"
        else
          echo "-A parameter must be cpp, java or object-c"
          exit 1
        fi
        ;;
      C)
        check_on_off $OPTARG C
        ENABLE_CONVERTER="$OPTARG"
        ;;
      o)
        check_on_off $OPTARG o
        ENABLE_TOOLS="$OPTARG"
        ;;
      W)
        if [[ "$OPTARG" != "sse" && "$OPTARG" != "off" && "$OPTARG" != "avx" && "$OPTARG" != "neon" ]]; then
          echo "Invalid value ${OPTARG} for option -W, -W parameter must be sse|neon|avx|off"
          usage
          exit 1
        fi
        if [[ "$OPTARG" == "sse" || "$OPTARG" == "avx" ]]; then
          X86_64_SIMD="$OPTARG"
        fi
        if [[ "$OPTARG" == "neon" ]]; then
          ARM_SIMD="$OPTARG"
        fi
        ;;
      H)
        check_on_off $OPTARG H
        ENABLE_HIDDEN="$OPTARG"
        echo "${OPTARG} hidden"
        ;;
      L)
        ENABLE_TRT="on"
        TENSORRT_HOME="$CASE_SENSIVE_ARG"
        echo "Link Tensor-RT library. Path: ${CASE_SENSIVE_ARG}"
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done

  if [[ "X$RUN_TESTCASES" == "Xon" && "X$DEVICE" != "X" && "X$COMPILE_LITE" != "Xon" ]]; then
    echo "WARNING:Option -e can't be set while option -t on/ut is set, reset device to empty."
    DEVICE=""
  fi

  # Parse device
  # Process build option
  if [[ "X$DEVICE" == "Xgpu" ]]; then
    LITE_ENABLE_GPU="opencl"
    ENABLE_GPU="on"
    ENABLE_CPU="on"
    ENABLE_MPI="on"
    # version default 10.1
    if [[ "X$DEVICE_VERSION" == "X" ]]; then
      DEVICE_VERSION=10.1
    fi
    if [[ "X$DEVICE_VERSION" != "X11.1" && "X$DEVICE_VERSION" != "X10.1" ]]; then
      echo "Invalid value ${DEVICE_VERSION} for option -V"
      usage
      exit 1
    fi
    CUDA_VERSION="$DEVICE_VERSION"
  elif [[ "X$DEVICE" == "Xd" || "X$DEVICE" == "Xascend" ]]; then
    # version default 910
    if [[ "X$DEVICE_VERSION" == "X" ]]; then
      DEVICE_VERSION=910
    fi
    if [[ "X$DEVICE_VERSION" == "X310" ]]; then
      ENABLE_ACL="on"
    elif [[ "X$DEVICE_VERSION" == "X910" ]]; then
      ENABLE_D="on"
      ENABLE_CPU="on"
    else
      echo "Invalid value ${DEVICE_VERSION} for option -V"
      usage
      exit 1
    fi
  elif [[ "X$DEVICE" == "Xnpu" ]]; then
    ENABLE_NPU="on"
    ENABLE_CPU="on"
  elif [[ "X$DEVICE" == "Xcpu" ]]; then
    ENABLE_CPU="on"
  elif [[ "X$DEVICE" == "Xopencl" ]]; then
    LITE_ENABLE_GPU="opencl"
  elif [[ "X$DEVICE" == "Xvulkan" ]]; then
    LITE_ENABLE_GPU="vulkan"
  elif [[ "X$DEVICE" == "Xcuda" ]]; then
    LITE_ENABLE_GPU="cuda"
  elif [[ "X$DEVICE" == "X" ]]; then
    :
  else
    echo "Invalid value ${DEVICE} for option -e"
    usage
    exit 1
  fi
}

checkopts "$@"
echo "---------------- MindSpore: build start ----------------"
mkdir -pv "${BUILD_PATH}/package/mindspore/lib"
git submodule update --init graphengine
cd "${BASEPATH}/graphengine"
git submodule update --init metadef
cd "${BASEPATH}"
if [[ "X$ENABLE_AKG" = "Xon" ]] && [[ "X$ENABLE_D" = "Xon" || "X$ENABLE_GPU" = "Xon" ]]; then
    git submodule update --init --recursive akg
fi


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
    CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH -DX86_64_SIMD=${X86_64_SIMD}"
    if [[ "X$ENABLE_COVERAGE" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_COVERAGE=ON"
    fi
    if [[ "X$RUN_TESTCASES" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TESTCASES=ON"
    fi
    if [[ "X$RUN_CPP_ST_TESTS" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CPP_ST=ON"
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
    if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
    fi
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_IR=${ENABLE_DUMP_IR}"
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PYTHON=${ENABLE_PYTHON}"
    if [[ "X$ENABLE_MPI" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MPI=ON"
    fi
    if [[ "X$ENABLE_D" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON"
    fi
    if [[ "X$ENABLE_GPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON -DUSE_CUDA=ON -DCUDA_PATH=$CUDA_PATH -DMS_REQUIRE_CUDA_VERSION=${CUDA_VERSION}"
    fi
    if [[ "X$ENABLE_CPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CPU=ON -DX86_64_SIMD=${X86_64_SIMD} -DARM_SIMD=${ARM_SIMD}"
    fi
    if [[ "X$COMPILE_MINDDATA" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MINDDATA=ON"
    fi
    if [[ "X$USE_GLOG" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DUSE_GLOG=ON"
    fi
    if [[ "X$ENABLE_AKG" = "Xon" ]] && [[ "X$ENABLE_D" = "Xon" || "X$ENABLE_GPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_AKG=ON"
    fi
    if [[ "X$ENABLE_ACL" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ACL=ON"
    fi
    if [[ "X$ENABLE_DEBUGGER" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DEBUGGER=ON"
    fi

    if [[ "X$ENABLE_IBVERBS" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_IBVERBS=ON"
    fi
    if [[ "X$ENABLE_HIDDEN" = "Xoff" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_HIDDEN=OFF"
    fi
    if [[ "X$ENABLE_TRT" == "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DTENSORRT_HOME=${TENSORRT_HOME}"
    fi
    echo "${CMAKE_ARGS}"
    if [[ "X$INC_BUILD" = "Xoff" ]]; then
      cmake ${CMAKE_ARGS} ../..
    fi
    if [[ -n "$VERBOSE" ]]; then
      CMAKE_VERBOSE="--verbose"
    fi
    cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
    echo "success building mindspore project!"
}

checkndk() {
    if [ "${ANDROID_NDK}" ]; then
        echo -e "\e[31mANDROID_NDK=$ANDROID_NDK  \e[0m"
    else
        echo -e "\e[31mplease set ANDROID_NDK in environment variable for example: export ANDROID_NDK=/root/usr/android-ndk-r20b/ \e[0m"
        exit 1
    fi
}

checkddk() {
    if [ "${HWHIAI_DDK}" ]; then
        echo -e "\e[31mHWHIAI_DDK=$HWHIAI_DDK  \e[0m"
    else
        echo -e "\e[31mplease set HWHIAI_DDK in environment variable for example: export HWHIAI_DDK=/root/usr/hwhiai-ddk-100.500.010.010/ \e[0m"
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

build_lite()
{
    [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output
    get_version
    echo "============ Start building MindSpore Lite ${VERSION_STR} ============"
    local local_lite_platform=${LITE_PLATFORM}
    local local_lite_inc_build=${INC_BUILD}
    local local_lite_enable_gpu=${LITE_ENABLE_GPU}
    local local_lite_enable_npu=${ENABLE_NPU}

    if [[ "${LITE_LANGUAGE}" == "java" ]]; then
      if [[ "X$1" != "X" ]]; then
        local_lite_platform=$1
      else
        local_lite_platform=""
      fi
      if [[ "X$2" != "X" ]]; then
        local_lite_inc_build=$2
      else
        local_lite_inc_build=""
      fi
      if [[ "X$3" != "X" ]]; then
        local_lite_enable_gpu=$3
      else
        local_lite_enable_gpu=""
      fi
      mkdir -p ${BASEPATH}/mindspore/lite/build/java
      cd ${BASEPATH}/mindspore/lite/build/
      [ -n "${BASEPATH}" ] && find . -maxdepth 1 | grep -v java | grep '/' | xargs -I {} rm -rf {}
    fi
    if [[ "${LITE_LANGUAGE}" == "cpp"  ]]; then
      if [[ "${DEVICE}" == "" ]]; then
        if [[ "${local_lite_platform}" == "arm64" || "${local_lite_platform}" == "arm32" ]]; then
          local_lite_enable_npu="on"
        fi
        if [[ "${local_lite_platform}" == "arm64" ]]; then
          local_lite_enable_gpu="opencl"
        fi
      fi

      if [[ "${local_lite_inc_build}" == "off" ]]; then
        [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/mindspore/lite/build
      fi
      mkdir -pv ${BASEPATH}/mindspore/lite/build
    fi

    if [ "${local_lite_enable_npu}" == "on" ]; then
      checkddk
    fi

    cd ${BASEPATH}/mindspore/lite/build
    write_commit_file
    BUILD_TYPE="Release"
    if [[ "${DEBUG_MODE}" == "on" ]]; then
      BUILD_TYPE="Debug"
    fi

    if [[ "${local_lite_platform}" == "arm64" ]]; then
      if [ "${OS}" == "Darwin" ]; then
        cmake -DCMAKE_BUILD_TYPE="Release" \
              -DCMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake \
              -DARCHS="arm64" -DENABLE_BITCODE=0 \
              -DSUPPORT_TRAIN="off" \
              -DENABLE_MINDRT=off \
              -DPLATFORM_ARM64="on" -DENABLE_NEON=on -DENABLE_FP16="on" \
              -DENABLE_TOOLS="on" -DENABLE_CONVERTER="off" -DBUILD_TESTCASES="off" \
              -DSUPPORT_GPU="" -DSUPPORT_NPU="off" -DENABLE_V0="on" \
              -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DBUILD_MINDDATA="" -DENABLE_ASAN=${ENABLE_ASAN} \
              -DCMAKE_INSTALL_PREFIX=${BUILD_PATH}/output/tmp \
              -G Xcode ..
      else
        checkndk
        cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
              -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
              -DANDROID_STL=${ANDROID_STL} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSUPPORT_TRAIN=${SUPPORT_TRAIN}                     \
              -DPLATFORM_ARM64=on -DENABLE_NEON=on -DENABLE_FP16="on"      \
              -DENABLE_TOOLS=${ENABLE_TOOLS} -DENABLE_CONVERTER=${ENABLE_CONVERTER} -DBUILD_TESTCASES=${RUN_TESTCASES} \
              -DSUPPORT_GPU=${local_lite_enable_gpu} -DSUPPORT_NPU=${local_lite_enable_npu} -DENABLE_V0=on \
              -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} -DENABLE_ASAN=${ENABLE_ASAN} \
              -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp -DMS_VERSION_MAJOR=${VERSION_MAJOR}                           \
              -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
              "${BASEPATH}/mindspore/lite"
      fi
    elif [[ "${local_lite_platform}" == "arm32" ]]; then
      if [ "${OS}" == "Darwin" ]; then
        cmake -DCMAKE_BUILD_TYPE="Release" \
              -DCMAKE_TOOLCHAIN_FILE=${BASEPATH}/cmake/lite_ios.cmake \
              -DARCHS="armv7;armv7s" -DENABLE_BITCODE=0 \
              -DSUPPORT_TRAIN="off" \
              -DENABLE_MINDRT=off \
              -DPLATFORM_ARM32="on" -DENABLE_NEON=on \
              -DENABLE_TOOLS="on" -DENABLE_CONVERTER="off" -DBUILD_TESTCASES="off" \
              -DSUPPORT_GPU="" -DSUPPORT_NPU="off" -DENABLE_V0=on \
              -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DBUILD_MINDDATA="" -DENABLE_ASAN=${ENABLE_ASAN} \
              -DCMAKE_INSTALL_PREFIX=${BUILD_PATH}/output/tmp \
              -G Xcode ..
      else
        checkndk
        cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
              -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="clang"                      \
              -DANDROID_STL=${ANDROID_STL}  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                                                      \
              -DPLATFORM_ARM32=on -DENABLE_NEON=on -DSUPPORT_TRAIN=${SUPPORT_TRAIN}  -DENABLE_FP16="on" \
              -DENABLE_TOOLS=${ENABLE_TOOLS} -DENABLE_CONVERTER=${ENABLE_CONVERTER} -DBUILD_TESTCASES=${RUN_TESTCASES} \
              -DSUPPORT_GPU=${local_lite_enable_gpu} -DSUPPORT_NPU=${local_lite_enable_npu} -DENABLE_V0=on \
              -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} -DENABLE_ASAN=${ENABLE_ASAN} \
              -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp -DMS_VERSION_MAJOR=${VERSION_MAJOR}                           \
              -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
               "${BASEPATH}/mindspore/lite"
      fi
    else
        cmake -DPLATFORM_ARM64=off -DSUPPORT_TRAIN=${SUPPORT_TRAIN}   \
        -DENABLE_TOOLS=${ENABLE_TOOLS} -DENABLE_CONVERTER=${ENABLE_CONVERTER} -DBUILD_TESTCASES=${RUN_TESTCASES} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSUPPORT_GPU=${local_lite_enable_gpu} -DSUPPORT_NPU=${local_lite_enable_npu} \
        -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} -DENABLE_V0=on -DENABLE_ASAN=${ENABLE_ASAN} \
        -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp  \
        -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
        -DENABLE_VERBOSE=${ENABLE_VERBOSE} -DX86_64_SIMD=${X86_64_SIMD} "${BASEPATH}/mindspore/lite"
    fi
    if [ "${OS}" == "Darwin" ]; then
      xcodebuild ONLY_ACTIVE_ARCH=NO -configuration Release -scheme mindspore_lite -target mindspore_lite -sdk iphoneos -quiet
    else
      make -j$THREAD_NUM && make install && make package
    fi
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build failed ----------------"
        exit 1
    else
        if [[ "${LITE_LANGUAGE}" == "cpp"  ]]; then
          if [ "${OS}" == "Darwin" ]; then
            mkdir -p ${BASEPATH}/output
            cp -r ${BASEPATH}/mindspore/lite/build/src/Release-iphoneos/mindspore_lite.framework ${BASEPATH}/output/mindspore_lite.framework
            cd ${BASEPATH}/output
            tar -zcvf mindspore_lite.framework-${VERSION_STR}-${local_lite_platform}.tar.gz mindspore_lite.framework/
            sha256sum mindspore_lite.framework-${VERSION_STR}-${local_lite_platform}.tar.gz > \
                      mindspore_lite.framework-${VERSION_STR}-${local_lite_platform}.tar.gz.sha256
            rm -r mindspore_lite.framework
          else
            mv ${BASEPATH}/output/tmp/*.tar.gz* ${BASEPATH}/output/
          fi
        elif [[ "${LITE_LANGUAGE}" == "java" ]]; then
          mv ${BASEPATH}/output/tmp/*.tar.gz* ${BASEPATH}/mindspore/lite/build/java
        fi
        [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/output/tmp/
        echo "---------------- mindspore lite: build success ----------------"
        if [[ "X$LITE_LANGUAGE" = "Xcpp" ]]; then
            exit 0
        fi
        if [ "${OS}" == "Darwin" ]; then
          exit 0
        fi
    fi
}

build_lite_java_arm64() {
    # build mindspore-lite arm64
    local local_pkg_name=mindspore-lite-${VERSION_STR}-inference-android-aarch64
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
        local_pkg_name=mindspore-lite-${VERSION_STR}-train-android-aarch64
    fi
    if [[ "X$INC_BUILD" == "Xoff" ]] || [[ ! -f "${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}.tar.gz" ]]; then
      if [[ "X${DEVICE}" == "Xcpu" ]]; then
          build_lite "arm64" "off" ""
      elif [[ "X${DEVICE}" == "Xnpu" ]]; then
          echo "NPU only support c++."
          exit 1
      else
          build_lite "arm64" "off" "opencl"
      fi
    fi
    # copy arm64 so
    cd ${BASEPATH}/mindspore/lite/build/java/
    [ -n "${local_pkg_name}" ] && rm -rf ${local_pkg_name}
    tar -zxvf ${local_pkg_name}.tar.gz
    [ -n "${JAVA_PATH}" ] && rm -rf ${JAVA_PATH}/java/app/libs/arm64-v8a/
    mkdir -p ${JAVA_PATH}/java/app/libs/arm64-v8a/
    mkdir -p ${JAVA_PATH}/native/libs/arm64-v8a/
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmindspore-lite-train.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmindspore-lite-train.so ${JAVA_PATH}/native/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmslite_kernel_reg.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmslite_kernel_reg.so ${JAVA_PATH}/native/libs/arm64-v8a/

      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libminddata-lite.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libminddata-lite.so ${JAVA_PATH}/native/libs/arm64-v8a/

      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/third_party/libjpeg-turbo/lib/*.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/third_party/libjpeg-turbo/lib/*.so ${JAVA_PATH}/native/libs/arm64-v8a/
    else
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmindspore-lite.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmindspore-lite.so ${JAVA_PATH}/native/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmslite_kernel_reg.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmslite_kernel_reg.so ${JAVA_PATH}/native/libs/arm64-v8a/
    fi
    [ -n "${local_pkg_name}" ] && rm -rf ${local_pkg_name}
}

build_lite_java_arm32() {
    # build mindspore-lite arm32
    local local_pkg_name=mindspore-lite-${VERSION_STR}-inference-android-aarch32
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
        local_pkg_name=mindspore-lite-${VERSION_STR}-train-android-aarch32
    fi
    if [[ "X$INC_BUILD" == "Xoff" ]] || [[ ! -f "${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}.tar.gz" ]]; then
      build_lite  "arm32" "off" ""
    fi
    # copy arm32 so
    cd ${BASEPATH}/mindspore/lite/build/java/
    [ -n "${local_pkg_name}" ] && rm -rf ${local_pkg_name}
    tar -zxvf ${local_pkg_name}.tar.gz
    [ -n "${JAVA_PATH}" ] && rm -rf ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    mkdir -p ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    mkdir -p ${JAVA_PATH}/native/libs/armeabi-v7a/
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmindspore-lite-train.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmindspore-lite-train.so ${JAVA_PATH}/native/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmslite_kernel_reg.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmslite_kernel_reg.so ${JAVA_PATH}/native/libs/armeabi-v7a/

      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libminddata-lite.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libminddata-lite.so ${JAVA_PATH}/native/libs/armeabi-v7a/

      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/third_party/libjpeg-turbo/lib/*.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/third_party/libjpeg-turbo/lib/*.so ${JAVA_PATH}/native/libs/armeabi-v7a/
    else
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmindspore-lite.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmindspore-lite.so ${JAVA_PATH}/native/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmslite_kernel_reg.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmslite_kernel_reg.so ${JAVA_PATH}/native/libs/armeabi-v7a/
    fi
    [ -n "${local_pkg_name}" ] && rm -rf ${local_pkg_name}
}

build_lite_java_x86() {
    # build mindspore-lite x86
    local inference_or_train=inference
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
      inference_or_train=train
    fi
    local local_pkg_name=mindspore-lite-${VERSION_STR}-${inference_or_train}-linux-x64
    if [[ "X$INC_BUILD" == "Xoff" ]] || [[ ! -f "${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}.tar.gz" ]]; then
      build_lite "x86_64" "off" ""
    fi
    # copy x86 so
    cd ${BASEPATH}/mindspore/lite/build/java
    [ -n "${local_pkg_name}" ] && rm -rf ${local_pkg_name}
    tar -zxvf ${local_pkg_name}.tar.gz
    [ -n "${JAVA_PATH}" ] && rm -rf ${JAVA_PATH}/java/linux_x86/libs/
    mkdir -p ${JAVA_PATH}/java/linux_x86/libs/
    mkdir -p ${JAVA_PATH}/native/libs/linux_x86/
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmindspore-lite-train.so ${JAVA_PATH}/java/linux_x86/libs/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmindspore-lite-train.so ${JAVA_PATH}/native/libs/linux_x86/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmslite_kernel_reg.so ${JAVA_PATH}/java/linux_x86/libs/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libmslite_kernel_reg.so ${JAVA_PATH}/native/libs/linux_x86/

      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libminddata-lite.so ${JAVA_PATH}/java/linux_x86/libs/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/lib/libminddata-lite.so ${JAVA_PATH}/native/libs/linux_x86/

      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/third_party/libjpeg-turbo/lib/*.so* ${JAVA_PATH}/java/linux_x86/libs/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/train/third_party/libjpeg-turbo/lib/*.so* ${JAVA_PATH}/native/libs/linux_x86/
    else
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmindspore-lite.so ${JAVA_PATH}/java/linux_x86/libs/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmindspore-lite.so ${JAVA_PATH}/native/libs/linux_x86/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmslite_kernel_reg.so ${JAVA_PATH}/java/linux_x86/libs/
      cp ${BASEPATH}/mindspore/lite/build/java/${local_pkg_name}/inference/lib/libmslite_kernel_reg.so ${JAVA_PATH}/native/libs/linux_x86/
    fi
    [ -n "${local_pkg_name}" ] && rm -rf ${local_pkg_name}
}

build_jni_arm64() {
    # build jni so
    cd "${BASEPATH}/mindspore/lite/build"
    [ -n "${BASEPATH}" ] && rm -rf java/jni
    mkdir -pv java/jni
    cd java/jni
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DANDROID_STL="c++_static" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
          -DSUPPORT_TRAIN=${SUPPORT_TRAIN} -DPLATFORM_ARM64=on "${JAVA_PATH}/native/"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni arm64 failed----------------"
        exit 1
    fi
    mkdir -p ${JAVA_PATH}/java/app/libs/arm64-v8a/
    cp ${BASEPATH}/mindspore/lite/build/java/jni/libmindspore-lite-jni.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
    mkdir -p ${JAVA_PATH}/native/libs/arm64-v8a/
    cp ${BASEPATH}/mindspore/lite/build/java/jni/libmindspore-lite-jni.so ${JAVA_PATH}/native/libs/arm64-v8a/
}

build_jni_arm32() {
    # build jni so
    cd "${BASEPATH}/mindspore/lite/build"
    [ -n "${BASEPATH}" ] && rm -rf java/jni
    mkdir -pv java/jni
    cd java/jni
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DANDROID_STL="c++_static" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
          -DSUPPORT_TRAIN=${SUPPORT_TRAIN} -DPLATFORM_ARM32=on "${JAVA_PATH}/native"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni arm32 failed----------------"
        exit 1
    fi
    mkdir -p ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    cp ${BASEPATH}/mindspore/lite/build/java/jni/libmindspore-lite-jni.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    mkdir -p ${JAVA_PATH}/native/libs/armeabi-v7a/
    cp ${BASEPATH}/mindspore/lite/build/java/jni/libmindspore-lite-jni.so ${JAVA_PATH}/native/libs/armeabi-v7a/
}

build_jni_x86_64() {
    # build jni so
    cd "${BASEPATH}/mindspore/lite/build"
    [ -n "${BASEPATH}" ] && rm -rf java/jni
    mkdir -pv java/jni
    cd java/jni
    cmake -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
        -DENABLE_VERBOSE=${ENABLE_VERBOSE} -DSUPPORT_TRAIN=${SUPPORT_TRAIN} "${JAVA_PATH}/native/"
    make -j$THREAD_NUM
    if [[ $? -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni x86_64 failed----------------"
        exit 1
    fi
    mkdir -p ${JAVA_PATH}/java/linux_x86/libs/
    cp ${BASEPATH}/mindspore/lite/build/java/jni/libmindspore-lite-jni.so ${JAVA_PATH}/java/linux_x86/libs/
    mkdir -p ${JAVA_PATH}/native/libs/linux_x86/
    cp ${BASEPATH}/mindspore/lite/build/java/jni/libmindspore-lite-jni.so ${JAVA_PATH}/native/libs/linux_x86/
}

check_java_home() {
    if [ "${JAVA_PATH}" ]; then
        echo -e "\e[31mJAVA_HOME=$JAVA_HOME  \e[0m"
    else
        echo -e "\e[31mplease set $JAVA_HOME in environment variable for example: export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64 \e[0m"
        exit 1
    fi
}

build_java() {
    JAVA_PATH=${BASEPATH}/mindspore/lite/java
    get_version
    if [[ "X${INC_BUILD}" == "Xoff" ]]; then
        [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/mindspore/lite/build
    fi
    # build common module
    cd ${JAVA_PATH}/java/common
    gradle clean
    gradle build

    # build aar
    build_lite_java_arm64
    build_jni_arm64
    build_lite_java_arm32
    build_jni_arm32

    mkdir -p ${JAVA_PATH}/java/linux_x86/libs
    cp ${JAVA_PATH}/java/common/build/libs/mindspore-lite-java-common.jar ${JAVA_PATH}/java/app/libs
    cd ${JAVA_PATH}/java/app
    gradle clean
    gradle build

    gradle publish -PLITE_VERSION=${VERSION_STR}

    cd ${JAVA_PATH}/java/app/build
    zip -r mindspore-lite-maven-${VERSION_STR}.zip mindspore

    local inference_or_train=inference
    if [[ "X$SUPPORT_TRAIN" = "Xon" ]]; then
      inference_or_train=train
    fi

    # build linux x86 jar
    check_java_home
    build_lite_java_x86
    build_jni_x86_64

    mkdir -p ${JAVA_PATH}/java/linux_x86/libs
    cp ${JAVA_PATH}/java/common/build/libs/mindspore-lite-java-common.jar ${JAVA_PATH}/java/linux_x86/libs/
    # build java
    cd ${JAVA_PATH}/java/linux_x86/
    gradle clean
    gradle releaseJar
    # install and package
    mkdir -p ${JAVA_PATH}/java/linux_x86/build/lib
    cp ${JAVA_PATH}/java/linux_x86/libs/*.so* ${JAVA_PATH}/java/linux_x86/build/lib/jar
    cd ${JAVA_PATH}/java/linux_x86/build/

    local local_pkg_name=mindspore-lite-${VERSION_STR}-${inference_or_train}-linux-x64-jar
    cp -r ${JAVA_PATH}/java/linux_x86/build/lib ${JAVA_PATH}/java/linux_x86/build/${local_pkg_name}
    tar czvf ${local_pkg_name}.tar.gz ${local_pkg_name}
    # copy output
    cp ${JAVA_PATH}/java/app/build/mindspore-lite-maven-${VERSION_STR}.zip ${BASEPATH}/output
    cp ${local_pkg_name}.tar.gz ${BASEPATH}/output

    cd ${BASEPATH}/output
    [ -n "${BASEPATH}" ] && rm -rf ${BASEPATH}/mindspore/lite/build/java/mindspore-lite-${VERSION_STR}-${inference_or_train}-linux-x64
    exit 0
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}/mindspore"
  cmake --build . --target clean
}

OS=$(uname)
if [[ "X$COMPILE_LITE" = "Xon" ]]; then
  if [[ "X$LITE_LANGUAGE" = "Xjava" ]]; then
    build_java
  else
    build_lite
  fi
else
    build_mindspore
fi

if [[ "X$ENABLE_MAKE_CLEAN" = "Xon" ]]; then
  make_clean
fi

cp -rf ${BUILD_PATH}/package/mindspore/lib ${BUILD_PATH}/../mindspore
cp -rf ${BUILD_PATH}/package/mindspore/*.so ${BUILD_PATH}/../mindspore


echo "---------------- mindspore: build end   ----------------"
