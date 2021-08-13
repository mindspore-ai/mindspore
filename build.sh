#!/bin/bash
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
  echo "              [-a on|off] [-p on|off] [-i] [-R] [-D on|off] [-j[n]] [-e gpu|ascend|cpu] \\"
  echo "              [-P on|off] [-z [on|off]] [-M on|off] [-V 10.1|11.1|310|910] [-I arm64|arm32|x86_64] [-K] \\"
  echo "              [-B on|off] [-E] [-l on|off] [-n full|lite|off] [-H on|off] \\"
  echo "              [-A on|off] [-S on|off] [-k on|off] [-W sse|neon|avx|avx512|off] \\"
  echo "              [-L Tensor-RT path] [-y on|off]  \\"
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
  echo "    -e Use cpu, gpu or ascend"
  echo "    -s Enable security, default off"
  echo "    -P Enable dump anf graph to file in ProtoBuffer format, default on"
  echo "    -D Enable dumping of function graph ir, default on"
  echo "    -z Compile dataset & mindrecord, default on"
  echo "    -n Compile minddata with mindspore lite, available: off, lite, full, lite_cv, full mode in lite train and lite_cv, wrapper mode in lite predict"
  echo "    -M Enable MPI and NCCL for GPU training, gpu default on"
  echo "    -V Specify the device version, if -e gpu, default CUDA 10.1, if -e ascend, default Ascend 910"
  echo "    -I Enable compiling mindspore lite for arm64, arm32 or x86_64, default disable mindspore lite compilation"
  echo "    -A Enable compiling mindspore lite aar package, option: on/off, default: off"
  echo "    -K Compile with AKG, default on"
  echo "    -B Enable debugger, default on"
  echo "    -E Enable IBVERBS for parameter server, default off"
  echo "    -l Compile with python dependency, default on"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "    -k Enable make clean, clean up compilation generated cache "
  echo "    -W Enable SIMD instruction set, use [sse|neon|avx|avx512|off], default avx for cloud CPU backend"
  echo "    -H Enable hidden"
  echo "    -L Link and specify Tensor-RT library path, default disable Tensor-RT lib linking"
  echo "    -y Compile the symbol table switch and save the symbol table to the directory output"
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
  ENABLE_SECURITY="off"
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
  LITE_ENABLE_AAR="off"
  USE_GLOG="on"
  ENABLE_AKG="on"
  ENABLE_ACL="off"
  ENABLE_D="off"
  ENABLE_DEBUGGER="on"
  ENABLE_IBVERBS="off"
  ENABLE_PYTHON="on"
  ENABLE_GPU="off"
  ENABLE_VERBOSE="off"
  ENABLE_GITEE="off"
  ENABLE_MAKE_CLEAN="off"
  X86_64_SIMD="off"
  ARM_SIMD="off"
  DEVICE_VERSION=""
  DEVICE=""
  ENABLE_HIDDEN="on"
  TENSORRT_HOME=""
  USER_ENABLE_DUMP_IR=false
  USER_ENABLE_DEBUGGER=false
  ENABLE_SYM_FILE="off"
  # Process the options
  while getopts 'drvj:c:t:hb:s:a:g:p:ie:m:l:I:RP:D:zM:V:K:B:En:A:S:k:W:H:L:y' opt
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
      y)
        ENABLE_SYM_FILE="on"
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
      s)
        check_on_off $OPTARG s
        if [[ "X$OPTARG" == "Xon" ]]; then
          if [[ $USER_ENABLE_DUMP_IR == true ]]; then
            echo "enable security, the dump ir is not available"
            usage
            exit 1
          fi
          if [[ $USER_ENABLE_DEBUGGER == true ]]; then
            echo "enable security, the debugger is not available"
            usage
            exit 1
          fi
          ENABLE_DUMP_IR="off"
          ENABLE_DEBUGGER="off"
        fi
        ENABLE_SECURITY="$OPTARG"
        echo "enable security"
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
        if [[ "X$OPTARG" == "Xon" ]]; then
          if [[ "X$ENABLE_SECURITY" == "Xon" ]]; then
            echo "enable security, the dump ir is not available"
            usage
            exit 1
          fi
          USER_ENABLE_DUMP_IR=true
        fi
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
          LITE_PLATFORM="arm64"
        elif [[ "$OPTARG" == "arm32" ]]; then
          LITE_PLATFORM="arm32"
        elif [[ "$OPTARG" == "x86_64" ]]; then
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
        if [[ "X$OPTARG" == "Xon" ]]; then
          if [[ "X$ENABLE_SECURITY" == "Xon" ]]; then
            echo "enable security, the debugger is not available"
            usage
            exit 1
          fi
          USER_ENABLE_DEBUGGER=true
        fi
        ENABLE_DEBUGGER="$OPTARG"
        ;;
      E)
        ENABLE_IBVERBS="on"
        echo "enable IBVERBS for parameter server"
        ;;
      A)
        COMPILE_LITE="on"
        if [[ "$OPTARG" == "on" ]]; then
          LITE_ENABLE_AAR="on"
        fi
        ;;
      W)
        if [[ "$OPTARG" != "sse" && "$OPTARG" != "off" && "$OPTARG" != "avx" && "$OPTARG" != "avx512" && "$OPTARG" != "neon" ]]; then
          echo "Invalid value ${OPTARG} for option -W, -W parameter must be sse|neon|avx|avx512|off"
          usage
          exit 1
        fi
        if [[ "$OPTARG" == "sse" || "$OPTARG" == "avx" || "$OPTARG" == "avx512" ]]; then
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

  if [[ "X$RUN_TESTCASES" == "Xon" && "X$DEVICE" != "X" ]]; then
    echo "WARNING:Option -e can't be set while option -t on/ut is set, reset device to empty."
    DEVICE=""
  fi

  # Parse device
  # Process build option
  if [[ "X$DEVICE" == "Xgpu" ]]; then
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
    # building 310 package by giving specific -V 310 instruction
    if [[ "X$DEVICE_VERSION" == "X310" ]]; then
      ENABLE_ACL="on"
    # universal ascend package
    elif [[ "X$DEVICE_VERSION" == "X910" ]]; then
      ENABLE_D="on"
      ENABLE_ACL="on"
      ENABLE_CPU="on"
    else
      echo "Invalid value ${DEVICE_VERSION} for option -V"
      usage
      exit 1
    fi
  elif [[ "X$DEVICE" == "Xcpu" ]]; then
    ENABLE_CPU="on"
  elif [[ "X$DEVICE" == "X" ]]; then
    :
  else
    echo "Invalid value ${DEVICE} for option -e"
    usage
    exit 1
  fi
}

update_submodule()
{
  git submodule update --init graphengine
  cd "${BASEPATH}/graphengine"
  git submodule update --init metadef
  cd "${BASEPATH}"
  if [[ "X$ENABLE_AKG" = "Xon" ]] && [[ "X$ENABLE_D" = "Xon" || "X$ENABLE_GPU" = "Xon" ]]; then
      git submodule update --init --recursive akg
  fi
}

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
    if [[ "X$ENABLE_SYM_FILE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_SYM_FILE=ON"
    fi
    if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
    fi
    if [[ "X$ENABLE_PROFILE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PROFILE=ON"
    fi
    if [[ "X$ENABLE_SECURITY" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_SECURITY=ON"
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
      cmake ${CMAKE_ARGS} ${BASEPATH}
    fi
    if [[ -n "$VERBOSE" ]]; then
      CMAKE_VERBOSE="--verbose"
    fi
    cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
    echo "success building mindspore project!"
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}/mindspore"
  cmake --build . --target clean
}

echo "---------------- MindSpore: build start ----------------"
checkopts "$@"

if [[ "X$COMPILE_LITE" = "Xon" ]]; then
  export COMPILE_MINDDATA_LITE
  export ENABLE_VERBOSE
  export LITE_PLATFORM
  export LITE_ENABLE_AAR
  source mindspore/lite/build_lite.sh
else
  mkdir -pv "${BUILD_PATH}/package/mindspore/lib"
  update_submodule

  build_mindspore

  if [[ "X$ENABLE_MAKE_CLEAN" = "Xon" ]]; then
    make_clean
  fi
  if [[ "X$ENABLE_ACL" == "Xon" ]] && [[ "X$ENABLE_D" == "Xoff" ]]; then
      echo "acl mode, skipping deploy phase"
      rm -rf ${BASEPATH}/output/_CPack_Packages/
    else
      cp -rf ${BUILD_PATH}/package/mindspore/lib ${BASEPATH}/mindspore
      cp -rf ${BUILD_PATH}/package/mindspore/*.so ${BASEPATH}/mindspore
  fi
fi
echo "---------------- MindSpore: build end   ----------------"
