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
export CUDA_PATH=""
export BUILD_PATH="${BASEPATH}/build/"

source ./scripts/build/usage.sh
source ./scripts/build/default_options.sh
source ./scripts/build/option_proc_debug.sh
source ./scripts/build/option_proc_mindspore.sh
source ./scripts/build/option_proc_lite.sh
source ./scripts/build/process_options.sh
source ./scripts/build/parse_device.sh
source ./scripts/build/build_mindspore.sh

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

update_submodule()
{
  git submodule update --init graphengine
  cd "${BASEPATH}/graphengine"
  GRAPHENGINE_SUBMODULE="910/metadef"
  if [[ "X$ASCEND_VERSION" = "X910b" ]]; then
    GRAPHENGINE_SUBMODULE="910b/metadef"
  fi
  git submodule update --init ${GRAPHENGINE_SUBMODULE}
  cd "${BASEPATH}"
  if [[ "X$ENABLE_AKG" = "Xon" ]]; then
    if [[ "X$ENABLE_D" == "Xon" ]]; then
      git submodule update --init akg
    else
      GIT_LFS_SKIP_SMUDGE=1 git submodule update --init akg
    fi
  fi
}

build_exit()
{
    echo "$@" >&2
    stty echo
    exit 1
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}/mindspore"
  cmake --build . --target clean
}

echo "---------------- MindSpore: build start ----------------"
init_default_options
process_options "$@"
parse_device

if [[ "X$COMPILE_LITE" = "Xon" ]]; then
  export COMPILE_MINDDATA_LITE
  export ENABLE_VERBOSE
  export LITE_PLATFORM
  export LITE_ENABLE_AAR
  source mindspore/lite/build_lite.sh
else
  mkdir -pv "${BUILD_PATH}/package/mindspore/lib"
  mkdir -pv "${BUILD_PATH}/package/mindspore/lib/plugin"
  update_submodule

  build_mindspore

  if [[ "X$ENABLE_MAKE_CLEAN" = "Xon" ]]; then
    make_clean
  fi
  if [[ "X$ENABLE_ACL" == "Xon" ]] && [[ "X$ENABLE_D" == "Xoff" ]]; then
      echo "acl mode, skipping deploy phase"
      rm -rf ${BASEPATH}/output/_CPack_Packages/
  elif [[ "X$FASTER_BUILD_FOR_PLUGINS" == "Xon" ]]; then
      echo "plugin mode, skipping deploy phase"
      rm -rf ${BASEPATH}/output/_CPack_Packages/
  else
      cp -rf ${BUILD_PATH}/package/mindspore/lib ${BASEPATH}/mindspore/python/mindspore
      cp -rf ${BUILD_PATH}/package/mindspore/*.so ${BASEPATH}/mindspore/python/mindspore
  fi
fi
echo "---------------- MindSpore: build end   ----------------"
