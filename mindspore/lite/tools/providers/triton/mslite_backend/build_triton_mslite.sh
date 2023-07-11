#!/bin/bash

function Run_Build() {
  # decompress release_pkg
  cd ${open_source_ms_path}/output/ || exit 1
  if [[ ${platform} = "arm64" ]]; then
    platform="aarch64"
  elif [[ ${platform} = "x86_64" ]]; then
    platform="x64"
  fi
  file_name=$(ls ./*linux-${platform}.tar.gz)
  IFS="-" read -r -a file_name_array <<< "$file_name"
  version=${file_name_array[2]}
  tar -xf mindspore-lite-${version}-linux-${platform}.tar.gz

  export MINDSPORE_LITE_PKG_ROOT_PATH=${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}
  # compile triton mslite backend
  cd ${open_source_ms_path}/mindspore/lite/tools/providers/triton_mslite_backend
  rm -rf build; mkdir build; cd build;
  cmake -DCMAKE_INSTALL_PREFIX:PATH=../install .. || exit 1
  make install -j ${thread_num} || exit 1
  cd -
  echo "build for triton backend success"

  # cp to release package folder
  mkdir ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}/providers/
  mkdir ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}/providers/triton_mslite_backend/
  cp -r install/backends/mslite ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}/providers/triton_mslite_backend/ || exit 1
  echo "cp triton backend so to release pkg success"

  cd ${open_source_ms_path}/output
  rm ./mindspore-lite-${version}-linux-${platform}.tar.gz
  tar -zcf ./mindspore-lite-${version}-linux-x64.tar.gz ./mindspore-lite-${version}-linux-${platform}/ || exit 1
  sha256sum ./mindspore-lite-${version}-linux-x64.tar.gz > ./mindspore-lite-${version}-linux-${platform}.tar.gz.sha256 || exit 1
  rm -rf ./mindspore-lite-${version}-linux-${platform}
  echo "package ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}.tar.gz updated."
  exit
}

# bashpath should be /home/jenkins/agent-working-dir/workspace/Compile_Lite_xxx/
basepath=$(pwd)
echo "basepath is ${basepath}"
#set -e
open_source_ms_path=${basepath}/mindspore

# Example:sh build_triton_mslite.sh -I arm64 -jn
while getopts "I:j:" opt; do
    case ${opt} in
        I)
            platform=${OPTARG}
            echo "platform is ${OPTARG}"
            ;;
        j)
            thread_num=${OPTARG}
            echo "thread_num is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

echo "start building for triton mslite backend..."
Run_Build
Run_build_PID=$!
exit ${Run_build_PID}
