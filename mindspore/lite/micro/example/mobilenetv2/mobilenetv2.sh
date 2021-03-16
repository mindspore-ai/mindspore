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

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MINDSPORE_ROOT_DIR=${${CURRENT_DIR}%%/mindspore/lite/micro/example/mobilenetv2}

OUTPUT_DIR=${1:-${MINDSPORE_ROOT_DIR}/output}
THREAD_NUM=${2:-32}
MODULE_NAME=mobilenetv2
OUTPUT_IR=Reshape-64.ir
CALIB_OUT=${CURRENT_DIR}/Reshape-64.out

echo "current dir is: ${CURRENT_DIR}"
echo "packed output dir is :${OUTPUT_DIR}"

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "folder ${OUTPUT_DIR} does not exist"
  return 1
fi

# rm if already exist
WORKSPACE=${CURRENT_DIR}/build
rm -rf ${WORKSPACE}
mkdir ${WORKSPACE} || exit 1
PROJECT_DIR=${WORKSPACE}/${MODULE_NAME}

compare_output() {
  local OUTPUT_FILE=$1
  local CALIB_FILE=$2
  if [[ ! -f "${OUTPUT_FILE}" || ! -f "${CALIB_FILE}" ]]; then
    echo "file ${OUTPUT_FILE}, ${CALIB_FILE} does not exist, pwd $(pwd)"
    exit 1
  fi
  lines=$(cat ${CALIB_FILE} | wc -l)
  for ((i = 1; i <= $lines; i++)); do
    line1=$(awk 'NR=="'${i}'"{print $0}' ${CALIB_FILE})
    line2=$(awk 'NR=="'${i}'"{print $0}' ${OUTPUT_FILE})
    if [[ "${line1}" != "${line2}" ]]; then
      echo -e "file ${OUTPUT_FILE}, ${CALIB_FILE}, compare failed! line: ${i}"
      exit 1
    fi
  done
  echo -e "compare success, ${OUTPUT_FILE}, ${CALIB_FILE}"
}

# cp oplib and codegen
cp ${OUTPUT_DIR}/mindspore-lite-*-codegen-linux-x64.tar.gz ${WORKSPACE}/ || exit 1
cd ${WORKSPACE} || exit 1
tar -zxf mindspore-lite-*-codegen-linux-x64.tar.gz || exit 1
cd mindspore-lite-*-codegen-linux-x64 || exit 1
mv operator_library/ ${WORKSPACE}/ || exit 1
mv codegen ${WORKSPACE}/ || exit 1
cd -
rm -r mindspore-lite-*-codegen-linux-x64 || exit 1
rm mindspore-lite-*-codegen-linux-x64.tar.gz || exit 1

# convert model
cp ${OUTPUT_DIR}/mindspore-lite-*-converter-linux-x64.tar.gz ${WORKSPACE}/ || exit 1
cd ${WORKSPACE} || exit 1
tar -zxf mindspore-lite-*-converter-linux-x64.tar.gz || exit 1
rm mindspore-lite-*-converter-linux-x64.tar.gz || exit 1
cd mindspore-lite-*-converter-linux-x64 || exit 1
export LD_LIBRARY_PATH=./lib/:./third_party/protobuf/lib:./third_party/flatbuffers/lib:./third_party/glog/lib
converter/converter_lite --fmk=TFLITE \
                         --modelFile=${CURRENT_DIR}/mobilenet_v2_1.0_224_quant.tflite \
                         --outputFile=${WORKSPACE}/mobilenet_v2
cd -
rm -rf mindspore-lite-*-converter-linux-x64 || exit 1

# generate code
${WORKSPACE}/codegen --modelPath=${WORKSPACE}/mobilenet_v2.ms \
                     --moduleName=${MODULE_NAME} \
                     --isWeightFile=true \
                     --debugMode=true
rm codegen

if [ ! -d "${PROJECT_DIR}" ]; then
  echo "folder ${PROJECT_DIR} does not exist"
  return 1
fi
cd ${PROJECT_DIR} || exit 1

# 1. build static lib.a
echo -e "building static library"
mkdir -p src/build && cd src/build || exit 1
OP_HEADER_PATH=${WORKSPACE}/operator_library/include
OP_LIB=${WORKSPACE}/operator_library/lib/x86/libops.a
echo "Head Path: ${OP_HEADER_PATH}"
echo "Lib Path: ${OP_LIB}"
cmake -DCMAKE_BUILD_TYPE=Debug       \
      -DOP_LIB=${OP_LIB}             \
      -DOP_HEADER_PATH=${OP_HEADER_PATH} ..
make -j${THREAD_NUM}

# 2. build benchmark
cd ${PROJECT_DIR}/benchmark && mkdir -p build && cd build || exit 1
cmake -DMODEL_LIB="${PROJECT_DIR}/src/build/libnet.a" ..
make -j${THREAD_NUM}

echo "net file: ${PROJECT_DIR}/src/${MODULE_NAME}.net"
# 3. run benchmark
./benchmark ${CURRENT_DIR}/input_1_224_224_3_uint8.bin ${PROJECT_DIR}/src/${MODULE_NAME}.net
compare_output ${OUTPUT_IR} ${CALIB_OUT}

RET=$?
if [[ "${RET}" -eq 0 ]]; then
  echo -e "run benchmark success: ${MODULE_NAME}"
else
  echo -e "run benchmark failed: ${MODULE_NAME}"
  exit 1
fi