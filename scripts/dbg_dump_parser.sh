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

set -x 
set -e

export SAVE_GRAPHS=YES

# print usage message
function usage()
{
  echo "Usage:"
  echo "bash $0 [-g] [-d] [-a] [-h] [-f file]"
  echo "e.g. $0 -f 3_specialize.dat"
  echo ""
  echo "Options:"
  echo "    -g Generate ir file for debug"
  echo "    -d Debug dumped ir"
  echo "    -a Execute all steps, default"
  echo "    -f File to be parse"
  echo "    -h Print usage"
}

# check and set options
function checkopts()
{
  # init variable
  MODE_GEN=0
  MODE_DBG=1
  MODE_ALL=2
  FILE_NAME="3_optimize.dat"
  mode="${MODE_ALL}"    # default execute all steps

  # Process the options
  while getopts 'gdaf:h' opt
  do
    case "${opt}" in
      g)
        mode="${MODE_GEN}"
        ;;
      d)
        mode="${MODE_DBG}"
        ;;
      a)
        mode="${MODE_ALL}"
        ;;
      f)
        FILE_NAME="$OPTARG"
        if ! [ -f "${FILE_NAME}" ]; then
          echo "File $FILE_NAME does not exist"
          usage
          exit 1
        fi
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

# init variable
# check options
checkopts "$@"

CUR_PATH=$(pwd)
cd "`dirname $0`/.."

cd build/mindspore/
make -j8
cp -v mindspore/ccsrc/_c_expression.cpython-*.so ../../mindspore/
cd -

UT_NAME="./tests/ut/python/model/test_lenet.py::test_lenet5_train_sens"
#UT_NAME="./tests/python/ops/test_math_ops.py::test_matmul_grad"
#UT_NAME="./tests/python/exec/resnet_example.py::test_compile"
#UT_NAME="./tests/perf_test/test_bert_train.py::test_bert_train"

if [[ "${mode}" == "${MODE_GEN}" || "${mode}" == "${MODE_ALL}" ]]; then
  rm -rf pkl_objs
  mkdir -p pkl_objs

  echo "MS_IR_PATH=$(pwd)/pkl_objs pytest -s ${UT_NAME}"
  MS_IR_PATH=$(pwd)/pkl_objs/ pytest -s "${UT_NAME}"
  #pytest -s $UT_NAME

  # 1_resolve.dat
  # 3_specialize.dat
  # 4_simplify_data_structures.dat
  # 5_opt.dat
  # 6_opt2.dat
  # 7_opt_ge_adaptor_special.dat
  # 8_cconv.dat
  # 9_validate.dat
  cp "${FILE_NAME}" anf_ir_file.dbg

  rm -rf pkl_objs.dbg
  cp -rf pkl_objs pkl_objs.dbg
fi

if [[ "${mode}" == "${MODE_DBG}" || "${mode}" == "${MODE_ALL}" ]]; then
  echo "MS_IR_FILE=$(pwd)/anf_ir_file.dbg MS_IR_PATH=$(pwd)/pkl_objs.dbg/ pytest -s ${UT_NAME}"
  MS_IR_FILE=$(pwd)/anf_ir_file.dbg MS_IR_PATH=$(pwd)/pkl_objs.dbg/ pytest -s "${UT_NAME}"
fi

cd $CUR_PATH
