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

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-v] [-c on|off] [-t ut|st] [-g on|off] [-h] [-b ge] [-m infer|train] \\"
  echo "              [-a on|off] [-p on|off] [-i] [-R] [-D on|off] [-j[n]] [-e gpu|ascend|cpu] \\"
  echo "              [-P on|off] [-z [on|off]] [-M on|off] [-V 10.1|11.1|310|910|910b] [-I arm64|arm32|x86_64] [-K on|off] \\"
  echo "              [-B on|off] [-E] [-l on|off] [-n full|lite|off] [-H on|off] \\"
  echo "              [-A on|off] [-S on|off] [-k on|off] [-W sse|neon|avx|avx512|off] \\"
  echo "              [-L Tensor-RT path] [-y on|off] [-F on|off] [-G common|auto]\\"
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
  echo "    -F Use fast hash table in mindspore compiler, default on"
  echo "    -G Select an architecture to build, set 'common' to build with common architectures(eg. gpu: 5.3, 6.0, 6.2, 7.0, 7.2, 7.5),\\"
  echo "       set auto to detect automatically, default: 'auto'. Only effective for GPU currently."
  echo "    -f Faster build process for device plugins, only build plugin."
}
