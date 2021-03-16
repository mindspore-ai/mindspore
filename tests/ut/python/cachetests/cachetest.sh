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

# This script is the driver of the individual test scenarios
CURRPATH=$(cd "$(dirname $0)"; pwd)

echo "----------------------------------------------"
echo "Invalid syntax and cache_admin failure testing"
echo "----------------------------------------------"
echo
${CURRPATH}/cachetest_args.sh
num_failures=$?
echo
echo "Invalid syntax and cache_admin failure testing complete.  Number of failures: $num_failures"
echo

echo "----------------------------------------------"
echo "Test pipelines with cache (python)"
echo "----------------------------------------------"
echo
${CURRPATH}/cachetest_py.sh
num_failures=$?
echo
echo "Test pipelines with cache complete.  Number of failures: $num_failures"
echo

echo "----------------------------------------------"
echo "Cache cpp tests"
echo "----------------------------------------------"
echo
${CURRPATH}/cachetest_cpp.sh
num_failures=$?
echo
echo "Cache cpp tests complete.  Number of failures: $num_failures"
echo
