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

for((i=0;i<4;i++));
do
	rm -rf device$i
	mkdir device$i
	cd device$i
    mkdir output
    source ../../dist_env_4p.sh $i
    env  >log$i.log
    pytest -s ../test_add_relu_parallel_4p.py>../../log/test_add_relu_parallel_4p_log$i.log  2>&1 &
    cd ..    
done
