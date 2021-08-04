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
CUR_PATH=`pwd`
DATA_PATH=${CUR_PATH}/../data
TRAIN_URL=http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz
VALID_URL=http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz
TEST_URL=http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/mmt16_task1_test.tar.gz

mkdir ${DATA_PATH}
cd ${DATA_PATH}
wget --no-check-certificate ${TRAIN_URL} 
wget --no-check-certificate ${VALID_URL}
wget --no-check-certificate ${TEST_URL}
tar xvf training.tar.gz
tar xvf validation.tar.gz
tar xvf mmt16_task1_test.tar.gz
/bin/rm training.tar.gz
/bin/rm validation.tar.gz
/bin/rm mmt16_task1_test.tar.gz

