#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unlesee required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================================================================

export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export OPENCV_HOME=/usr/local
export CPLUS_INCLUDE_PATH=/usr/local/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib:/usr/local/include
export MINDSPORE_HOME=/home/work/user-job-dir/psenet/mindspore
