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
CUR_DIR=`pwd`
SAVE_PATH=${CUR_DIR}/save_models
EXPORT_PATH=${SAVE_PATH}
python ${CUR_DIR}/export.py --device_id=0 \
        --batch_size=1  \
        --number_labels=3  \
        --ckpt_file="${SAVE_PATH}/classifier-3_302.ckpt"  \
        --file_name="${EXPORT_PATH}/emotect.mindir"  \
        --file_format="MINDIR"  \
        --device_target="Ascend"
