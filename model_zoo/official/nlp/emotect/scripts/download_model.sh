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
mkdir -p pretrain_models
cd pretrain_models

# download pretrain model file to ./pretrain_models/
MODEL_ERNIE=https://baidu-nlp.bj.bcebos.com/emotion_detection_ernie_finetune-1.0.0.tar.gz
wget --no-check-certificate ${MODEL_ERNIE}

# tar xvf emotion_detection_textcnn-1.0.0.tar.gz
tar xvf emotion_detection_ernie_finetune-1.0.0.tar.gz

# /bin/rm emotion_detection_textcnn-1.0.0.tar.gz
/bin/rm emotion_detection_ernie_finetune-1.0.0.tar.gz