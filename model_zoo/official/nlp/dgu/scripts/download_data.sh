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

# download dataset file to ./
DATA_URL=https://paddlenlp.bj.bcebos.com/datasets/DGU_datasets.tar.gz
wget --no-check-certificate ${DATA_URL}
# unzip dataset file to ./DGU_datasets
tar -zxvf DGU_datasets.tar.gz

cd src
# download vocab file to ./src/
VOCAB_URL=https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
wget --no-check-certificate ${VOCAB_URL}
