# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
network config setting, will be used in main.py
"""
from easydict import EasyDict as edict


cfg = edict({
    'dataset': 'ml-1m', # Dataset to be trained and evaluated, choice: ["ml-1m", "ml-20m"]

    'data_dir': '../dataset', # The location of the input data.

    'train_epochs': 14, # The number of epochs used to train.

    'batch_size': 256, # Batch size for training and evaluation

    'eval_batch_size': 160000, # The batch size used for evaluation.

    'num_neg': 4, # The Number of negative instances to pair with a positive instance.

    'layers': [64, 32, 16], # The sizes of hidden layers for MLP

    'num_factors': 16 # The Embedding size of MF model.

    })
