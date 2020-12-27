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
network config
"""
from easydict import EasyDict as edict

# LSTM CONFIG
textrcnn_cfg = edict({
    'pos_dir': 'data/rt-polaritydata/rt-polarity.pos',
    'neg_dir': 'data/rt-polaritydata/rt-polarity.neg',
    'num_epochs': 10,
    'lstm_num_epochs': 15,
    'batch_size': 64,
    'cell': 'gru',
    'ckpt_folder_path': './ckpt',
    'preprocess_path': './preprocess',
    'preprocess': 'false',
    'data_path': './data/',
    'lr': 1e-3,
    'lstm_lr_init': 2e-3,
    'lstm_lr_end': 5e-4,
    'lstm_lr_max': 3e-3,
    'lstm_lr_warm_up_epochs': 2,
    'lstm_lr_adjust_epochs': 9,
    'emb_path': './word2vec',
    'embed_size': 300,
    'save_checkpoint_steps': 149,
    'keep_checkpoint_max': 10,
})
