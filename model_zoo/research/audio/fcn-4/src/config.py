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
network config setting, will be used in train.py, eval.py
"""
from easydict import EasyDict as edict

data_cfg = edict({
    'num_classes': 50,
    'num_consumer': 4,
    'get_npy': 1,
    'get_mindrecord': 1,
    'audio_path': "/dev/data/Music_Tagger_Data/fea/",
    'npy_path': "/dev/data/Music_Tagger_Data/fea/",
    'info_path': "/dev/data/Music_Tagger_Data/fea/",
    'info_name': 'annotations_final.csv',
    'device_target': 'Ascend',
    'device_id': 0,
    'mr_path': '/dev/data/Music_Tagger_Data/fea/',
    'mr_name': ['train', 'val'],
})

music_cfg = edict({
    'pre_trained': False,
    'lr': 0.0005,
    'batch_size': 32,
    'epoch_size': 10,
    'loss_scale': 1024.0,
    'num_consumer': 4,
    'mixed_precision': False,
    'train_filename': 'train.mindrecord0',
    'val_filename': 'val.mindrecord0',
    'data_dir': '/dev/data/Music_Tagger_Data/fea/',
    'device_target': 'Ascend',
    'device_id': 0,
    'keep_checkpoint_max': 10,
    'save_step': 2000,
    'checkpoint_path': '/dev/data/Music_Tagger_Data/model',
    'prefix': 'MusicTagger',
    'model_name': 'MusicTagger_3-50_543.ckpt',
})
