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
network config setting, will be used in train.py and eval.py
"""


class DataConfig:
    """
    Define parameters of dataset.
    """
    data_vocab_size = 184965
    train_num_of_parts = 21
    test_num_of_parts = 3
    batch_size = 16000
    data_field_size = 39
    # dataset format, 1: mindrecord, 2: tfrecord, 3: h5
    data_format = 1


class ModelConfig:
    """
    Define parameters of model.
    """
    batch_size = DataConfig.batch_size
    data_field_size = DataConfig.data_field_size
    data_vocab_size = DataConfig.data_vocab_size
    data_emb_dim = 80
    deep_layer_args = [[1024, 512, 256, 128], "relu"]
    init_args = [-0.01, 0.01]
    weight_bias_init = ['normal', 'normal']
    keep_prob = 0.9


class TrainConfig:
    """
    Define parameters of training.
    """
    batch_size = DataConfig.batch_size
    l2_coef = 8e-5
    learning_rate = 5e-4
    epsilon = 5e-8
    loss_scale = 1024.0
    train_epochs = 3
    save_checkpoint = True
    ckpt_file_name_prefix = "deepfm"
    save_checkpoint_steps = 1
    keep_checkpoint_max = 15
    eval_callback = True
    loss_callback = True
