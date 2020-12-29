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
"""get config setting"""


def get_train_setting(finetune_config):
    """get train config setting"""
    cfg = finetune_config

    print("Loading GPT2 Finetune Config setting......")
    print(" | optimizer: {}".format(cfg.optimizer))
    opt = cfg['optimizer']
    print(" | learning rate: {}".format(cfg[opt]['learning_rate']))
    print(" | end learning rate: {}".format(
        cfg[opt]['end_learning_rate'] if 'end_learning_rate' in cfg[opt] else 'None'))
    print(" | weight decay: {}\n".format(cfg[opt]['weight_decay'] if 'weight_decay' in cfg[opt] else 'None'))


def get_model_setting(finetune_config, model_config):
    """get GPT-2 model config setting"""
    cfg = finetune_config
    gpt2_net_cfg = model_config

    print("Loading GPT2 Model Config setting......")
    print(" | model size: {}".format(cfg.gpt2_network))
    print(" | batch_size: {}".format(gpt2_net_cfg.batch_size))
    print(" | seq_length: {}".format(gpt2_net_cfg.seq_length))
    print(" | vocab_size: {}".format(gpt2_net_cfg.vocab_size))
    print(" | d_model: {}".format(gpt2_net_cfg.d_model))
    print(" | num_hidden_layers: {}".format(gpt2_net_cfg.num_hidden_layers))
    print(" | num_attention_heads: {}".format(gpt2_net_cfg.num_attention_heads))
    print(" | hidden_dropout: {}".format(gpt2_net_cfg.hidden_dropout))
    print(" | attention_dropout: {}".format(gpt2_net_cfg.attention_dropout))
    print(" | summary_first_dropout: {}\n".format(gpt2_net_cfg.summary_first_dropout))
