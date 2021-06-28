# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0#
#
# Unless required by applicable law or agreed to in writing software
# distributed under the License is distributed on an "AS IS" BASIS
# WITHOUT WARRANT IES OR CONITTONS OF ANY KIND， either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ====================================================================================
"""Configuration class for GNMT."""
import json

import mindspore.common.dtype as mstype


class Config:
    """
    AttentionLSTM config
    """

    def __init__(self,
                 rseed=4373337,
                 epoch=25,
                 batch_size=1,
                 dataset_sink_mode=True,
                 vocab_size=5177,
                 dim_hidden=300,
                 dim_word=300,
                 dim_aspect=100,
                 optimizer='Momentum',
                 dropout_prob=0.6,
                 lr=0.0125,
                 momentum=0.91,
                 weight_decay=0.001,
                 aspect_num=5,
                 grained=3,
                 save_graphs=False,
                 dtype=mstype.float32):

        self.save_graphs = save_graphs
        self.rseed = rseed

        self.epoch = epoch
        self.dataset_sink_mode = dataset_sink_mode

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dim_hidden = dim_hidden
        self.dropout_prob = dropout_prob

        self.compute_type = mstype.float16
        self.dtype = dtype

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dim_word = dim_word
        self.dim_aspect = dim_aspect
        self.aspect_num = aspect_num
        self.grained = grained

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
        _params = {}
        for key in json_object:
            _params[key] = json_object[key]
        return cls(**_params)

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `TransformerConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            return cls.from_dict(json.load(reader))
