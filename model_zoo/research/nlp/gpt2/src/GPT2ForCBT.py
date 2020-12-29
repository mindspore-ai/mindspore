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
GPT-2 downstream task (CBT) model script.
"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

from .GPT2_model import GPT2Model


class GPT2CBTModel(nn.Cell):
    """
    GPT2CBTModel is responsible for Children's Book Test (CBT) task, i.e. CBT-CN, CBT-NE datasets.
    """
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        """
        Args:
            config: the configuration of GPT-2 model
            is_training (bool): `True` for train (finetune), `False` for evaluation.
            use_one_hot_embeddings (bool): default False.
        """
        super(GPT2CBTModel, self).__init__()
        if not is_training:
            config.summary_first_dropout = 0.0

        self.is_training = is_training
        self.d_model = config.d_model
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.log_softmax = P.LogSoftmax(axis=-1)

        self.dtype = config.dtype
        self.lm_head = nn.Dense(config.d_model,
                                config.vocab_size,
                                weight_init=TruncatedNormal(config.initializer_range),
                                has_bias=False).to_float(config.compute_type)

        self.first_dropout = nn.Dropout(1 - config.summary_first_dropout)

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): shape with [batch_size, seq_len]
            input_mask (Tensor): shape with [batch_size, seq_len] 0 indicates padding mask

        Returns:
            lm_logits (Tensor): language model distribution with log_softmax,
                                shape with [batch_size, seq_len, vocab_size]

        """
        output, _ = self.gpt2(input_ids, input_mask)  # output shape is [batch_size, seq_len, d_model]
        output = self.cast(output, self.dtype)
        output = self.reshape(output, (-1, self.d_model))
        output = self.first_dropout(output)
        lm_logits = self.lm_head(output)  # [batch_size * seq_len, vocab_size]
        lm_logits = self.reshape(lm_logits, (self.batch_size, self.seq_length, self.vocab_size))
        lm_logits = self.cast(lm_logits, self.dtype)
        lm_logits = self.log_softmax(lm_logits)

        return lm_logits

    def get_lm_head(self):
        return self.lm_head.weight
