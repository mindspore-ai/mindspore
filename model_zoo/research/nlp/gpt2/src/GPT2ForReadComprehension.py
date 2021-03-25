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
GPT-2 downstream task (Reading Comprehension) model script.
"""
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P

from .GPT2_model import GPT2Model


class GPT2CoQAModel(nn.Cell):
    """
    This class is responsible for CoQA
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2CoQAModel, self).__init__()
        if not is_training:
            config.hidden_dropout = 0.0

        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.d_model,
                               config.vocab_size,
                               weight_init=self.weight_init,
                               has_bias=False).to_float(config.compute_type)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.vocab_size = config.vocab_size
        self.dtype = config.dtype

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                                 where 0 indicates padding position.

        Returns:
            logits (Tensor): language model distribution with log_softmax, shape with[batch_size, seq_len, d_model].
        """
        decoder_output, _ = self.gpt2(input_ids, input_mask)
        decoder_output = P.Cast()(decoder_output, self.dtype)
        batch_size, seq_length, d_model = P.Shape()(decoder_output)
        reshaped_ouput = P.Reshape()(decoder_output, (-1, d_model)) # [batch_size * seq_length, d_model]
        logits = self.dense1(reshaped_ouput)
        logits = P.Cast()(logits, self.dtype)
        logits = self.log_softmax(logits)
        logits = P.Reshape()(logits, (batch_size, seq_length, self.vocab_size))
        return logits
