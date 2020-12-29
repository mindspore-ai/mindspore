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
GPT-2 downstream task (Language Modeling) model script.
"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

from .GPT2_model import GPT2Model


class GPT2LanguageModel(nn.Cell):
    """
    GPT2LanguageModel is responsible for Language Modeling task, i.e. WikiText2, WikiText103, PTB, 1BW datasets.
    """
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        """
        Args:
            config: the configuration of GPT-2 model
            is_training (bool): `True` for train (finetune), `False` for evaluation.
            use_one_hot_embeddings (bool): default False.
        """
        super(GPT2LanguageModel, self).__init__()
        if not is_training:
            config.hidden_dropout = 0.0

        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.dtype = config.dtype
        self.dense1 = nn.Dense(config.d_model,
                               config.vocab_size,
                               weight_init=TruncatedNormal(config.initializer_range),
                               has_bias=False).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - config.hidden_dropout)
        self.log_softmax = P.LogSoftmax(axis=-1)

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                                 where 0 indicates padding position.

        Returns:
            lm_logits (Tensor): language model distribution with log_softmax, shape with[batch_size, seq_len, d_model].
        """
        output, _ = self.gpt2(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        batch_size, seq_length, d_model = self.shape(output)
        output_reshape = P.Reshape()(output, (-1, d_model)) # [batch_size * seq_len, d_model]
        logits = self.dense1(output_reshape)
        logits = self.cast(logits, self.dtype)
        logits = self.log_softmax(logits)
        lm_logits = P.Reshape()(logits, (batch_size, seq_length, self.vocab_size)) # [batch_size, seq_len, vocab]

        return lm_logits
