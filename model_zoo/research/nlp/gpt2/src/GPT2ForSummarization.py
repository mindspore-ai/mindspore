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
GPT-2 downstream task (Summarization) model script.
"""
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import TruncatedNormal

from .GPT2_model import GPT2Model


class GPT2SummarizationModel(nn.Cell):
    """
        GPT2SummarizationModel is responsible for summary task, i.e. cnn_dailymail datasets.

        Args:
            config: the configuration of GPT-2 model
            is_training (bool): `True` for train (finetune), `False` for evaluation.
            use_one_hot_embeddings (bool): default False.
    """
    def __init__(self, config, is_training=True, use_one_hot_embeddings=False):
        super(GPT2SummarizationModel, self).__init__()
        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.lm_head = nn.Dense(config.d_model,
                                config.vocab_size,
                                weight_init=TruncatedNormal(config.initializer_range),
                                has_bias=False).to_float(mstype.float16)
        self.reshape = P.Reshape()
        self.dtype = config.dtype
        self.cast = P.Cast()
        self.shape = P.Shape()

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                                 where 0 indicates padding position.

        Returns:
            lm_logits (Tensor): language model distribution without log_softmax,
                                shape with [batch_size, seq_len, d_model].
        """
        output, _ = self.gpt2(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        batch_size, seq_length, d_model = self.shape(output)

        hidden_state = self.reshape(output, (-1, d_model))
        hidden_state = self.cast(hidden_state, self.dtype)
        lm_logits = self.lm_head(hidden_state)
        lm_logits = self.cast(lm_logits, self.dtype)
        lm_logits = self.reshape(lm_logits, (batch_size, seq_length, -1))

        return lm_logits
