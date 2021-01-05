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
"""FastText model."""
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.initializer import XavierUniform
from mindspore.common import dtype as mstype

class FastText(nn.Cell):
    """
    FastText model
    Args:

        vocab_size: vocabulary size
        embedding_dims: The size of each embedding vector
        num_class: number of labels
    """
    def __init__(self, vocab_size, embedding_dims, num_class):
        super(FastText, self).__init__()
        self.vocab_size = vocab_size
        self.embeding_dims = embedding_dims
        self.num_class = num_class
        self.embeding_func = nn.Embedding(vocab_size=self.vocab_size,
                                          embedding_size=self.embeding_dims,
                                          padding_idx=0, embedding_table='Zeros')
        self.fc = nn.Dense(self.embeding_dims, out_channels=self.num_class,
                           weight_init=XavierUniform(1)).to_float(mstype.float16)
        self.reducesum = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.squeeze = P.Squeeze(axis=1)
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.realdiv = P.RealDiv()
        self.fill = P.Fill()
        self.log_softmax = nn.LogSoftmax(axis=1)
    def construct(self, src_tokens, src_token_length):
        """
        construct network
        Args:

            src_tokens: source sentences
            src_token_length: source sentences length

        Returns:
            Tuple[Tensor], network outputs
        """
        src_tokens = self.embeding_func(src_tokens)
        embeding = self.reducesum(src_tokens, 1)

        embeding = self.realdiv(embeding, src_token_length)

        embeding = self.cast(embeding, mstype.float16)
        classifer = self.fc(embeding)
        classifer = self.cast(classifer, mstype.float32)

        return classifer
