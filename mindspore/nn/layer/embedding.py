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
"""embedding"""
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from ..cell import Cell
from ..._checkparam import Validator as validator

__all__ = ['Embedding']

class Embedding(Cell):
    r"""
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output is
    the corresponding word embeddings.

    Note:
        When 'use_one_hot' is set to True, the input should be of type mindspore.int32.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot (bool): Specifies whether to apply one_hot encoding form. Default: False.
        embedding_table (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        dtype (:class:`mindspore.dtype`): Data type of input. Default: mindspore.float32.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(\text{vocab_size})`.

    Outputs:
        Tensor of shape :math:`(\text{vocab_size}, \text{embedding_size})`.

    Examples:
        >>> net = nn.Embedding(20000, 768,  True)
        >>> input_data = Tensor(np.ones([8, 128]), mindspore.int32)
        >>>
        >>> # Maps the input word IDs to word embedding.
        >>> output = net(input_data)
        >>> output.shape()
        (8, 128, 768)
    """
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mstype.float32):
        super(Embedding, self).__init__()
        validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot = use_one_hot
        self.embedding_table = Parameter(initializer(embedding_table, [vocab_size, embedding_size]),
                                         name='embedding_table')
        self.dtype = dtype
        self.expand = P.ExpandDims()
        self.reshape_flat = P.Reshape()
        self.shp_flat = (-1,)
        self.gather = P.GatherV2()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.get_shp = P.Shape()

    def construct(self, ids):
        extended_ids = self.expand(ids, -1)
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(extended_ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        s = 'vocab_size={}, embedding_size={},' \
            'use_one_hot={}, ' \
            'embedding_table={}, dtype={}'.format(
                self.vocab_size,
                self.embedding_size,
                self.use_one_hot,
                self.embedding_table,
                self.dtype)
        return s
