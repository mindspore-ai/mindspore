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
        - **input** (Tensor) - Tensor of shape :math:`(\text{batch_size}, \text{input_length})`. The element of
          the Tensor should be integer and not larger than vocab_size. else the corresponding embedding vector is zero
          if larger than vocab_size.

    Outputs:
        Tensor of shape :math:`(\text{batch_size}, \text{input_length}, \text{embedding_size})`.

    Examples:
        >>> net = nn.Embedding(20000, 768,  True)
        >>> input_data = Tensor(np.ones([8, 128]), mindspore.int32)
        >>>
        >>> # Maps the input word IDs to word embedding.
        >>> output = net(input_data)
        >>> output.shape
        (8, 128, 768)
    """
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mstype.float32):
        super(Embedding, self).__init__()
        validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        validator.check_value_type('use_one_hot', use_one_hot, [bool], self.cls_name)
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

class EmbeddingLookup(Cell):
    r"""
    Returns a slice of input tensor based on the specified indices.

    Note:
        When 'target' is set to 'CPU', this module will use
        P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU') which
        specified 'offset = 0' to lookup table.
        when 'target' is set to 'DEVICE', this module will use P.GatherV2() which
        specified 'axis = 0' to lookup table.

    Args:
        target (str): Specify the target where the op is executed. Default: 'CPU'.

    Inputs:
        - **input_params** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
          The Tensor slice, instead of the entire Tensor.
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of `input_params`,
          and the exceeding part will be filled with 0 in the output.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_params = Tensor(np.array([[8, 9], [10, 11], [12, 13], [14, 15]]), mindspore.float32)
        >>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
        >>> out = nn.EmbeddingLookup()(input_params, input_indices)
        [[[10, 11], [8 ,9]], [[14, 15], [12, 13]]]
    """
    def __init__(self, target='CPU'):
        super(EmbeddingLookup, self).__init__()
        self.target = target
        if target not in ('CPU', 'DEVICE'):
            raise ValueError('Attr \'target\' of \'EmbeddingLookup\' Op passed '
                             + str(target) + ', should be one of values in \'CPU\', \'DEVICE\'.')
        self.gatherv2 = P.GatherV2()
        self.embeddinglookup = P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU')

    def construct(self, params, indices):
        if self.target == "CPU":
            out = self.embeddinglookup(params, ids, 0)
        else:
            out = self.gatherv2(param, ids, 0)
        return out
