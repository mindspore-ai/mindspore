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
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.ops.primitive import constexpr
from .basic import ClipByNorm
from ..cell import Cell

__all__ = ['Embedding', 'EmbeddingLookup']


class Embedding(Cell):
    r"""
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output is
    the corresponding word embeddings.

    Note:
        When 'use_one_hot' is set to True, the type of the input must be mindspore.int32.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot (bool): Specifies whether to apply one_hot encoding form. Default: False.
        embedding_table (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        dtype (:class:`mindspore.dtype`): Data type of input. Default: mindspore.float32.
        padding_idx (int, None): When the padding_idx encounters index, the output embedding vector of this index
                                 will be initialized to zero. Default: None. The feature is inactivated.
    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(\text{batch_size}, \text{input_length})`. The elements of
          the Tensor must be integer and not larger than vocab_size. Otherwise the corresponding embedding vector will
          be zero.

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

    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal',
                 dtype=mstype.float32, padding_idx=None):
        super(Embedding, self).__init__()
        self.vocab_size = validator.check_value_type('vocab_size', vocab_size, [int], self.cls_name)
        self.embedding_size = validator.check_value_type('embedding_size', embedding_size, [int], self.cls_name)
        validator.check_value_type('use_one_hot', use_one_hot, [bool], self.cls_name)
        validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.init_tensor = initializer(embedding_table, [vocab_size, embedding_size])
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = validator.check_int_range(padding_idx, 0, vocab_size, Rel.INC_BOTH,
                                                         "padding_idx", self.cls_name)
            self.init_tensor = self.init_tensor.to_tensor().asnumpy()
            self.init_tensor[self.padding_idx] = 0
        self.embedding_table = Parameter(self.init_tensor, name='embedding_table')
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
        s = 'vocab_size={}, embedding_size={}, use_one_hot={}, embedding_table={}, dtype={}, padding_idx={}'.format(
            self.vocab_size, self.embedding_size, self.use_one_hot, self.embedding_table, self.dtype, self.padding_idx)
        return s


@constexpr
def _make_axis_range(start, end):
    axis = tuple(range(start, end))
    return axis


class EmbeddingLookup(Cell):
    r"""
    Returns a slice of input tensor based on the specified indices.

    Note:
        When 'target' is set to 'CPU', this module will use
        P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU') which
        specified 'offset = 0' to lookup table.
        When 'target' is set to 'DEVICE', this module will use P.GatherV2() which
        specified 'axis = 0' to lookup table.
        In field slice mode, the manual_shapes must be given. It is a tuple ,where
        the element is vocab[i], vocab[i] is the row numbers for i-th part.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        param_init (str): The initialize way of embedding table. Default: 'normal'.
        target (str): Specifies the target where the op is executed. The value must in
            ['DEVICE', 'CPU']. Default: 'CPU'.
        slice_mode (str): The slicing way in semi_auto_parallel/auto_parallel. The value must get through
            nn.EmbeddingLookup. Default: nn.EmbeddingLookup.BATCH_SLICE.
        manual_shapes (tuple): The accompaniment array in field slice mode.
        max_norm (Union[float, None]): A maximum clipping value. The data type must be float16, float32
                                       or None. Default: None
        sparse (bool): Using sparse mode. When 'target' is set to 'CPU', 'sparse' has to be true. Default: True.
    Inputs:
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of embedding_table,
          and the exceeding part will be filled with 0 in the output. Input_indices must only be a 2d tensor in
          this interface.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Examples:
        >>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
        >>> out = nn.EmbeddingLookup(4,2)(input_indices)
        >>> output.shape
        (2, 2, 2)
    """
    BATCH_SLICE = "batch_slice"
    FIELD_SLICE = "field_slice"
    TABLE_ROW_SLICE = "table_row_slice"
    TABLE_COLUMN_SLICE = "table_column_slice"

    def __init__(self, vocab_size, embedding_size, param_init='normal',
                 target='CPU', slice_mode='batch_slice', manual_shapes=None,
                 max_norm=None, sparse=True):
        super(EmbeddingLookup, self).__init__()
        self.target = target
        if target not in ('CPU', 'DEVICE'):
            raise ValueError('Attr \'target\' of \'EmbeddingLookup\' Op passed '
                             + str(target) + ', should be one of values in \'CPU\', \'DEVICE\'.')
        if not sparse and target == 'CPU':
            raise ValueError('When target is CPU, embedding_lookup must be sparse.')
        if sparse:
            self.gatherv2 = P.SparseGatherV2()
        else:
            self.gatherv2 = P.GatherV2()
        self.embeddinglookup = P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU')
        self.vocab_size = validator.check_value_type('vocab_size', vocab_size, [int], self.cls_name)
        self.embedding_size = validator.check_value_type('embedding_size', embedding_size, [int], self.cls_name)
        self.embedding_table = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                         name='embedding_table')
        parallel_mode = _get_parallel_mode()
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if slice_mode == "field_slice" and is_auto_parallel:
            if not manual_shapes:
                raise ValueError("in slice field mode, the manual_shapes should not be none")
            if not isinstance(manual_shapes, tuple):
                raise TypeError("manual_shapes type must be tuple(int) cannot be {}!".format(type(manual_shapes)))
            for dim in manual_shapes:
                validator.check_positive_int(dim, 'manual shape dim', self.cls_name)
            self.gatherv2.add_prim_attr("manual_split", manual_shapes)
            self.embeddinglookup.add_prim_attr("manual_split", manual_shapes)
            self.gatherv2.shard(((get_group_size(), 1), (1, get_group_size())))
            self.embeddinglookup.shard(((get_group_size(), 1), (1, get_group_size())))
        elif slice_mode == "table_row_slice" and is_auto_parallel:
            self.gatherv2.shard(((get_group_size(), 1), (1, 1)))
            self.embeddinglookup.shard(((get_group_size(), 1), (1, 1)))
        elif slice_mode == "table_column_slice" and is_auto_parallel:
            self.gatherv2.shard(((1, get_group_size()), (1, 1)))
            self.embeddinglookup.shard(((1, get_group_size()), (1, 1)))
        elif slice_mode == "batch_slice" and is_auto_parallel:
            self.gatherv2.shard(((1, 1), (get_group_size(), 1)))
            self.embeddinglookup.shard(((1, 1), (get_group_size(), 1)))
        else:
            if is_auto_parallel:
                raise ValueError("slice_mode should support mode in nn.EmbeddingLookup, but get "
                                 + str(slice_mode))
        self.max_norm = max_norm
        if self.max_norm is not None:
            self.max_norm = validator.check_positive_float(self.max_norm, 'max_norm', self.cls_name)
            self.max_norm = Tensor(self.max_norm, dtype=mstype.float32)

    def construct(self, indices):
        if self.target == "CPU":
            out = self.embeddinglookup(self.embedding_table, indices, 0)
        else:
            out = self.gatherv2(self.embedding_table, indices, 0)
        if self.max_norm is not None:
            axis = _make_axis_range(F.rank(indices), F.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)
        return out
