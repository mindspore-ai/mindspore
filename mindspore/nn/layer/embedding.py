# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.communication.management import get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _get_full_batch
from mindspore.parallel._ps_context import _is_role_worker, _get_ps_context
from mindspore.parallel._ps_context import _insert_hash_table_size, _set_cache_enable, _set_rank_id
from mindspore import context
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.ops.primitive import constexpr
from .basic import ClipByNorm
from .math import Range
from ..cell import Cell

__all__ = ['Embedding', 'EmbeddingLookup', 'MultiFieldEmbeddingLookup']

@constexpr
def _check_input_2d(input_shape, param_name, func_name):
    if len(input_shape) != 2:
        raise ValueError(f"{func_name} {param_name} should be 2d, but got shape {input_shape}")
    return True

@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


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

    Raises:
        TypeError: If `vocab_size` or `embedding_size` is not an int.
        TypeError: If `use_one_hot` is not a bool.
        ValueError: If `padding_idx` is an int which not in range [0, `vocab_size`].

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.Embedding(20000, 768,  True)
        >>> input_data = Tensor(np.ones([8, 128]), mindspore.int32)
        >>>
        >>> # Maps the input word IDs to word embedding.
        >>> output = net(input_data)
        >>> result = output.shape
        >>> print(result)
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
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor)
        self.embedding_table = Parameter(self.init_tensor, name='embedding_table')
        self.expand = P.ExpandDims()
        self.reshape_flat = P.Reshape()
        self.shp_flat = (-1,)
        self.gather = P.Gather()
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
    Returns a slice of the input tensor based on the specified indices.

    Note:
        When 'target' is set to 'CPU', this module will use
        P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU') which
        specified 'offset = 0' to lookup table.
        When 'target' is set to 'DEVICE', this module will use P.Gather() which
        specified 'axis = 0' to lookup table.
        In field slice mode, the manual_shapes must be given. It is a tuple ,where
        the element is vocab[i], vocab[i] is the row numbers for i-th part.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        target (str): Specifies the target where the op is executed. The value must in
            ['DEVICE', 'CPU']. Default: 'CPU'.
        slice_mode (str): The slicing way in semi_auto_parallel/auto_parallel. The value must get through
            nn.EmbeddingLookup. Default: nn.EmbeddingLookup.BATCH_SLICE.
        manual_shapes (tuple): The accompaniment array in field slice mode.
        max_norm (Union[float, None]): A maximum clipping value. The data type must be float16, float32
                                       or None. Default: None
        sparse (bool): Using sparse mode. When 'target' is set to 'CPU', 'sparse' has to be true. Default: True.
        vocab_cache_size (int): Cache size of the dictionary of embeddings. Default: 0. It is valid only in
            'DEVICE' target. And the moment parameter of corresponding optimizer will also be set to the cache size.
            In addition, it should be noted that it will cost the 'DEVICE'
            memory, so suggests setting a reasonable value to avoid insufficient memory.

    Inputs:
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of embedding_table,
          and the exceeding part will be filled with 0 in the output. Values does not support negative and the result
          is undefined if values are negative. Input_indices must only be a 2d tensor in
          this interface when run in semi auto parallel/auto parallel mode.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Raises:
        TypeError: If `vocab_size` or `embedding_size` or `vocab_cache_size` is not an int.
        TypeError: If `sparse` is not a bool or `manual_shapes` is not a tuple.
        ValueError: If `vocab_size` or `embedding_size` is less than 1.
        ValueError: If `vocab_cache_size` is less than 0.
        ValueError: If `target` is neither 'CPU' nor 'DEVICE'.
        ValueError: If `slice_mode` is not one of 'batch_slice' or 'field_slice' or
                    'table_row_slice' or 'table_column_slice'.
        ValueError: If `sparse` is False and `target` is 'CPU'.
        ValueError: If `slice_mode` is 'field_slice' and `manual_shapes` is None.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
        >>> result = nn.EmbeddingLookup(4,2)(input_indices)
        >>> print(result.shape)
        (2, 2, 2)
    """
    BATCH_SLICE = "batch_slice"
    FIELD_SLICE = "field_slice"
    TABLE_ROW_SLICE = "table_row_slice"
    TABLE_COLUMN_SLICE = "table_column_slice"

    def __init__(self, vocab_size, embedding_size, param_init='normal',
                 target='CPU', slice_mode='batch_slice', manual_shapes=None,
                 max_norm=None, sparse=True, vocab_cache_size=0):
        super(EmbeddingLookup, self).__init__()
        validator.check_value_type('sparse', sparse, [bool], self.cls_name)
        self.vocab_size = validator.check_positive_int(vocab_size, 'vocab_size')
        self.vocab_cache_size = validator.check_non_negative_int(vocab_cache_size, 'vocab_cache_size')
        self.target = target
        self.sparse = sparse
        self.cache_enable = self.vocab_cache_size > 0
        self.forward_unique = False
        if target not in ('CPU', 'DEVICE'):
            raise ValueError('Attr \'target\' of \'EmbeddingLookup\' Op passed '
                             + str(target) + ', should be one of values in \'CPU\', \'DEVICE\'.')
        if not sparse and target == 'CPU':
            raise ValueError('When target is CPU, embedding_lookup must be sparse.')
        if sparse:
            self.gatherv2 = P.SparseGatherV2()
        else:
            self.gatherv2 = P.Gather()
        self.embeddinglookup = P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU')
        enable_ps = _get_ps_context("enable_ps")
        if enable_ps:
            self._process_vocab_cache(slice_mode)
        self.embedding_size = validator.check_positive_int(embedding_size, 'embedding_size')
        self.embedding_table = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size]),
                                         name='embedding_table')
        parallel_mode = _get_parallel_mode()
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.gather_revert = P.Gather()
        self.reshape_first = P.Reshape()
        self.reshape = P.Reshape()
        self.unique = P.Unique()
        self.shape = P.Shape()
        if is_auto_parallel:
            self.unique = P.Unique().shard(((1,),))
        if self.cache_enable and enable_ps:
            self._set_voacb_cache_enable_for_ps(vocab_cache_size, embedding_size, vocab_size)
            if is_auto_parallel:
                self.unique.add_prim_attr('cache_enable', True)
        indices_shape_size = 2
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
            full_batch = _get_full_batch()
            if (target == 'DEVICE' and not full_batch) or (self.cache_enable and enable_ps and sparse):
                indices_shape_size = 1
                self.gather_revert.shard(((1, 1), (get_group_size(),)))
                self.forward_unique = True
            indices_strategy = (1,)*indices_shape_size
            self.gatherv2.shard(((get_group_size(), 1), indices_strategy))
            self.embeddinglookup.shard(((get_group_size(), 1), indices_strategy))
        elif slice_mode == "table_column_slice" and is_auto_parallel:
            if target == 'DEVICE':
                indices_shape_size = 1
                self.gather_revert.shard(((1, get_group_size()), (1,)))
                self.forward_unique = True
            indices_strategy = (1,)*indices_shape_size
            self.gatherv2.shard(((1, get_group_size()), indices_strategy))
            self.embeddinglookup.shard(((1, get_group_size()), indices_strategy))
        elif slice_mode == "batch_slice" and is_auto_parallel:
            indices_strategy = [get_group_size()]
            indices_strategy.extend([1]*(indices_shape_size - 1))
            indices_strategy = tuple(indices_strategy)
            self.gatherv2.shard(((1, 1), indices_strategy))
            self.embeddinglookup.shard(((1, 1), indices_strategy))
        else:
            if is_auto_parallel:
                raise ValueError("slice_mode should support mode in nn.EmbeddingLookup, but get "
                                 + str(slice_mode))
        if self.cache_enable and not enable_ps:
            if parallel_mode != ParallelMode.STAND_ALONE:
                raise ValueError("parallel mode haven't supported cache enable yet.")
            self._set_cache_enable()
        self.embedding_table.unique = self.forward_unique
        self.max_norm = max_norm
        if self.max_norm is not None:
            self.max_norm = validator.check_positive_float(self.max_norm, 'max_norm', self.cls_name)
            self.max_norm = Tensor(self.max_norm, dtype=mstype.float32)

    def _set_cache_enable(self):
        """EmbeddingLookup cache check for not ps env, which is only support 'ascend'."""
        if self.target != 'DEVICE':
            raise ValueError("The configuration of 'vocab_cache_size' is valid only in 'DEVICE' target.")
        if not self.sparse:
            raise ValueError("The configuration of 'vocab_cache_size' is valid only 'sparse' is true.")
        if context.get_context("device_target") != 'Ascend':
            raise ValueError("The configuration of 'vocab_cache_size' is valid only in 'ascend'.")

        logger.info("EmbeddingLookup cache enable takes effect.")
        self.forward_unique = True
        self.unique = P.Unique().add_prim_attr('primitive_target', 'CPU')
        self.unique.add_prim_attr('cache_enable', True)
        self.embedding_table.cache_enable = self.cache_enable
        self.embedding_table.cache_shape = (self.vocab_cache_size, self.embedding_size)
        self.reshape_first = P.Reshape().add_prim_attr('primitive_target', 'CPU')

    def _process_vocab_cache(self, slice_mode):
        """PS embeddingLookup cache check and process."""
        self.cache_enable = False
        if self.vocab_cache_size > 0:
            if self.target == 'CPU':
                logger.warning("The configuration of 'vocab_cache_size' is valid only in 'DEVICE' target, "
                               "current target is CPU, so it will be ignored.")
                return
            enable_ps = _get_ps_context("enable_ps")
            if not enable_ps:
                logger.warning("The configuration of 'vocab_cache_size' is valid only in parameter server trainning "
                               "mode, current mode is not parameter server trainning mode, so it will be ignored.")
                return
            parallel_mode = _get_parallel_mode()
            is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
            if is_auto_parallel:
                rank_size = get_group_size()
                rank_id = get_rank()
                full_batch = _get_full_batch()
                if rank_size > 1 and not (full_batch and slice_mode == "table_row_slice"):
                    raise ValueError("The embeddingLookup cache of parameter server parallel only be used "
                                     "in 'full_batch' and 'table_row_slice' parallel strategy.")
                self.vocab_cache_size = self.vocab_cache_size * rank_size
                _set_rank_id(rank_id)
            self.cache_enable = True
            if _is_role_worker():
                self.vocab_size = self.vocab_cache_size
                if context.get_context("enable_sparse") != self.sparse:
                    raise ValueError("The value of parameter 'sparse' must be same for all EmbeddingLookup "
                                     "kernels and equal the value of 'enable_sparse' in context setting in "
                                     "parameter server cache mode")

    def _set_voacb_cache_enable_for_ps(self, vocab_cache_size, embedding_size, vocab_size):
        """PS embeddingLookup cache enable set."""
        self.embedding_table.cache_enable = True
        self.embedding_table.is_param_ps = True
        _set_cache_enable(True)
        if self.sparse:
            self.forward_unique = True
        if _is_role_worker():
            _insert_hash_table_size(self.embedding_table.name, vocab_cache_size, embedding_size, vocab_size)

    def construct(self, indices):
        if self.target == "CPU":
            out = self.embeddinglookup(self.embedding_table, indices, 0)
        else:
            if self.forward_unique:
                shp = self.shape(indices) + (self.embedding_size,)
                indices_flatten = self.reshape_first(indices, (-1,))
                unique_id, unique_idx = self.unique(indices_flatten)
                weight_unique = self.gatherv2(self.embedding_table, unique_id, 0)
                weight_flatten = self.gather_revert(weight_unique, unique_idx, 0)
                out = self.reshape(weight_flatten, shp)
            else:
                out = self.gatherv2(self.embedding_table, indices, 0)
        if self.max_norm is not None:
            axis = _make_axis_range(F.rank(indices), F.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)
        return out


class MultiFieldEmbeddingLookup(EmbeddingLookup):
    r"""
    Returns a slice of input tensor based on the specified indices and the field ids. This operation
    supports looking up embeddings using multi hot and one hot fields simultaneously.

    Note:
        When 'target' is set to 'CPU', this module will use
        P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU') which
        specified 'offset = 0' to lookup table.
        When 'target' is set to 'DEVICE', this module will use P.Gather() which
        specified 'axis = 0' to lookup table.
        The vectors with the same field_ids  will be combined by the 'operator', such as 'SUM', 'MAX' and
        'MEAN'. Ensure the input_values of the padded id is zero, so that they can be ignored. The final
        output will be zeros if the sum of absolute weight of the field is zero. This class only
        supports ['table_row_slice', 'batch_slice' and 'table_column_slice']. For the operation 'MAX' on
        device Ascend, there is a constrain where batch_size * (seq_length + field_size) < 3500.

    Args:
        vocab_size (int): The size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        field_size (int): The field size of the final outputs.
        param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        target (str): Specifies the target where the op is executed. The value must in
            ['DEVICE', 'CPU']. Default: 'CPU'.
        slice_mode (str): The slicing way in semi_auto_parallel/auto_parallel. The value must get through
            nn.EmbeddingLookup. Default: nn.EmbeddingLookup.BATCH_SLICE.
        feature_num_list (tuple): The accompaniment array in field slice mode. This is unused currently.
        max_norm (Union[float, None]): A maximum clipping value. The data type must be float16, float32
                                       or None. Default: None
        sparse (bool): Using sparse mode. When 'target' is set to 'CPU', 'sparse' has to be true. Default: True.
        operator (str): The pooling method for the features in one field. Support 'SUM, 'MEAN' and 'MAX'

    Inputs:
        - **input_indices** (Tensor) - The shape of tensor is :math:`(batch\_size, seq\_length)`.
          Specifies the indices of elements of the original Tensor. Input_indices must be a 2d tensor in
          this interface. Type is Int32, Int64.
        - **input_values** (Tensor) - The shape of tensor is :math:`(batch\_size, seq\_length)`.
          Specifies the weights of elements of the input_indices. The lookout vector will multiply with
          the input_values. Type is Float32.
        - **field_ids** (Tensor)  - The shape of tensor is :math:`(batch\_size, seq\_length)`.
          Specifies the field id of elements of the input_indices. Type is Int32.

    Outputs:
        Tensor, the shape of tensor is :math:`(batch\_size, field\_size, embedding\_size)`. Type is Float32.

    Raises:
        TypeError: If `vocab_size` or `embedding_size` or `field_size` is not an int.
        TypeError: If `sparse` is not a bool or `feature_num_list` is not a tuple.
        ValueError: If `vocab_size` or `embedding_size` or `field_size` is less than 1.
        ValueError: If `target` is neither 'CPU' nor 'DEVICE'.
        ValueError: If `slice_mode` is not one of 'batch_slice', 'field_slice', 'table_row_slice', 'table_column_slice'.
        ValueError: If `sparse` is False and `target` is 'CPU'.
        ValueError: If `slice_mode` is 'field_slice' and `feature_num_list` is None.
        ValueError: If `operator` is not one of 'SUM', 'MAX', 'MEAN'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_indices = Tensor([[2, 4, 6, 0, 0], [1, 3, 5, 0, 0]], mindspore.int32)
        >>> input_values = Tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], mindspore.float32)
        >>> field_ids = Tensor([[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]], mindspore.int32)
        >>> net = nn.MultiFieldEmbeddingLookup(10, 2, field_size=2, operator='SUM', target='DEVICE')
        >>> out = net(input_indices, input_values, field_ids)
        >>> print(out.shape)
        (2, 2, 2)
    """
    OPERATOR_SUM = 'SUM'
    OPERATOR_MEAN = 'MEAN'
    OPERATOR_MAX = 'MAX'
    def __init__(self, vocab_size, embedding_size, field_size, param_init='normal', target='CPU',
                 slice_mode='batch_slice', feature_num_list=None, max_norm=None, sparse=True, operator='SUM'):
        super(MultiFieldEmbeddingLookup, self).__init__(vocab_size, embedding_size, param_init, target,
                                                        slice_mode, feature_num_list, max_norm, sparse)
        self.field_size = validator.check_positive_int(field_size, 'field_size')
        self.operator = operator

        self.mul = P.Mul()
        self.inf_mask_mul = P.Mul()
        self.bias_add = P.Add()
        self.inf_add = P.Add()
        self.merge_op = None
        self.count_op = P.UnsortedSegmentSum()
        self.abs = P.Abs()
        self.equal = P.Equal()
        self.add = P.Add()
        self.cast = P.Cast()
        self.div_no_nan = P.DivNoNan()
        self.expand = P.ExpandDims()
        self.max_mask_mul = P.Mul()
        self.max_no_equal = P.NotEqual()

        if operator == MultiFieldEmbeddingLookup.OPERATOR_SUM:
            self.merge_op = P.UnsortedSegmentSum()
        elif operator == MultiFieldEmbeddingLookup.OPERATOR_MAX:
            self.merge_op = P.UnsortedSegmentMax()
        elif operator == MultiFieldEmbeddingLookup.OPERATOR_MEAN:
            self.merge_op = P.UnsortedSegmentSum()
        else:
            raise ValueError("The operator supports ['SUM', 'MAX', 'MEAN'], but found: "+str(operator))

        parallel_mode = _get_parallel_mode()
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if slice_mode in ["table_row_slice", "batch_slice"] and is_auto_parallel:
            self.merge_op.shard(((get_group_size(), 1, 1), (get_group_size(), 1)))
            self.expand.shard(((get_group_size(),),))
            self.bias_add.shard(((1, 1), (1, 1)))
            self.mul.shard(((get_group_size(), 1, 1), (get_group_size(), 1, 1)))
            self.count_op.shard(((get_group_size(), 1), (get_group_size(), 1)))
            self.add.shard(((get_group_size(),), (get_group_size(),)))
            self.div_no_nan.shard(((get_group_size(), 1), (get_group_size(), 1)))
            self.max_mask_mul.shard(((get_group_size(), 1), (get_group_size(), 1)))
            self.max_no_equal.shard(((1,), ()))
            if operator == MultiFieldEmbeddingLookup.OPERATOR_MAX:
                self.equal.shard(((get_group_size(), 1, 1), ()))
                self.inf_mask_mul.shard(((get_group_size(), 1, 1), ()))
                self.merge_op.shard(((get_group_size(), 1), (get_group_size(),)))
                self.count_op.shard(((get_group_size(),), (get_group_size(),)))
                self.inf_add.shard(((get_group_size(), 1, 1), (get_group_size(), 1, 1)))
        elif slice_mode == "table_column_slice" and is_auto_parallel:
            self.merge_op.shard(((1, 1, get_group_size()), (1, 1)))
            self.div_no_nan.shard(((1, get_group_size()), (1, 1)))
            self.bias_add.shard(((1, 1), (1, 1)))
            self.mul.shard(((1, 1, 1), (1, 1, get_group_size())))
            self.count_op.shard(((1, 1), (1, 1)))
            self.add.shard(((1,), (1,)))
            self.max_mask_mul.shard(((1, get_group_size()), (1, 1)))
            self.expand.shard(((1,),))
            self.max_no_equal.shard(((1,), ()))
            if operator == MultiFieldEmbeddingLookup.OPERATOR_MAX:
                self.equal.shard(((1, 1, 1), ()))
                self.inf_mask_mul.shard(((1, 1, 1), ()))
                self.merge_op.shard(((1, get_group_size()), (1,)))
                self.count_op.shard(((1,), (1,)))
                self.inf_add.shard(((1, 1, get_group_size()), (1, 1, 1)))
        else:
            if is_auto_parallel:
                raise ValueError("slice_mode should be  ['table_row_slice', 'batch_slice' and \
                       'table_column_slice'], but get " + str(slice_mode))

        # Min value for fp32
        self.negative_inf_value = -3.402823466E+38

    def construct(self, input_indices, input_values, field_ids):

        _check_input_2d(F.shape(input_indices), "input_indices", self.cls_name)
        _check_input_2d(F.shape(input_values), "input_values", self.cls_name)
        _check_input_2d(F.shape(field_ids), "field_ids", self.cls_name)
        _check_input_dtype(F.dtype(input_indices), "input_indices", [mstype.int32, mstype.int64], self.cls_name)
        _check_input_dtype(F.dtype(input_values), "input_values", [mstype.float32], self.cls_name)
        _check_input_dtype(F.dtype(field_ids), "field_ids", [mstype.int32], self.cls_name)

        batch_size = self.shape(input_indices)[0]
        num_segments = batch_size * self.field_size
        bias = Range(0, num_segments, self.field_size)()
        bias = self.reshape(bias, (batch_size, -1))
        field_ids = self.bias_add(field_ids, bias)

        if self.target == "CPU":
            out = self.embeddinglookup(self.embedding_table, input_indices, 0)
        else:
            if self.forward_unique:
                shp = self.shape(input_indices) + (self.embedding_size,)
                indices_flatten = self.reshape(input_indices, (-1,))
                unique_id, unique_idx = self.unique(indices_flatten)
                weight_unique = self.gatherv2(self.embedding_table, unique_id, 0)
                weight_flatten = self.gather_revert(weight_unique, unique_idx, 0)
                out = self.reshape(weight_flatten, shp)
            else:
                out = self.gatherv2(self.embedding_table, input_indices, 0)
        if self.max_norm is not None:
            axis = _make_axis_range(F.rank(input_indices), F.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)

        weights = self.reshape(input_values, (batch_size, self.shape(input_indices)[1], 1))
        embedding = self.mul(weights, out)

        if self.operator == 'MAX':
            # Fill the padding value to -inf, so the padded value will not influence the results
            negative_inf_mask = self.cast(self.equal(weights, 0), mstype.float32)
            inf_mask = self.inf_mask_mul(negative_inf_mask, self.negative_inf_value)
            embedding = self.inf_add(embedding, inf_mask)
            embedding = self.reshape(embedding, (-1, self.embedding_size))
            field_ids = self.reshape(field_ids, (-1,))

        merged_vectors = self.merge_op(embedding, field_ids, num_segments)

        if self.operator == 'MAX':
            value_count = self.count_op(self.abs(self.reshape(input_values, (-1,))), field_ids, num_segments)
            value_zeros = self.cast(self.max_no_equal(value_count, 0.0), mstype.float32)
            count = self.expand(value_zeros, -1)
            merged_vectors = self.max_mask_mul(merged_vectors, count)

        if self.operator == 'MEAN':
            value_count = self.count_op(self.abs(input_values), field_ids, num_segments)
            value_count = self.expand(value_count, -1)
            merged_vectors = self.div_no_nan(merged_vectors, value_count)

        merged_vectors = self.reshape(merged_vectors, (batch_size, self.field_size, -1))
        return merged_vectors
