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
from __future__ import absolute_import

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Normal
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _get_full_batch
from mindspore import _checkparam as Validator
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.nn.layer.basic import ClipByNorm
from mindspore.nn.cell import Cell

__all__ = ['Embedding', 'EmbeddingExt', 'EmbeddingLookup', 'MultiFieldEmbeddingLookup']


@_primexpr
def _check_input_2d(input_shape, param_name, func_name):
    if len(input_shape) != 2:
        raise ValueError(
            f"For '{func_name}', the dimension of '{param_name}' must be 2d, but got {len(input_shape)}")


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    Validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


class Embedding(Cell):
    r"""
    A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using
    indices. The input to the module is a list of indices, and the output is
    the corresponding word embeddings.

    .. warning::
        After version 2.9.0, the constructor signature will be changed to
        ``Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
        scale_grad_by_freq=False, sparse=False, dtype=None)``.
        Parameters `vocab_size` and `embedding_size` will be renamed to `num_embeddings` and `embedding_dim`,
        and `use_one_hot` and `embedding_table` will be removed.

    Note:
        When 'use_one_hot' is set to True, the type of the `x` must be mindspore.int32.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot (bool): Specifies whether to apply one_hot encoding form. Default: ``False`` .
        embedding_table (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `mindspore.common.initializer
            <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_
            for the values of string when a string is specified. Default: ``'normal'`` .
        dtype (:class:`mindspore.dtype`): Data type of `x`. Default: ``mstype.float32`` .
        padding_idx (int, None): When the padding_idx encounters index, the output embedding vector of this index
                                 will be initialized to zero. Default: ``None`` . The feature is inactivated.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(\text{batch_size}, \text{x_length})`. The elements of
          the Tensor must be integer and not larger than vocab_size. Otherwise the corresponding embedding vector will
          be zero. The data type is int32 or int64.

    Outputs:
        Tensor of shape :math:`(\text{batch_size}, \text{x_length}, \text{embedding_size})`.

    Raises:
        TypeError: If `vocab_size` or `embedding_size` is not an int.
        TypeError: If `use_one_hot` is not a bool.
        ValueError: If `padding_idx` is an int which not in range [0, `vocab_size`).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Embedding(20000, 768,  True)
        >>> x = Tensor(np.ones([8, 128]), mindspore.int32)
        >>> # Maps the input word IDs to word embedding.
        >>> output = net(x)
        >>> result = output.shape
        >>> print(result)
        (8, 128, 768)
    """

    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal',
                 dtype=mstype.float32, padding_idx=None):
        """Initialize Embedding."""
        super().__init__()
        self.vocab_size = Validator.check_value_type(
            'vocab_size', vocab_size, [int], self.cls_name)
        self.embedding_size = Validator.check_value_type(
            'embedding_size', embedding_size, [int], self.cls_name)
        Validator.check_value_type(
            'use_one_hot', use_one_hot, [bool], self.cls_name)
        Validator.check_subclass(
            "dtype", dtype, mstype.number_type, self.cls_name)
        self.use_one_hot = use_one_hot
        self.dtype = dtype
        self.init_tensor = initializer(
            embedding_table, [vocab_size, embedding_size])
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = Validator.check_int_range(padding_idx, 0, vocab_size, Validator.INC_LEFT,
                                                         "padding_idx", self.cls_name)
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            init_tensor_type = self.init_tensor.dtype
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor, init_tensor_type)
        self.embedding_table = Parameter(
            self.init_tensor, name='embedding_table')
        self.expand = ops.ExpandDims()
        self.reshape_flat = ops.Reshape()
        self.shp_flat = (-1,)
        self.gather = ops.Gather()
        self.one_hot = ops.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.array_mul = ops.MatMul()
        self.reshape = ops.Reshape()
        self.get_shp = ops.Shape()
        self.concat = ops.Concat()

    def construct(self, ids):
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(
                flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        return f'vocab_size={self.vocab_size}, embedding_size={self.embedding_size}, use_one_hot={self.use_one_hot}, ' \
            f'embedding_table={self.embedding_table}, dtype={self.dtype}, padding_idx={self.padding_idx}'


class EmbeddingExt(Cell):
    r"""
    The value in `input` is used as the index, and the corresponding embedding vector is queried from `weight` .

    .. warning::
        - This is an experimental API that is subject to change or deletion.
        - On Ascend, the behavior is unpredictable when the value of `input` is invalid.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int, optional): If specified, the embedding vector at padding_idx is not updated during training.
            For a newly constructed Embedding, the embedding vector at padding_idx will default to all zeros.
            The value should be in range `[-num_embeddings, num_embeddings)`. Default: ``None``.
        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is renormalized
            to have norm max_norm. Vector specified by `input` where p is specified by `norm_type`; if the result is
            larger than `max_norm`, update the embedding vector with :math:`\frac{max\_norm}{result+1e^{-7}}`.
            Default: ``None``.
        norm_type (float, optional): Indicates the value of p in p-norm. Default: ``2.0``.
        scale_grad_by_freq (bool, optional): If ``True``, the gradient will be scaled by the inverse of frequency
            of the index in `input`. Default: ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. `weight` matrix will be a sparse tensor which
            has not been supported. Default: ``False``.
        _weight (Tensor, optional): Use to initialize the `weight` of Embedding. If ``None``, the weight will be
            initialized from normal distribution :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})`. Default: ``None``.
        _freeze(bool, optional): If `weight`, the learnable weight of this module, should be frozen.
            Default: ``False``.
        dtype (mindspore.dtype, optional): Dtype of Embedding's `weight`. It is meaningless when `_weight` is
            not None. Default: ``None``.

    Variables:
        - **weight** (Parameter) - The learnable weights of this module of shape (num_embeddings, embedding_dim),
          which is initialized from :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})` or `_weight` .

    Inputs:
        - **input** (Tensor) - The input tensor containing the indices to extract, which in range `[0, num_embeddings)`.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> mindspore.set_seed(0)
        >>> input = mindspore.tensor([[1, 0, 1, 1], [0, 0, 1, 0]])
        >>> embedding = mindspore.nn.layer.embedding.EmbeddingExt(num_embeddings=10, embedding_dim=3)
        >>> output = embedding(input)
        >>> print(output)
        [[[ 0.6712398   0.5407775   1.0317237]
          [-0.49091062 -0.42302188 -1.4807187]
          [ 0.6712398   0.5407775   1.0317237]
          [ 0.6712398   0.5407775   1.0317237]]
         [[-0.49091062 -0.42302188 -1.4807187]
          [-0.49091062 -0.42302188 -1.4807187]
          [ 0.6712398   0.5407775   1.0317237]
          [-0.49091062 -0.42302188 -1.4807187]]]
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, dtype=None):
        """Initialize Embedding."""
        super().__init__()
        self.sparse = Validator.check_value_type('sparse', sparse, [bool], self.cls_name)
        if self.sparse:
            raise ValueError("For Embedding, the scenerio, where `sparse` is True, has not be supported.")
        self.num_embeddings = Validator.check_value_type(
            'num_embeddings', num_embeddings, [int], self.cls_name)
        self.embedding_dim = Validator.check_value_type(
            'embedding_dim', embedding_dim, [int], self.cls_name)
        self.dtype = dtype if dtype is not None else mstype.float32
        Validator.check_subclass(
            "dtype", self.dtype, mstype.number_type, self.cls_name)
        self.padding_idx = padding_idx
        if _weight is None:
            init_tensor = Tensor(shape=[num_embeddings, embedding_dim], dtype=self.dtype, init=Normal(1, 0))
            init_tensor = self._zero_weight_by_index(init_tensor)
            self.weight = Parameter(init_tensor, name='weight', requires_grad=not _freeze)
        else:
            if _weight.shape != (num_embeddings, embedding_dim):
                raise ValueError(f"For Embedding, shape of weight should be match with num_embeddings "
                                 f"and embedding_dim, but got weight.shape: {_weight.shape}, "
                                 f"and (num_embeddings, embedding_dim): ({num_embeddings}, {embedding_dim})")
            self.weight = Parameter(_weight, name='weight', requires_grad=not _freeze)

        self.max_norm = max_norm
        if max_norm is not None:
            self.max_norm = Validator.check_value_type('max_norm', max_norm, [float], self.cls_name)

        self.norm_type = norm_type
        if norm_type is not None:
            self.norm_type = Validator.check_value_type('norm_type', norm_type,
                                                        [float], self.cls_name)

        self.scale_grad_by_freq = scale_grad_by_freq
        if scale_grad_by_freq is not None:
            self.scale_grad_by_freq = Validator.check_value_type('scale_grad_by_freq',
                                                                 scale_grad_by_freq,
                                                                 [bool], self.cls_name)

    def _zero_weight_by_index(self, init_tensor):
        if self.padding_idx is not None:
            self.padding_idx = Validator.check_int_range(self.padding_idx, -self.num_embeddings, self.num_embeddings,
                                                         Validator.INC_LEFT, "padding_idx", self.cls_name)
            if isinstance(init_tensor, Tensor) and init_tensor.init is not None:
                init_tensor = init_tensor.init_data()
            init_tensor[self.padding_idx] = 0

        return init_tensor

    def construct(self, input):
        return ops.embedding(input, self.weight, self.padding_idx, self.max_norm,
                             self.norm_type, self.scale_grad_by_freq)

    def extend_repr(self):
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, ' \
               f'padding_idx={self.padding_idx}, max_norm={self.max_norm}, norm_type={self.norm_type}, ' \
               f'scale_grad_by_freq={self.scale_grad_by_freq}, dtype={self.dtype}'


@_primexpr
def _make_axis_range(start, end):
    axis = tuple(range(start, end))
    return axis


class EmbeddingLookup(Cell):
    r"""
    EmbeddingLookup layer.
    Same function as the embedding layer, mainly used for heterogeneous parallel scenarios
    where large-scale embedding layers exist
    when automatic parallelism or semi-automatic parallelism is present.

    .. warning::
        This interface is deprecated and will be removed after version 2.9.0.

    Note:
        - When `target` is set to 'CPU', this module will use
          ops.EmbeddingLookup().set_device('CPU') which
          specified 'offset = 0' to lookup table.
        - When `target` is set to 'DEVICE', this module will use ops.Gather() which
          specified 'axis = 0' to lookup table.
        - In field slice mode, the manual_shapes must be given. It is a tuple ,where
          the element is vocab[i], vocab[i] is the row numbers for i-th part.
        - This module does not support the PyNative mode.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: ``'normal'`` .
        target (str): Specifies the target where the op is executed. The value must in
            [ ``'DEVICE'`` , ``'CPU'`` ]. Default: ``'CPU'`` .
        slice_mode (str): The slicing way in semi_auto_parallel/auto_parallel. Default: ``'batch_slice'`` .

          - batch_slice (str): Divides the input index tensor into batches and retrieves
            the corresponding embedding vectors. This is applicable when each sample has the same number of indices.
          - field_slice (str): Divides the input index tensor into fields and retrieves the corresponding embedding
            vectors. This is applicable when each sample may have a different number of indices, but have the same
            feature dimensions.
          - table_row_slice (str): Treats the input index tensor as a 2D table, divides it by rows, and retrieves
            the corresponding embedding vectors.
          - table_column_slice (str): Treats the input index tensor as a 2D table, divides it by columns, and retrieves
            the corresponding embedding vectors.

        manual_shapes (tuple): The accompaniment array in field slice mode. Default: ``None`` .
        max_norm (Union[float, None]): A maximum clipping value. The data type must be float16, float32
                                       or None. Default: ``None`` .
        sparse (bool): Using sparse mode. When 'target' is set to 'CPU', 'sparse' has to be true. Default: ``True`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
          Specifies the indices of elements of the original Tensor. Values can be out of range of embedding_table,
          and the exceeding part will be filled with 0 in the output. Values does not support negative and the result
          is undefined if values are negative. Input_indices must only be a 2d tensor in
          this interface when run in semi auto parallel/auto parallel mode.

    Outputs:
        Tensor, the shape of tensor is :math:`(z_1, z_2, ..., z_N)`.

    Raises:
        TypeError: If `vocab_size` or `embedding_size` is not an int.
        TypeError: If `sparse` is not a bool or `manual_shapes` is not a tuple.
        ValueError: If `vocab_size` or `embedding_size` is less than 1.
        ValueError: If `target` is neither 'CPU' nor 'DEVICE'.
        ValueError: If `slice_mode` is not one of 'batch_slice' or 'field_slice' or
                    'table_row_slice' or 'table_column_slice'.
        ValueError: If `sparse` is False and `target` is 'CPU'.
        ValueError: If `slice_mode` is 'field_slice' and `manual_shapes` is None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
        >>> result = nn.EmbeddingLookup(4, 2)(input_indices)
        >>> print(result.shape)
        (2, 2, 2)
    """
    BATCH_SLICE = "batch_slice"
    FIELD_SLICE = "field_slice"
    TABLE_ROW_SLICE = "table_row_slice"
    TABLE_COLUMN_SLICE = "table_column_slice"

    def __init__(self, vocab_size, embedding_size, param_init='normal',
                 target='CPU', slice_mode='batch_slice', manual_shapes=None,
                 max_norm=None, sparse=True, dtype=mstype.float32):
        """Initialize EmbeddingLookup."""
        super().__init__()
        Validator.check_value_type('sparse', sparse, [bool], self.cls_name)
        self.vocab_size = Validator.check_positive_int(
            vocab_size, 'vocab_size')
        self.target = target
        self.sparse = sparse
        self.forward_unique = False
        Validator.check_string(
            target, ['CPU', 'DEVICE'], 'target', self.cls_name)
        if not sparse and target == 'CPU':
            raise ValueError(f"For '{self.cls_name}', 'sparse' must be True when 'target' is \"CPU\", "
                             f"but got 'sparse': {sparse} and 'target': {target}")
        if sparse:
            self.gatherv2 = ops.SparseGatherV2()
        else:
            self.gatherv2 = ops.Gather()
        self.embeddinglookup = ops.EmbeddingLookup().set_device('CPU')
        self.embedding_size = Validator.check_positive_int(
            embedding_size, 'embedding_size', self.cls_name)
        self.embedding_table = Parameter(initializer(param_init, [self.vocab_size, self.embedding_size],
                                                     dtype=dtype), name='embedding_table')
        parallel_mode = _get_parallel_mode()
        is_auto_parallel = parallel_mode in (
            ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.gather_revert = ops.Gather()
        self.reshape_first = ops.Reshape()
        self.reshape = ops.Reshape()
        self.unique = ops.Unique()
        self.shape = ops.Shape()
        if is_auto_parallel:
            self.unique = ops.Unique().shard(((1,),))
        indices_shape_size = 2
        if slice_mode == "field_slice" and is_auto_parallel:
            if not manual_shapes:
                raise ValueError(f"For '{self.cls_name}', the 'manual_shapes' should not be none "
                                 f"when the 'slice_mode' is \"filed_slice\", but got {manual_shapes}.")
            if not isinstance(manual_shapes, tuple):
                raise TypeError(f"For '{self.cls_name}', the type of 'manual_shapes' must be tuple(int), "
                                f"but got {type(manual_shapes).__name__}!")
            for dim in manual_shapes:
                Validator.check_positive_int(
                    dim, 'manual shape dim', self.cls_name)
            self.gatherv2.add_prim_attr("manual_split", manual_shapes)
            self.embeddinglookup.add_prim_attr("manual_split", manual_shapes)
            self.gatherv2.shard(((get_group_size(), 1), (1, get_group_size())))
            self.embeddinglookup.shard(
                ((get_group_size(), 1), (1, get_group_size())))
        elif slice_mode == "table_row_slice" and is_auto_parallel:
            full_batch = _get_full_batch()
            if (target == 'DEVICE' and not full_batch):
                indices_shape_size = 1
                self.gather_revert.shard(((1, 1), (get_group_size(),)))
                self.forward_unique = True
            indices_strategy = (1,)*indices_shape_size
            self.gatherv2.shard(((get_group_size(), 1), indices_strategy))
            self.embeddinglookup.shard(
                ((get_group_size(), 1), indices_strategy))
        elif slice_mode == "table_column_slice" and is_auto_parallel:
            if target == 'DEVICE':
                indices_shape_size = 1
                self.gather_revert.shard(((1, get_group_size()), (1,)))
                self.forward_unique = True
            indices_strategy = (1,)*indices_shape_size
            self.gatherv2.shard(((1, get_group_size()), indices_strategy))
            self.embeddinglookup.shard(
                ((1, get_group_size()), indices_strategy))
        elif slice_mode == "batch_slice" and is_auto_parallel:
            indices_strategy = [get_group_size()]
            indices_strategy.extend([1] * (indices_shape_size - 1))
            indices_strategy = tuple(indices_strategy)
            self.gatherv2.shard(((1, 1), indices_strategy))
            self.embeddinglookup.shard(((1, 1), indices_strategy))
        else:
            if is_auto_parallel:
                support_mode = ["field_slice", "table_row_slice",
                                "table_column_slice", "batch_slice"]
                raise ValueError(f"For '{self.cls_name}', the 'slice_mode' must be in {support_mode}, "
                                 f"but got \"{slice_mode}\".")
        self.embedding_table.unique = self.forward_unique
        self.max_norm = max_norm
        if self.max_norm is not None:
            self.max_norm = Validator.check_positive_float(
                self.max_norm, 'max_norm', self.cls_name)
            self.max_norm = Tensor(self.max_norm, dtype=mstype.float32)

    def construct(self, indices):
        if self.target == "CPU":
            out = self.embeddinglookup(self.embedding_table, indices, 0)
        else:
            if self.forward_unique:
                shp = self.shape(indices) + (self.embedding_size,)
                indices_flatten = self.reshape_first(indices, (-1,))
                unique_id, unique_idx = self.unique(indices_flatten)
                weight_unique = self.gatherv2(
                    self.embedding_table, unique_id, 0)
                weight_flatten = self.gather_revert(
                    weight_unique, unique_idx, 0)
                out = self.reshape(weight_flatten, shp)
            else:
                out = self.gatherv2(self.embedding_table, indices, 0)
        if self.max_norm is not None:
            axis = _make_axis_range(ops.rank(indices), ops.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)
        return out


class MultiFieldEmbeddingLookup(EmbeddingLookup):
    r"""
    Returns a slice of input tensor based on the specified indices and the field ids. This operation
    supports looking up embeddings using multi hot and one hot fields simultaneously.

    .. warning::
        This interface is deprecated and will be removed after version 2.9.0.

    Note:
        When 'target' is set to 'CPU', this module will use
        ops.EmbeddingLookup().set_device('CPU') which
        specified 'offset = 0' to lookup table.
        When 'target' is set to 'DEVICE', this module will use ops.Gather() which
        specified 'axis = 0' to lookup table.
        The vectors with the same field_ids will be combined by the `operator`, such as 'SUM', 'MAX' and
        'MEAN'. Ensure the input_values of the padded id is zero, so that they can be ignored. The final
        output will be zeros if the sum of absolute weight of the field is zero. This class only
        supports ['table_row_slice', 'batch_slice' and 'table_column_slice']. For the operation 'MAX' on
        device Ascend, there is a constraint where :math:`batch\_size * (seq\_length + field\_size) < 3500`.

    Args:
        vocab_size (int): The size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        field_size (int): The field size of the final outputs.
        param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: ``'normal'`` .
        target (str): Specifies the target where the op is executed. The value must in
            [ ``'DEVICE'`` , ``'CPU'`` ]. Default: ``'CPU'`` .
        slice_mode (str): The slicing way in semi_auto_parallel/auto_parallel. Default: ``'batch_slice'``.

          - batch_slice (str): Divides the input index tensor into batches and retrieves
            the corresponding embedding vectors. This is applicable when each sample has the same number of indices.
          - field_slice (str): Divides the input index tensor into fields and retrieves the corresponding embedding
            vectors. This is applicable when each sample may have a different number of indices, but have the same
            feature dimensions.
          - table_row_slice (str): Treats the input index tensor as a 2D table, divides it by rows, and retrieves
            the corresponding embedding vectors.
          - table_column_slice (str): Treats the input index tensor as a 2D table, divides it by columns, and retrieves
            the corresponding embedding vectors.

        feature_num_list (tuple): The accompaniment array in field slice mode. This is unused currently.
            Default: ``None`` .
        max_norm (Union[float, None]): A maximum clipping value. The data type must be float16, float32.
            Default: ``None`` .
        sparse (bool): Using sparse mode. When 'target' is set to ``'CPU'`` , 'sparse' has to be true.
            Default: ``True`` .
        operator (str): The pooling method for the features in one field. Support ``'SUM'`` , ``'MEAN'`` and
            ``'MAX'`` . Default: ``'SUM'`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **input_indices** (Tensor) - The shape of tensor is :math:`(batch\_size, seq\_length)`.
          Specifies the indices of elements of the original Tensor. Input_indices must be a 2d tensor in
          this interface. Type is Int32, Int64.
        - **input_values** (Tensor) - The shape of tensor is :math:`(batch\_size, seq\_length)`.
          Specifies the weights of elements of the input_indices. The lookout vector will multiply with
          the input_values. Type is float32.
        - **field_ids** (Tensor) - The shape of tensor is :math:`(batch\_size, seq\_length)`.
          Specifies the field id of elements of the input_indices. Type is Int32.

    Outputs:
        Tensor, the shape of tensor is :math:`(batch\_size, field\_size, embedding\_size)`. Type is float32.

    Raises:
        TypeError: If `vocab_size` or `embedding_size` or `field_size` is not an int.
        TypeError: If `sparse` is not a bool or `feature_num_list` is not a tuple.
        ValueError: If `vocab_size` or `embedding_size` or `field_size` is less than 1.
        ValueError: If `target` is neither ``'CPU'`` nor ``'DEVICE'``.
        ValueError: If `slice_mode` is not one of ``'batch_slice'``, ``'field_slice'``, ``'table_row_slice'``,
                    ``'table_column_slice'`` .
        ValueError: If `sparse` is False and `target` is ``'CPU'`` .
        ValueError: If `slice_mode` is ``'field_slice'`` and `feature_num_list` is None.
        ValueError: If `operator` is not one of ``'SUM'``, ``'MAX'``, ``'MEAN'`` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
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
                 slice_mode='batch_slice', feature_num_list=None, max_norm=None, sparse=True, operator='SUM',
                 dtype=mstype.float32):
        """Initialize MultiFieldEmbeddingLookup."""
        super().__init__(vocab_size, embedding_size, param_init, target,
                                                        slice_mode, feature_num_list, max_norm, sparse, dtype=dtype)
        self.field_size = Validator.check_positive_int(
            field_size, 'field_size', self.cls_name)
        self.operator = operator

        self.mul = ops.Mul()
        self.inf_mask_mul = ops.Mul()
        self.bias_add = ops.Add()
        self.inf_add = ops.Add()
        self.merge_op = None
        self.count_op = ops.UnsortedSegmentSum()
        self.abs = ops.Abs()
        self.equal = ops.Equal()
        self.add = ops.Add()
        self.cast = ops.Cast()
        self.div_no_nan = ops.DivNoNan()
        self.expand = ops.ExpandDims()
        self.max_mask_mul = ops.Mul()
        self.max_no_equal = ops.NotEqual()

        Validator.check_string(
            operator, ['SUM', 'MAX', 'MEAN'], 'operator', self.cls_name)
        if operator == MultiFieldEmbeddingLookup.OPERATOR_SUM:
            self.merge_op = ops.UnsortedSegmentSum()
        elif operator == MultiFieldEmbeddingLookup.OPERATOR_MAX:
            self.merge_op = ops.UnsortedSegmentMax()
        else:
            self.merge_op = ops.UnsortedSegmentSum()


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
                raise ValueError(
                    f"For '{self.cls_name}', the 'slice_mode' must be in ['table_row_slice', 'batch_slice' "
                    f"and 'table_column_slice'], but got {str(slice_mode)}.")

        # Min value for fp32
        self.negative_inf_value = -3.402823466E+38

    def construct(self, input_indices, input_values, field_ids):
        _check_input_2d(ops.shape(input_indices), "input_indices", self.cls_name)
        _check_input_2d(ops.shape(input_values), "input_values", self.cls_name)
        _check_input_2d(ops.shape(field_ids), "field_ids", self.cls_name)
        _check_input_dtype(ops.dtype(input_indices), "input_indices", [mstype.int32, mstype.int64], self.cls_name)
        _check_input_dtype(ops.dtype(input_values), "input_values", [mstype.float32], self.cls_name)
        _check_input_dtype(ops.dtype(field_ids), "field_ids", [mstype.int32], self.cls_name)

        batch_size = self.shape(input_indices)[0]
        num_segments = batch_size * self.field_size
        bias = ops.tuple_to_array(ops.make_range(0, num_segments, self.field_size))
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
            axis = _make_axis_range(ops.rank(input_indices), ops.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)

        weights = self.reshape(
            input_values, (batch_size, self.shape(input_indices)[1], 1))
        embedding = self.mul(weights, out)

        if self.operator == 'MAX':
            # Fill the padding value to -inf, so the padded value will not influence the results
            negative_inf_mask = self.cast(
                self.equal(weights, 0), mstype.float32)
            inf_mask = self.inf_mask_mul(
                negative_inf_mask, self.negative_inf_value)
            embedding = self.inf_add(embedding, inf_mask)
            embedding = self.reshape(embedding, (-1, self.embedding_size))
            field_ids = self.reshape(field_ids, (-1,))

        merged_vectors = self.merge_op(embedding, field_ids, num_segments)

        if self.operator == 'MAX':
            value_count = self.count_op(self.abs(self.reshape(
                input_values, (-1,))), field_ids, num_segments)
            value_zeros = self.cast(self.max_no_equal(
                value_count, 0.0), mstype.float32)
            count = self.expand(value_zeros, -1)
            merged_vectors = self.max_mask_mul(merged_vectors, count)

        if self.operator == 'MEAN':
            value_count = self.count_op(
                self.abs(input_values), field_ids, num_segments)
            value_count = self.expand(value_count, -1)
            merged_vectors = self.div_no_nan(merged_vectors, value_count)

        merged_vectors = self.reshape(
            merged_vectors, (batch_size, self.field_size, -1))
        return merged_vectors
