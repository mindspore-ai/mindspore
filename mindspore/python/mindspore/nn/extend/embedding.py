# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore.common.initializer import Normal
from mindspore import _checkparam as Validator
from mindspore.nn.cell import Cell
from mindspore import ops
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor

__all__ = ['Embedding']


class Embedding(Cell):
    r"""
    Embedding layer.
    Retrieve the word embeddings in weight stored in the layer using indices specified in `input`.

    .. warning::
        On Ascend, the behavior is unpredictable when the value of `input` is invalid.

    Args:
        num_embeddings (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        padding_idx (int, optional): If the value is not None, the corresponding row of embedding vector
            will not be updated in training. The value of embedding vector at `padding_idx` will default
            to zeros when the Embedding layer is newly constructed. The value should be in range
            `[-num_embeddings, num_embeddings)` if it's not ``None``. Default ``None``.
        max_norm (float, optional): If the value is not None, firstly get the p-norm result of the embedding
            vector specified by `input` where p is specified by `norm_type`; if the result is larger then `max_norm`,
            update the embedding vector` with :math:`\frac{max\_norm}{result+1e^{-7}}`. Default ``None``.
        norm_type (float, optional): Indicated the value of p in p-norm. Default ``2.0``.
        scale_grad_by_freq (bool, optional): If ``True`` the gradients will be scaled by the inverse of frequency
            of the index in `input`. Default ``False``.
        _weight (Tensor, optional): Used to initialize the weight of Embedding. If ``None``, the weight will be
            initialized from normal distribution :math:`{N}(\text{sigma=1.0}, \text{mean=0.0})`. Default ``None``.
        dtype (mindspore.dtype, optional) : Dtype of Parameters. It is meaningless when `_weight` is not None.
            Default: ``mindspore.float32``.

    Inputs:
        - **input** (Tensor) - The indices used to lookup in the embedding vector. The data type must be
          mindspore.int32 or mindspore.int64, and the value should be in range `[0, num_embeddings)`.

    Outputs:
        Tensor, has the same data type as weight, the shape is :math:`(*input.shape, embedding\_dim)`.

    Raises:
        TypeError: If `num_embeddings` is not an int.
        TypeError: If `embedding_dim` is not an int.
        ValueError: If `padding_idx` is out of valid range.
        TypeError: If `max_norm` is not a float.
        TypeError: If `norm_type` is not a float.
        TypeError: If `scale_grad_by_freq` is not a bool.
        TypeError: If `dtype` is not one of mindspore.dtype.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> input = Tensor([[1, 0, 1, 1], [0, 0, 1, 0]])
        >>> embedding = nn.extend.Embedding(num_embeddings=10, embedding_dim=3)
        >>> output = embedding(input)
        >>> print(output)
        [[[-0.0024154  -0.01203444  0.00811537]
          [ 0.00233847 -0.00596091  0.00536799]
          [-0.0024154  -0.01203444  0.00811537]
          [-0.0024154  -0.01203444  0.00811537]]
         [[ 0.00233847 -0.00596091  0.00536799]
          [ 0.00233847 -0.00596091  0.00536799]
          [-0.0024154  -0.01203444  0.00811537]
          [ 0.00233847 -0.00596091  0.00536799]]]
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                 scale_grad_by_freq=False, _weight=None, dtype=mstype.float32):
        """Initialize Embedding."""
        super().__init__()
        self.num_embeddings = Validator.check_value_type(
            'num_embeddings', num_embeddings, [int], self.cls_name)
        self.embedding_dim = Validator.check_value_type(
            'embedding_dim', embedding_dim, [int], self.cls_name)
        Validator.check_subclass(
            "dtype", dtype, mstype.number_type, self.cls_name)
        self.dtype = dtype
        self.padding_idx = padding_idx
        if _weight is None:
            init_tensor = Tensor(shape=[num_embeddings, embedding_dim], dtype=dtype, init=Normal(1, 0))
            init_tensor = self._zero_weight_by_index(init_tensor)
            self.weight = Parameter(init_tensor, name='weight')
        else:
            self.weight = Parameter(_weight)

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
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, '  \
               f'padding_idx={self.padding_idx}, max_norm={self.max_norm}, norm_type={self.norm_type}, ' \
               f'scale_grad_by_freq={self.scale_grad_by_freq}, dtype={self.dtype}'
