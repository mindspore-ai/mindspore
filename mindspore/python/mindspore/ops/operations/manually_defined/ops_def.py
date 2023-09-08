# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Operators for nn."""
from __future__ import absolute_import
from __future__ import division

from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import gen_arg_handler as handler


class BatchNorm(Primitive):
    """
    BatchNorm.
    """
    __mindspore_signature__ = (sig.make_sig('input_x', dtype=sig.sig_dtype.T1),
                               sig.make_sig('scale',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T2),
                               sig.make_sig('bias',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T2),
                               sig.make_sig('mean',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T3),
                               sig.make_sig('variance',
                                            sig.sig_rw.RW_WRITE,
                                            dtype=sig.sig_dtype.T3))

    @prim_attr_register
    def __init__(self,
                 is_training=False,
                 epsilon=1e-5,
                 momentum=0.1,
                 data_format="NCHW"):
        """Initialize BatchNorm."""
        if is_training is False:
            self.set_signatures(tuple())
        else:
            self.add_prim_attr('side_effect_mem', True)
        self.is_training = is_training
        self.epsilon = epsilon
        self.momentum = momentum
        self.data_format = handler.format_to_enum(data_format)

    def __call__(self, *args):
        return super().__call__(self, *args, self.is_training, self.epsilon,
                                self.momentum, self.data_format)


def batch_norm_(input_x,
                scale,
                bias,
                mean,
                variance,
                is_training=False,
                epsilon=1e-5,
                momentum=0.1,
                data_format="NCHW"):
    r"""
    Batch Normalization for input data and updated parameters.

    Batch Normalization is widely used in convolutional neural networks. This operation
    applies Batch Normalization over inputs to avoid internal covariate shift as described
    in the paper `Batch Normalization: Accelerating Deep Network Training by Reducing Internal
    Covariate Shift <https://arxiv.org/abs/1502.03167>`_. It rescales and recenters the
    features using a mini-batch of data and the learned parameters can be described
    in the following formula,

    .. math::

        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is scale, :math:`\beta` is bias, :math:`\epsilon` is epsilon,
    :math:`mean` is the mean of :math:`x`,
    :math:`variance` is the variance of :math:`x`.

    .. warning::
        - If the operation is used for inference, and outputs "reserve_space_1" and "reserve_space_2" are available,
            then "reserve_space_1" has the same value as "mean" and "reserve_space_2" has the same value as "variance".
        - For Ascend 310, the result accuracy fails to reach 1‰ due to the square root instruction.

    Note:
        - If `training` is `False`, `weight`, `bias`, `running_mean` and `running_var` are tensors.
        - If `training` is `True`, `weight`, `bias`, `running_mean` and `running_var` are Parameters.

    Args:
        input_x (tensor): tensor of shape :math:`(N, C)`, with float16 or float32 data type.
        scale (Union[tensor, Parameter]): The shape :math:`(C,)`, has the same data type with `weight`.
        bias (Union[tensor, Parameter]): The shape :math:`(C,)`, has the same data type with `weight`.
        mean (Union[tensor, Parameter]): The shape :math:`(C,)`, with float16 or float32 data type.
        variance (Union[tensor, Parameter]): The shape :math:`(C,)`, has the same data type with `weight`.
        is_training (bool, optional): If `training` is `True`, `mean` and `variance` are computed during training.
            If `training` is `False`, they're loaded from checkpoint during inference. Default: False.
        epsilon (float): A small value added for numerical stability.
            Default: ``1e-5``, value must be (0, 1] .
        momentum (float): The hyper parameter to compute moving average for running_mean and running_var
            (e.g. :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`).
            Momentum value must be [0, 1].
            Default: ``0.1`` .
        data_format (str): The optional value for data format, is ``'NHWC'`` or ``'NCHW'``,
            and the ``'NHWC'`` format is only supported in GPU target.
            Default: ``"NCHW"`` .

    Returns:
        output_x (Tensor): The same type and shape as the input_x. The shape is :math:`(N, C)`.
        batch_mean (Tensor): Tensor of shape :math:`(C,)`.
        batch_variance (Tensor): Tensor of shape :math:`(C,)`.
        reserve_space_1 (Tensor): Tensor of shape :math:`(C,)`.
        reserve_space_2 (Tensor): Tensor of shape :math:`(C,)`.

    Raises:
        TypeError: If `is_training` is not a bool.
        TypeError: If dtype of `epsilon` or `momentum` is not float.
        TypeError: If `data_format` is not a str.
        TypeError: If `input_x`, `scale`, `bias`, `mean` or `variance` is not a Tensor.
        TypeError: If dtype of `input_x`, `scale` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.ones([2, 2]), mindspore.float32)
        >>> scale = Tensor(np.ones([2]), mindspore.float32)
        >>> bias = Tensor(np.ones([2]), mindspore.float32)
        >>> mean = Tensor(np.ones([2]), mindspore.float32)
        >>> variance = Tensor(np.ones([2]), mindspore.float32)
        >>> output = ops.batch_norm_(input_x, scale, bias, mean, variance, is_training, epsilon, momentum, data_format)
        >>> print(output[0])
        [[1. 1.]
        [1. 1.]]
    """
    batch_norm_op = _get_cache_prim(BatchNorm)(is_training, epsilon, momentum,
                                               data_format)
    return batch_norm_op(input_x, scale, bias, mean, variance)
