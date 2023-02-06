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
"""Dual BatchNorm Implementation"""
from typing import Tuple

import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.hypercomplex.hypercomplex._hc_bn_impl import _BaseBatchNormImpl as HCBatchNormImpl
from mindspore.hypercomplex.utils import get_x_and_y as get_real_and_dual


class _BatchNormImpl(HCBatchNormImpl):
    r"""
    The implementor class of the Batch Normalization layer for dual numbers.

    Implements the functionality specific to dual numbers and needed by the 'BatchNorm' class. This includes:
    getting the norm of dual number, applying scaling and shift to a dual tensor, and updating the running
    mean and variance, which are used during inference.

    Args:
        affine (bool) - A bool value. When set to True, gamma and beta can be learned.
        use_batch_statistics (bool): If true, use the mean value and variance value of current batch data. If false,
            use the mean value and variance value of specified value. If None, the training process will use the mean
            and variance of current batch data and track the running mean and variance, the evaluation process will use
            the running mean and variance.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
        num_features (int): The number of features in the input space.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def scale_and_shift(self,
                        u_x: Tensor,
                        u_y: Tensor,
                        scale_x: Tensor,
                        scale_y: Tensor,
                        shift_x: Tensor,
                        shift_y: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Applies dual scaling and shift to an input tensor.

        This function implements the operation as:

        .. math::
            \begin{align}
            \text{Re(out)} = \text{Re(inp)} * \text{Re(scale)} + \text{Re(shift)}\\
            \text{Du(out)} = \text{Re(inp)} * \text{Du(scale)} + \text{Du(inp)} * \text{Re(scale)} + \text{Du(shift)},
            \end{align}

        where :math:`inp` is the dual input tensors, :math:`scale` and :math:`shift` are dual parameters representing
        the scaling and shift coefficients respectively. :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are
        respectively real and dual parts of the dual-valued expression inside the parentheses.

        Args:
            u_x (Tensor): A tensor of shape (C,), which represents the real part of the normalized inputs.
            u_y (Tensor): A tensor of shape (C,), which represents the dual part of the normalized inputs.
            scale_x (Tensor): A tensor of shape (C,), which represents the real part of the scaling vector.
            scale_y (Tensor): A tensor of shape (C,), which represents the dual part of the scaling vector.
            shift_x (Tensor): A tensor of shape (C,), which represents the real part of the bias vector.
            shift_y (Tensor): A tensor of shape (C,), which represents the dual part of the bias vector.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the dual parts of rescaled and
            recentered inputs.
        """
        out_x = u_x * scale_x + shift_x
        out_y = u_x * scale_y + u_y * scale_x + shift_y
        return out_x, out_y

    def get_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates norm of dual elements of an input tensor.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero. The function implements the operation as:

        .. math::
            \text{out} = \left|\frac{Du(inp)}{2}\right|+\sqrt{Re(inp)^2+\frac{Du(inp)^2}{4}+\delta},

        where :math:`inp` is the dual input tensors and :math:`\delta` is a small positive constant, which is needed
        to avoid division by zero in case statistical variance is close to zero. :math:`\text{Re(...)}` and
        :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression inside
        the parentheses.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the dual domain
                and has a real and a dual parts.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """
        u_r, u_d = get_real_and_dual(u)
        dual_half = u_d.abs() / 2
        eps = 1e-7
        sqrt = u_r ** 2 + dual_half ** 2 + eps
        sqrt = ops.sqrt(sqrt)
        out = dual_half + sqrt
        return out

    def get_square_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates element-wise squared norm of dual elements of an input tensor.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero. The function implements the operation as:

        .. math::
            \text{out} = \left(\left|\frac{Du(inp)}{2}\right|+\sqrt{Re(inp)^2+\frac{Du(inp)^2}{4}+\delta}\right)^2,

        where :math:`inp` is the dual input tensors, :math:`\text{Re(...)}` and :math:`\text{Du(...)}`
        are respectively real and dual parts of the dual-valued expression inside the parentheses.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the dual domain
                and has a real and a dual parts.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """
        norm = self.get_norm(u)
        out = norm ** 2
        return out
