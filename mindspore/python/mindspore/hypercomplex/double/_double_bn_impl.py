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
"""Double BatchNorm implementation"""
from typing import Tuple

import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.hypercomplex.hypercomplex._hc_bn_impl import _BaseBatchNormImpl as HCBatchNormImpl
from mindspore.hypercomplex.utils import get_x_and_y

get_real_and_double = get_x_and_y
get_u1_and_u2 = get_x_and_y


class _BatchNormImpl(HCBatchNormImpl):
    r"""
    The implementor class of the Batch Normalization layer for double numbers in regular representation.

    Implements the functionality specific to double numbers and needed by the 'BatchNorm' class. This includes:
    getting the norm of double number, applying scaling and shift to a double-valued tensor, and updating the running
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
        Applies double scaling and shift to an input tensor in regular representation.

        This function implements the operation as:

        .. math::
            \begin{align}
            \text{Re(out)} = \text{Re(inp)} * \text{Re(scale)} + \text{Db(inp)} * \text{Db(scale)} + \text{Re(shift)}\\
            \text{Db(out)} = \text{Re(inp)} * \text{Db(scale)} + \text{Db(inp)} * \text{Re(scale)} + \text{Db(shift)},
            \end{align}

        where :math:`inp` is the double input tensors, :math:`scale` and :math:`shift` are double parameters
        representing the scaling and shift coefficients respectively. :math:`\text{Re(...)}` and :math:`\text{Db(...)}`
        are respectively real and double parts of the double-valued expression inside the parentheses.

        Args:
            u_x (Tensor): A tensor of shape (C,), which represents the real part of the normalized inputs.
            u_y (Tensor): A tensor of shape (C,), which represents the double part of the normalized inputs.
            scale_x (Tensor): A tensor of shape (C,), which represents the real part of the scaling vector.
            scale_y (Tensor): A tensor of shape (C,), which represents the double part of the scaling vector.
            shift_x (Tensor): A tensor of shape (C,), which represents the real part of the bias vector.
            shift_y (Tensor): A tensor of shape (C,), which represents the double part of the bias vector.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the double parts of rescaled and
            recentered inputs.
        """
        out_x = u_x * scale_x + u_y * scale_y + shift_x
        out_y = u_x * scale_y + u_y * scale_x + shift_y
        return out_x, out_y

    def get_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates norm of double elements of an input tensor in regular representation.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero. The function implements the operation as:

        .. math::
            \text{out} = |Re(inp)| + |Db(inp)|,

        where :math:`inp` is the double input tensors, :math:`\text{Re(...)}` and :math:`\text{Db(...)}`
        are respectively real and double parts of the double-valued expression inside the parentheses.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the double domain
                and has two components.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """
        u_r, u_d = get_real_and_double(u)
        out = ops.abs(u_r) + ops.abs(u_d)
        return out

    def get_square_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates element-wise squared norm of double elements of an input tensor in regular representation.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero. The function implements the operation as:

        .. math::
            \text{out} = \left(|Re(inp)| + |Db(inp)|\right)^2,

        where :math:`inp` is the double input tensors, :math:`\text{Re(...)}` and :math:`\text{Du(...)}`
        are respectively real and double parts of the double-valued expression inside the parentheses.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the double domain
                and has a real and a double parts.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """
        norm = self.get_norm(u)
        out = norm ** 2
        return out


class _J1J2BatchNormImpl(HCBatchNormImpl):
    r"""
    The implementor class of the Batch Normalization layer for double numbers in diagonal representation.

    Implements the functionality specific to double numbers and needed by the 'BatchNorm' class. This includes:
    getting the norm of double number, applying scaling and shift to a double-valued tensor, and updating the running
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
        Applies double scaling and shift to an input tensor in diagonal representation.

        This function implements the operation as:

        .. math::
            \begin{align}
            \text{Re(out)} = \text{X(inp)} * \text{Y(scale)} + \text{X(shift)}\\
            \text{Db(out)} = \text{X(inp)} * \text{Y(scale)} + \text{Y(inp)},
            \end{align}

        where :math:`inp` is the double input tensors in diagonal form, :math:`scale` and :math:`shift` are
        double parameters representing  the scaling and shift coefficients respectively. :math:`\text{X(...)}`
        and :math:`\text{Y(...)}` are respectively the first and the second components of the double-valued
        expression inside the parentheses.

        Args:
            u_x (Tensor): A tensor of shape (C,), which represents the first part of the normalized inputs.
            u_y (Tensor): A tensor of shape (C,), which represents the second part of the normalized inputs.
            scale_x (Tensor): A tensor of shape (C,), which represents the first part of the scaling vector.
            scale_y (Tensor): A tensor of shape (C,), which represents the second part of the scaling vector.
            shift_x (Tensor): A tensor of shape (C,), which represents the first part of the bias vector.
            shift_y (Tensor): A tensor of shape (C,), which represents the second part of the bias vector.

        Returns:
            Tuple of two tensors of shape (C,), which contains the first and the second parts of rescaled and
            recentered inputs in the diagonal representation.
        """
        out_x = u_x * scale_x + shift_x
        out_y = u_y * scale_y + shift_y
        return out_x, out_y

    def get_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates norm of double elements of an input tensor in diagonal representation.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero. The function implements the operation as:

        .. math::
            \text{out} = \text{max}(|X(inp)|, |Y(inp)|),

        where :math:`inp` is the double input tensors in diagonal form, :math:`\text{max}` is the maximum value of its
        arguments. :math:`\text{X(...)}` and :math:`\text{Y(...)}` are respectively the first and the second components
        of the double-valued expression inside the parentheses.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the double domain
                and has two components.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """
        u_1, u_2 = get_u1_and_u2(u)
        out = ops.maximum(ops.abs(u_1), ops.abs(u_2))
        return out

    def get_square_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates element-wise squared norm of double elements of an input tensor in diagonal representation.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero. The function implements the operation as:

        .. math::
            \text{out} = \left(\text{max}(|X(inp)|, |Y(inp)|)\right)^2,

        where :math:`inp` is the double input tensors in diagonal form, :math:`\text{max}` is the maximum value of its
        arguments. :math:`\text{X(...)}` and :math:`\text{Y(...)}` are respectively the first and the second components
        of the double-valued expression inside the parentheses.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the double domain
                and has two components.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """
        norm = self.get_norm(u)
        out = norm ** 2
        return out
