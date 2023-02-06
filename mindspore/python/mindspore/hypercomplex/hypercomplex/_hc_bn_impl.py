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
"""Hypercomplex BatchNorm Implementation"""
import numbers
from typing import Union, Tuple
from abc import abstractmethod
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor


class _BatchNormImpl(nn.Cell):
    r"""
    The interface of the implementor part of batch normalization layer on the second-order hypercomplex numbers.

    Defines the API for getting the norm of hypercomplex number, applying scaling and shift to a hypercomplex tensor,
    and updating the running mean and variance, which are used during inference. The API is used by the 'BatchNorm'
    class, and it must be implemented separately for every hypercomplex algebra:

    Args:
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc.
        num_features (int): The number of features in the input space.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number],
                 beta_init: Union[Tensor, str, Initializer, numbers.Number],
                 num_features: int) -> None:
        super(_BatchNormImpl, self).__init__()

    @abstractmethod
    def get_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates norm of a hypercomplex elements of an input tensor.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the hypercomplex
                domain and has a real and a hypercomplex parts.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """

    @abstractmethod
    def get_square_norm(self, u: Tensor) -> Tensor:
        r"""
        Calculates element-wise squared norm of hypercomplex elements of an input tensor.

        Norm is a non-negative real number that is a characteristic of 'magnitude' of that number, i.e. how far away it
        is from zero.

        Args:
            u (Tensor): Tensor of shape (2, *, ..., *). '2' denotes that the input tensor belongs to the hypercomplex
                domain and has a real and a hypercomplex parts.

        Returns:
            Tensor of shape (*, ..., *). The count and size of dimensions of the output tensor are the same ones as in
            the input tensor, but without the very first dimension because the output tensor is real-valued.
        """

    @abstractmethod
    def scale_and_shift(self,
                        u_x: Tensor,
                        u_y: Tensor,
                        scale_x: Tensor,
                        scale_y: Tensor,
                        shift_x: Tensor,
                        shift_y: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Applies hypercomplex scaling and shift to an input tensor.

        This function implements the operation as:

        .. math::
            \text{out} = \text{mul}(\text{inp}, \text{scale}) + \text{shift},

        where :math:`inp` is the hypercomplex input tensors, :math:`\text{mul}` is the channel-wise scaling operation,
        which depends on the type of the number system and provided by subclassess, :math:`\text{scale}` is
        a hypercomplex scaling vector with the same data type as the :math:`inp` created by the layer, and
        :math:`\text{shift}` is a hypercomplex bias vector with the same data type as the :math:`inp` created by
        the layer.

        Args:
            u_x (Tensor): A tensor of shape (C,), which represents the real part of the normalized inputs.
            u_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the normalized inputs.
            scale_x (Tensor): A tensor of shape (C,), which represents the real part of the scaling vector.
            scale_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the scaling vector.
            shift_x (Tensor): A tensor of shape (C,), which represents the real part of the bias vector.
            shift_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the bias vector.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the hypercomplex parts of rescaled and
            recentered inputs.
        """

    @abstractmethod
    def calculate_bn(self,
                     u_centered_x: Tensor,
                     u_centered_y: Tensor,
                     sigma: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Given a hypercomplex centered input tensor and the standard deviation of its elements, computes the
        corresponding rescaled and recentered tensor with normalized variance.

        This function implements the operation as:

        .. math::
            \text{out} = \text{mul}(\text{inp}, \frac{\text{scale}}{\sigma}) + \text{shift},

        where :math:`inp` is the hypercomplex input tensors centered over spatial and mini-batch dimensions,
        :math:`\sigma` is standard deviation of the input tensors over the same dimensions, :math:`\text{mul}` is a
        channel-wise scaling operation, which depends on the type of the number system and provided by subclassess,
        :math:`\text{scale}` is a hypercomplex scaling vector with the same data type as the :math:`inp` created
        by the layer, and :math:`\text{shift}` is a hypercomplex bias vector with the same data type as the
        :math:`inp` created by the layer.

        Args:
            u_centered_x (Tensor): A tensor of shape (C,), which represents the real part of the centered inputs.
            u_centered_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the
                centered inputs.
            sigma (Tensor): A tensor of shape (C,), which represents the statistical standard deviation of the inputs.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the hypercomplex parts of rescaled and
            recentered normalized inputs.
        """

    @abstractmethod
    def calculate_infer_bn(self,
                           moving_mean_x: Tensor,
                           moving_mean_y: Tensor,
                           moving_sigma: Tensor,
                           u_x: Tensor,
                           u_y: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Given a hypercomplex input tensor, computes the corresponding rescaled and recentered normalized tensor.

        This function is supposed to be used during inference. The mean and standard deviation are accumulated during
        the training phase. The function implements the operation as:

        .. math::
            \text{out} = \text{mul}(\text{inp}, \frac{\text{scale}}{\sigma})
            + \left(\text{mul}(-\mathrm{E}[inp], \frac{\text{scale}}{\sigma})+\text{shift}\right),

        where :math:`inp` is the hypercomplex input tensors, :math:`\sigma` is the accumulated standard deviation of
        the input tensors over spatial and mini-batch dimensions, :math:`\mathrm{E}[inp]` is the accumulated arithmetic
        mean of the input tensor over the same dimensions,:math:`\text{mul}` is a channel-wise scaling operation, which
        depends on the type of the number system and provided by subclassess, :math:`\text{scale}` is a hypercomplex
        scaling vector with the same data type as the :math:`inp` created by the layer, and :math:`\text{shift}` is a
        hypercomplex bias vector with the same data type as the :math:`inp` created by the layer.

        Args:
            moving_mean_x (Tensor): A tensor of shape (C,), which represents the real part of the accumulated
                arithmetic mean of inputs.
            moving_mean_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the accumulated
                arithmetic mean of inputs.
            moving_sigma (Tensor): A tensor of shape (C,), which represents the accumulated statistical standard
                deviation of inputs.
            u_x (Tensor): A tensor of shape (C,), which represents the real part of the input tensor.
            u_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the input tensor.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the hypercomplex parts of normalized,
            rescaled and recentered inputs.
        """


class _BaseBatchNormImpl(_BatchNormImpl):
    r"""
    The base implementor part of the batch normalization layer for all the hypercomplex numbers of the second order.

    Contains initialization and processing logic, which are shared by all specific implementations of the
    'BatchNormImpl' interface for dual, double, and complex numbers.

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

    def __init__(self,
                 affine: bool,
                 use_batch_statistics: bool,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number],
                 beta_init: Union[Tensor, str, Initializer, numbers.Number],
                 num_features: int) -> None:
        super(_BaseBatchNormImpl, self).__init__(gamma_init,
                                                 beta_init,
                                                 num_features)
        self.scale_x = Parameter(initializer(gamma_init, num_features), name="scale_x", requires_grad=affine)
        self.scale_y = Parameter(initializer(gamma_init, num_features), name="scale_y", requires_grad=affine)
        self.shift_x = Parameter(initializer(beta_init, num_features), name="shift_x", requires_grad=affine)
        self.shift_y = Parameter(initializer(beta_init, num_features), name="shift_y", requires_grad=affine)

    def calculate_infer_bn(self,
                           moving_mean_x: Tensor,
                           moving_mean_y: Tensor,
                           moving_sigma: Tensor,
                           u_x: Tensor,
                           u_y: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Given a hypercomplex input tensor, computes the corresponding rescaled and recentered normalized tensor.

        This function is supposed to be used during inference. The mean and standard deviation are accumulated during
        the training phase. The function implements the operation as:

        .. math::
            \text{out} = \text{mul}(\text{inp}, \frac{\text{scale}}{\sigma})
            + \left(\text{mul}(-\mathrm{E}[inp], \frac{\text{scale}}{\sigma})+\text{shift}\right),

        where :math:`inp` is the hypercomplex input tensors, :math:`\sigma` is the accumulated standard deviation of
        the input tensors over spatial and mini-batch dimensions, :math:`\mathrm{E}[inp]` is the accumulated arithmetic
        mean of the input tensor over the same dimensions,:math:`\text{mul}` is a channel-wise scaling operation, which
        depends on the type of the number system and provided by subclassess, :math:`\text{scale}` is a hypercomplex
        scaling vector with the same data type as the :math:`inp` created by the layer, and :math:`\text{shift}` is a
        hypercomplex bias vector with the same data type as the :math:`inp` created by the layer.

        Args:
            moving_mean_x (Tensor): A tensor of shape (C,), which represents the real part of the accumulated
                arithmetic mean of inputs.
            moving_mean_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the accumulated
                arithmetic mean of inputs.
            moving_sigma (Tensor): A tensor of shape (C,), which represents the accumulated statistical standard
                deviation of inputs.
            u_x (Tensor): A tensor of shape (C,), which represents the real part of the input tensor.
            u_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the input tensor.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the hypercomplex parts of normalized,
            rescaled and recentered inputs.
        """
        fused_scale_x = self.scale_x / moving_sigma
        fused_scale_y = self.scale_y / moving_sigma
        neg_mean_x = (-1) * moving_mean_x
        neg_mean_y = (-1) * moving_mean_y
        fused_shift_x, fused_shift_y = self.scale_and_shift(neg_mean_x,
                                                            neg_mean_y,
                                                            fused_scale_x,
                                                            fused_scale_y,
                                                            self.shift_x,
                                                            self.shift_y)
        out_x, out_y = self.scale_and_shift(u_x,
                                            u_y,
                                            fused_scale_x,
                                            fused_scale_y,
                                            fused_shift_x,
                                            fused_shift_y)
        return out_x, out_y

    def calculate_bn(self,
                     u_centered_x: Tensor,
                     u_centered_y: Tensor,
                     sigma: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Given a hypercomplex centered input tensor and the standard deviation of its elements, computes the
        corresponding rescaled and recentered tensor with normalized variance.

        This function implements the operation as:

        .. math::
            \text{out} = \text{mul}(\text{inp}, \frac{\text{scale}}{\sigma}) + \text{shift},

        where :math:`inp` is the hypercomplex input tensors centered over spatial and mini-batch dimensions,
        :math:`\sigma` is standard deviation of the input tensors over the same dimensions, :math:`\text{mul}` is a
        channel-wise scaling operation, which depends on the type of the number system and provided by subclassess,
        :math:`\text{scale}` is a hypercomplex scaling vector with the same data type as the :math:`inp` created
        by the layer, and :math:`\text{shift}` is a hypercomplex bias vector with the same data type as the
        :math:`inp` created by the layer.

        Args:
            u_centered_x (Tensor): A tensor of shape (C,), which represents the real part of the centered inputs.
            u_centered_y (Tensor): A tensor of shape (C,), which represents the hypercomplex part of the
                centered inputs.
            sigma (Tensor): A tensor of shape (C,), which represents the statistical standard deviation of the inputs.

        Returns:
            Tuple of two tensors of shape (C,), which contains the real and the hypercomplex parts of rescaled and
            recentered normalized inputs.
        """
        scale_x = self.scale_x / sigma
        scale_y = self.scale_y / sigma
        out_x, out_y = self.scale_and_shift(u_centered_x,
                                            u_centered_y,
                                            scale_x,
                                            scale_y,
                                            self.shift_x,
                                            self.shift_y)
        return out_x, out_y

    @abstractmethod
    def get_norm(self, u: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_square_norm(self, u: Tensor) -> Tensor:
        pass

    @abstractmethod
    def scale_and_shift(self,
                        u_x: Tensor,
                        u_y: Tensor,
                        scale_x: Tensor,
                        scale_y: Tensor,
                        shift_x: Tensor,
                        shift_y: Tensor) -> Tuple[Tensor, Tensor]:
        pass
