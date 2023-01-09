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
"""Hypercomplex batchnorm"""
import numbers
from typing import TypeVar, Type, Union, Any
from abc import abstractmethod

import numpy as np
import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as P
from mindspore._checkparam import Validator as validator
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.hypercomplex.hypercomplex._hc_bn_impl import _BatchNormImpl as BatchNormImpl
from mindspore.hypercomplex.utils import get_x_and_y, to_2channel

TBatchNormImpl = TypeVar('TBatchNormImpl', bound=BatchNormImpl)


class _BatchNorm(nn.Cell):
    r"""
    The base class of the abstract part of Batch Normalization layer over a second-order hypercomplex input
    of some number of dimensions.

    This layer applies Batch Normalization over a hypercomplex input to reduce internal covariate shift.
    Batch Normalization is widely used in convolutional networks. It rescales and recenters the feature using
    a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        out = \text{linear}(\frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}, \gamma) + \beta,
        \end{align}

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{linear}` is the linear transformation operation,
    which depends on the type of the number system and provided by the implementor part of the batch normalization
    layer, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor over the spatial and mini-batch
    dimensions, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over the same dimensions,
    :math:`\gamma` and :math:`\beta` are hypercomplex learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero.

    This is not a self-sufficient class. In order to construct a fully connected layer, one should instantiate a child
    class and an implementor class, which acts like a bridge pattern and determines the exact set of hypercomplex
    numbers. That implies the rules of multiplication and therefore affects how a linear transformation works.

    Args:
        bn_impl(BatchNormImpl): The implementor object of the batch normalization layer. Essentially, the concrete
            class name of this argument defines the algebra that the batch normalization layer will operate on.
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):
            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Default: 'NCHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, *, ..., *)` if data_format is 'NCHW', or
          :math:`(2, N, *, ..., *, C)` if data_format is 'NHWC', with float16 or float32 data type. '2' denotes that
          the input tensor belongs to the hypercomplex domain and has got a real and a hypercomplex parts. Or,
          :math:`(N, C, *, ..., *)` if data_format is 'NCHW', or :math:`(N, *, ..., *, C)` if data_format is 'NHWC',
          with complex64 data type. The `num_features` in `Args` has to be equal to :math:`C` in `Inputs`.
          The count of dimensions denoted by '*' must be equal to the number of spatial dimensions.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same data type and shape as :math:`inp`:
        :math:`(2, N, C, *, ..., *)` if data_format is 'NCHW', or :math:`(2, N, *, ..., *, C)` if data_format is 'NHWC',
        with float16 or float32 data type. Or, :math:`(N, C, *, ..., *)` if data_format is 'NCHW', or
        :math:`(N, *, ..., *, C)` if data_format is 'NHWC', with complex64 data type.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 bn_impl: Type[TBatchNormImpl],
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 beta_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_mean_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_var_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 use_batch_statistics: bool = None,
                 data_format: str = 'NCHW') -> None:
        """Initialize _BatchNorm."""
        super(_BatchNorm, self).__init__()
        validator.check_value_type('num_features', num_features, [int], self.cls_name)
        if num_features < 1:
            raise ValueError(f"For '{self.cls_name}', the 'num_features' must be at least 1, but got {num_features}.")

        if momentum < 0 or momentum > 1:
            raise ValueError(f"For '{self.cls_name}', the 'momentum' must be a number in range [0, 1], "
                             f"but got {momentum}.")
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.cls_name}', the 'NHWC' format only support in GPU target, but got device "
                             f"target {context.get_context('device_target')}.")
        self.use_batch_statistics = use_batch_statistics
        if self.use_batch_statistics is not None and not isinstance(self.use_batch_statistics, bool):
            raise ValueError(f"For '{self.cls_name}', the 'use_batch_statistics' must be a boolean value or None,"
                             f" but got {use_batch_statistics}.")
        self.num_features = num_features
        self.eps = eps
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.affine = affine

        self.bn_impl = bn_impl(affine, use_batch_statistics, gamma_init, beta_init, num_features)

        self.moving_mean_x = Parameter(
            initializer(moving_mean_init, (num_features)), name="mean_x", requires_grad=False
        )
        self.moving_mean_y = Parameter(
            initializer(moving_mean_init, (num_features)), name="mean_y", requires_grad=False
        )
        self.moving_sigma2 = Parameter(
            initializer(moving_var_init, num_features), name="sigma2", requires_grad=False
        )

        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")

        self._target = context.get_context("device_target")
        self.is_graph_mode = context.get_context("mode") == context.GRAPH_MODE
        self.momentum = 1.0 - momentum

        self.reduce_mean_op1 = P.ReduceMean(keep_dims=True)
        self.reduce_mean_op2 = P.ReduceMean(keep_dims=False)

        self.features_dim = data_format.lower().find('c')
        self.get_dtype = P.DType()
        self.get_shape = P.Shape()

    def construct(self, u: Tensor) -> Tensor:
        """construct"""
        u_dtype = self.get_dtype(u)
        u_shape = self.get_shape(u)
        self._check_input_dim(u_shape, u_dtype)
        if u_dtype == mindspore.complex64:
            hc_axis = None
            feature_axis = self.features_dim
        else:
            hc_axis = 0
            feature_axis = self.features_dim + 1

        if self.training or not self.use_batch_statistics:
            ndim = u.ndim
            hc_axis = hc_axis
            feature_axis = feature_axis
            sh = np.arange(ndim)
            sh = sh[sh != hc_axis]
            sh = sh[sh != feature_axis]
            if hc_axis is None:
                u_x, u_y = get_x_and_y(u)
                mu_x = self.reduce_mean_op1(u_x, sh.tolist())
                mu_y = self.reduce_mean_op1(u_y, sh.tolist())
                mu = to_2channel(mu_x, mu_y, mindspore.complex64)
            else:
                mu = self.reduce_mean_op1(u, sh.tolist())

            u_centered = u - mu
            norma2 = self.bn_impl.get_square_norm(u_centered)
            norma_feature_axis = feature_axis if hc_axis is None or feature_axis < hc_axis else feature_axis - 1
            ndim = norma2.ndim
            mean_dims = np.arange(ndim)
            mean_dims = mean_dims[mean_dims != norma_feature_axis]
            sigma2 = self.reduce_mean_op2(norma2, mean_dims.tolist()) + self.eps
            result = self._calculate_bn(u_centered, sigma2, feature_axis)

            if self.use_batch_statistics:
                momentum = self.momentum
                mu = mu.squeeze()
                mu_x, mu_y = get_x_and_y(mu)
                momentum_suppl = 1 - momentum
                self.moving_mean_x *= momentum_suppl
                self.moving_mean_x += mu_x * momentum
                self.moving_mean_y *= momentum_suppl
                self.moving_mean_y += mu_y * momentum
                self.moving_sigma2 *= momentum_suppl
                self.moving_sigma2 += sigma2 * momentum
        elif self.affine:
            result = self._calculate_infer_bn(u, axis=feature_axis)
        else:
            broadcast_mu_shape = [1] * u.ndim
            broadcast_mu_shape[feature_axis] = u_shape[feature_axis]
            if hc_axis is not None:
                broadcast_mu_shape[hc_axis] = 2
            moving_mean = to_2channel(self.moving_mean_x, self.moving_mean_y, u.dtype)
            moving_mean = moving_mean.reshape(tuple(broadcast_mu_shape))
            inference_centered = u - moving_mean
            result = self._calculate_bn(inference_centered, self.moving_sigma2, feature_axis)
        return result

    def _calculate_bn(self,
                      u_centered: Tensor,
                      sigma2: Tensor,
                      axis: int) -> Tensor:
        """_calculate_bn, implement the abstract function"""
        sigma = P.sqrt(sigma2)
        ndim = u_centered.ndim
        u_shape = list(np.arange(ndim))
        u_shape[ndim - 1] = axis
        u_shape[axis] = ndim - 1
        u_shape = tuple(int(i) for i in u_shape)
        out = P.transpose(u_centered, u_shape)
        if self.affine:
            out_x, out_y = get_x_and_y(out)
            out_x, out_y = self.bn_impl.calculate_bn(out_x, out_y, sigma)
            out = to_2channel(out_x, out_y, self.get_dtype(u_centered))
        else:
            out = out / sigma
        out = P.transpose(out, u_shape)
        return out

    def _calculate_infer_bn(self,
                            u: Tensor,
                            axis: int) -> Tensor:
        """_calculate_infer_bn, implement the abstract function"""
        ndim = u.ndim
        shape = list(np.arange(ndim))
        shape[ndim-1] = axis
        shape[axis] = ndim - 1
        shape = tuple(int(i) for i in shape)

        out = P.transpose(u, shape)
        out_x, out_y = get_x_and_y(out)
        out_x, out_y = self.bn_impl.calculate_infer_bn(self.moving_mean_x,
                                                       self.moving_mean_y,
                                                       P.sqrt(self.moving_sigma2),
                                                       out_x,
                                                       out_y)
        out = to_2channel(out_x, out_y, dtype=u.dtype)
        out = P.transpose(out, shape)
        return out

    @abstractmethod
    def _check_input_dim(self, shape: tuple, dtype: Any):
        raise NotImplementedError


class BatchNorm1d(_BatchNorm):
    r"""
    The class of the abstract part of Batch Normalization layer over a second-order hypercomplex input
    of four dimensions including one spatial dimension, or three dimensions.

    This layer applies Batch Normalization over a hypercomplex input of 'NCW' data format in order to reduce
    internal covariate shift. Batch Normalization is widely used in convolutional networks. It rescales and recenters
    the feature using a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        out = \text{linear}(\frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}, \gamma) + \beta,
        \end{align}

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{linear}` is the linear transformation operation,
    which depends on the type of the number system and provided by the implementor part of the batch normalization
    layer, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor over the spatial and mini-batch
    dimensions, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over the same dimensions,
    :math:`\gamma` and :math:`\beta` are hypercomplex learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero.

    This is not a self-sufficient class. In order to construct a fully connected layer, one should instantiate this
    class and an implementor class, which acts like a bridge pattern and determines the exact set of hypercomplex
    numbers. That implies the rules of multiplication and therefore affects how a linear transformation works.

    Args:
        bn_impl(BatchNormImpl): The implementor object of the batch normalization layer. Essentially, the concrete
            class name of this argument defines the algebra that the batch normalization layer will operate on.
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):

            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, W)` or :math:`(2, N, C)`, with float16 or float32 data
          type, or :math:`(N, C, W)` or :math:`(N, C)`, with complex64 data type. In the former case '2' denotes that
          the input tensor belongs to the hypercomplex domain and has got a real and a hypercomplex parts.
          The `num_features` in `Args` has to be equal to :math:`C` in `inp`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same data type and shape as :math:`inp`:
        :math:`(2, N, C, W)` or :math:`(2, N, C)`, with float16 or float32 data type, or :math:`(N, C, W)` or
        :math:`(N, C)`, with complex64 data type.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: if 'inp' is not a Tensor of 3 or 4 dimensions with float16 or float32 data type, and not a Tensor
            of 2 or 3 dimensions with complex64 data type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 bn_impl: Type[TBatchNormImpl],
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 beta_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_mean_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_var_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 use_batch_statistics: bool = None) -> None:
        """Initialize _BatchNorm."""

        super(BatchNorm1d, self).__init__(bn_impl,
                                          num_features,
                                          eps,
                                          momentum,
                                          affine,
                                          gamma_init,
                                          beta_init,
                                          moving_mean_init,
                                          moving_var_init,
                                          use_batch_statistics)

    def _check_input_dim(self, shape: tuple, dtype: Any):
        dim = len(shape)
        if dtype in [mindspore.float16, mindspore.float32]:
            if dim not in (4, 3):
                raise ValueError(f"For '{self.cls_name}', the in_shape must have 3-4 dims, but got {dim}.")
        elif dtype == mindspore.complex64:
            if dim not in (3, 2):
                raise ValueError(f"For '{self.cls_name}', the in_shape must have 2-3 dims, but got {dim}.")
        else:
            raise TypeError(f"Only float16, float32 and complex64 data types are supported, but got {dtype}.")
        return None


class BatchNorm2d(_BatchNorm):
    r"""
    The class of the abstract part of Batch Normalization layer over a second-order hypercomplex input
    of five dimensions, including two spatial dimensions.

    This layer applies Batch Normalization over a hypercomplex input to reduce internal covariate shift.
    Batch Normalization is widely used in convolutional networks. It rescales and recenters the feature
    using a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        y = \text{linear}(\frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}, \gamma) + \beta,
        \end{align}

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{linear}` is the linear transformation operation,
    which depends on the type of the number system and provided by the implementor part of the batch normalization
    layer, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor over the spatial and mini-batch
    dimensions, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over the same dimensions,
    :math:`\gamma` and :math:`\beta` are hypercomplex learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero.

    This is not a self-sufficient class. In order to construct a fully connected layer, one should instantiate this
    class and an implementor class, which acts like a bridge pattern and determines the exact set of hypercomplex
    numbers. That implies the rules of multiplication and therefore affects how a linear transformation works.

    Args:
        bn_impl(BatchNormImpl): The implementor object of the batch normalization layer. Essentially, the concrete
            class name of this argument defines the algebra that the batch normalization layer will operate on.
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):

            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Default: 'NCHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, H, W)` if data_format is 'NCHW', or
          :math:`(2, N, H, W, C)` if data_format is 'NHWC', with float16 or float32 data type. '2' denotes that the
          input tensor belongs to the hypercomplex domain and has got a real and a hypercomplex parts. Or,
          :math:`(N, C, H, W)` if data_format is 'NCHW', or :math:`(N, H, W, C)` if data_format is 'NHWC', with
          complex64 data type. The `num_features` in `Args` has to be equal to :math:`C` in `Inputs`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same data type and shape as :math:`inp`:
        :math:`(2, N, C, H, W)` if data_format is 'NCHW', or :math:`(2, N, H, W, C)` if data_format is 'NHWC', with
        float16 or float32 data type. Or, :math:`(N, C, H, W)` if data_format is 'NCHW', or :math:`(N, H, W, C)` if
        data_format is 'NHWC', with complex64 data type.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is neither 'NHWC' not 'NCHW'.
        ValueError: if 'inp' is not a Tensor of 5 dimensions with float16 or float32 data type, and not a Tensor of 4
            dimensions with complex64 data type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def _check_input_dim(self, shape: tuple, dtype: Any):
        dim = len(shape)
        if dtype in [mindspore.float16, mindspore.float32]:
            if dim != 5:
                raise ValueError(f"For '{self.cls_name}', the in_shape must have 5 dims, but got {dim}.")
        elif dtype == mindspore.complex64:
            if dim != 4:
                raise ValueError(f"For '{self.cls_name}', the in_shape must have 4 dims, but got {dim}.")
        else:
            raise TypeError(f"Only float16, float32 and complex64 data types are supported, but got {dtype}.")
        return None


class BatchNorm3d(nn.Cell):
    r"""
    The class of the abstract part of Batch Normalization layer over a second-order hypercomplex input
    of six dimensions, including three spatial dimensions.

    This layer applies Batch Normalization over a hypercomplex input to reduce internal covariate shift.
    Batch Normalization is widely used in convolutional networks. It rescales and recenters the feature
    using a mini-batch of data and the learned parameters which can be described by the following formula:

    .. math::
        \begin{align}
        \mathrm{Var}[inp] = \mathrm{E}[\| inp_i - \mathrm{E}[inp] \|^2]\\
        y = \text{linear}(\frac{inp - \mathrm{E}[inp]}{\sqrt{\mathrm{Var}[inp] + \delta}}, \gamma) + \beta,
        \end{align}

    where :math:`inp` is the hypercomplex input tensors, :math:`\text{linear}` is the linear transformation operation,
    which depends on the type of the number system and provided by the implementor part of the batch normalization
    layer, :math:`\mathrm{E}[inp]` is the arithmetic mean of the input tensor over the spatial and mini-batch
    dimensions, :math:`\mathrm{Var}[inp]` is the statistical variance of the input tensor over the same dimensions,
    :math:`\gamma` and :math:`\beta` are hypercomplex learnable parameters representing the scale and shift coefficients
    respectively, and :math:`\delta` is a small positive constant, which is needed to avoid division by zero in case
    statistical variance is close to zero.

    This is not a self-sufficient class. In order to construct a fully connected layer, one should instantiate this
    class and an implementor class, which acts like a bridge pattern and determines the exact set of hypercomplex
    numbers. That implies the rules of multiplication and therefore affects how a linear transformation works.

    Args:
        bn_impl(BatchNormImpl): The implementor object of the batch normalization layer. Essentially, the concrete
            class name of this argument defines the algebra that the batch normalization layer will operate on.
        num_features (int): The number of features in the input space.
        eps (float): A small positive threshold, which is needed to avoid division by zero. Default: :math:`10^{-5}`
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        momentum (float): A floating hyperparameter of the momentum for the running_mean and running_var computation.
            Default: 0.9.
        affine (bool): A bool value. When set to True, gamma and beta can be learned. Default: True.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_mean_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving mean.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'zeros'.
        moving_var_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the moving variance.
            The values of str refer to the function `initializer` including 'zeros', 'ones', etc. Default: 'ones'.
        use_batch_statistics (bool):

            - If True, use the mean value and variance value of current batch data and track running mean
              and running variance.
            - If False, use the mean value and variance value of specified value, and not track statistical value.
            - If None, the use_batch_statistics is automatically set to True or False according to the training
              and evaluation mode. During training, the parameter is set to True, and during evaluation, the
              parameter is set to False. Default: None.
        data_format (str): The optional value for data format. Only 'NCDHW' format is supported as of now.
            Default: 'NCDHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, D, H, W)`, with float16 or float32 data type, or
          :math:`(N, C, D, H, W)`, with complex64 data type. In the former case '2' denotes that the input tensor
          belongs to the hypercomplex domain and has got a real and a hypercomplex parts. The `num_features` in `Args`
          has to be equal to :math:`C` in `Inputs`.

    Outputs:
        Tensor, the normalized, scaled, offset tensor of the same data type and shape as :math:`inp`:
        :math:`(2, N, C, D, H, W)`, with float16 and float32 data type, or :math:`(N, C, D, H, W)`, with
        complex64 data type.

    Raises:
        TypeError: If `num_features` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `num_features` is less than 1.
        ValueError: If `momentum` is not in range [0, 1].
        ValueError: If `data_format` is not 'NCDHW'.
        ValueError: if 'inp' is not a Tensor of 6 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 bn_impl: Type[TBatchNormImpl],
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.9,
                 affine: bool = True,
                 gamma_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 beta_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_mean_init: Union[Tensor, str, Initializer, numbers.Number] = 'zeros',
                 moving_var_init: Union[Tensor, str, Initializer, numbers.Number] = 'ones',
                 use_batch_statistics: bool = None,
                 data_format: str = 'NCDHW') -> None:
        """Initialize _BatchNorm."""
        super(BatchNorm3d, self).__init__()
        self.format = validator.check_string(data_format, ['NCDHW'], 'format', self.cls_name)
        self.reshape = P.Reshape()
        self.bn2d = BatchNorm2d(bn_impl=bn_impl,
                                num_features=num_features,
                                eps=eps,
                                momentum=momentum,
                                affine=affine,
                                gamma_init=gamma_init,
                                beta_init=beta_init,
                                moving_mean_init=moving_mean_init,
                                moving_var_init=moving_var_init,
                                use_batch_statistics=use_batch_statistics,
                                data_format="NCHW")

    def construct(self, u: Tensor) -> Tensor:
        '''construct'''
        u_shape = F.shape(u)
        self._check_3d_shape(u_shape, F.dtype(u))
        reshape = list(u_shape)
        reshape[-3] *= reshape[-2]
        reshape = tuple(int(i) for i in reshape[:-2] + reshape[-1:])
        u = self.reshape(u, tuple(reshape))
        out = self.bn2d(u)
        out = self.reshape(out, u_shape)
        return out

    def _check_3d_shape(self, input_shape, dtype: Any) -> None:
        '''_check_3d_shape'''
        dim = len(input_shape)
        if dtype in [mindspore.float16, mindspore.float32]:
            if dim != 6:
                msg_prefix = f"For '{self.cls_name}', the" if self.cls_name else "The"
                raise ValueError(f"{msg_prefix} input_shape must be 6-dimensional, but got the length of input_shape: "
                                 f"{len(dim)}.")
        elif dtype == mindspore.complex64:
            if dim != 5:
                msg_prefix = f"For '{self.cls_name}', the" if self.cls_name else "The"
                raise ValueError(f"{msg_prefix} input_shape must be 5-dimensional, but got the length of input_shape: "
                                 f"{len(dim)}.")
        else:
            raise TypeError(f"Only float16, float32 and complex64 data types are supported, but got {dtype}.")
        return None
