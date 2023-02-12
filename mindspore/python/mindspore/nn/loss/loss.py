# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""loss"""
from __future__ import absolute_import, division
import math

import mindspore
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import log
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations.nn_ops import MultiMarginLoss as MultiMarginLossOp
from mindspore.ops.operations.nn_ops import MultilabelMarginLoss as MultilabelMarginLossOp
from mindspore.ops import functional as F
from mindspore import nn
from mindspore.ops.primitive import constexpr
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore import context


class LossBase(Cell):
    """
    Base class for other losses.

    Other losses derived from this should implement their own `construct` and use method `self.get_loss`
    to apply reduction to loss values.

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Raises:
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, reduction='mean'):
        """Initialize Loss."""
        super(LossBase, self).__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"For '{self.cls_name}', the 'reduction' must be in ['mean', 'sum', 'none'], "
                             f"but got {reduction}.")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def get_axis(self, x):
        """
        Get a range of axis for input.

        Args:
            x (Tensor): Tensor of any shape.

        Examples:
            >>> class Net(nn.LossBase):
            ...     def __init__(self, reduction='mean'):
            ...         super(Net, self).__init__(reduction)
            ...         self.abs = ops.Abs()
            ...
            ...     def construct(self, logits, labels):
            ...         x = self.abs(logits - labels)
            ...         axis = self.get_axis(x)
            ...         return axis
            >>> net = Net()
            >>> # Case 1: logits.shape = labels.shape = (3,)
            >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
            >>> labels = Tensor(np.array([1, 2, 3]), mindspore.float32)
            >>> output = net(logits, labels)
            >>> print(output)
            (0,)
            >>> # Case 2: logits.shape = labels.shape = (3, 3)
            >>> logits = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
            >>> labels = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
            >>> output = net(logits, labels)
            >>> print(output)
            (0, 1)
        """
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss.

        Args:
            x (Tensor): Tensor of shape :math:`(N, *)` where :math:`*` means, any number of
                additional dimensions.
            weights (Union[float, Tensor]): Optional `Tensor` whose rank is either 0, or the same rank as inputs,
                and must be broadcastable to inputs (i.e., all dimensions must be either `1`,
                or the same as the corresponding inputs dimension). Default: 1.0.

        Returns:
            Return the weighted loss.

        Examples:
            >>> class Net(nn.LossBase):
            ...     def __init__(self, reduction='mean'):
            ...         super(Net, self).__init__(reduction)
            ...         self.abs = ops.Abs()
            ...
            ...     def construct(self, logits, labels):
            ...         x = self.abs(logits - labels)
            ...         output = self.get_loss(x)
            ...         return output
            >>> net = Net()
            >>> # Case 1: logits.shape = labels.shape = (3,)
            >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
            >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
            >>> output = net(logits, labels)
            >>> print(output)
            0.33333334
            >>> # Case 2: logits.shape = labels.shape = (3, 3)
            >>> logits = Tensor(np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
            >>> labels = Tensor(np.array([[1, 2, 2],[1, 2, 3],[1, 2, 3]]), mindspore.float32)
            >>> output = net(logits, labels)
            >>> print(output)
            0.11111111
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        x = self.cast(x, input_dtype)
        return x

    def construct(self, logits, labels):
        raise NotImplementedError


class _Loss(LossBase):
    """
    Base class for other losses.
    """

    def __init__(self, reduction='mean'):
        """Initialize _Loss."""
        log.warning("'_Loss' is deprecated from version 1.3 and "
                    "will be removed in a future version, use 'LossBase' instead.")
        super(_Loss, self).__init__(reduction)

    def construct(self, logits, labels):
        raise NotImplementedError


@constexpr(check=False)
def _check_is_tensor(param_name, input_data, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and not isinstance(F.typeof(input_data), mstype.tensor_type):
        raise TypeError(f"For '{cls_name}', the '{param_name}' must be '{mstype.tensor_type}', "
                        f"but got '{F.typeof(input_data)}'")


class L1Loss(LossBase):
    r"""
    L1Loss is used to calculate the mean absolute error between the predicted value and the target value.

    Assuming that the :math:`x` and :math:`y` are 1-D Tensor, length :math:`N`, then calculate the loss of :math:`x` and
    :math:`y` without dimensionality reduction (the reduction parameter is set to "none"). The formula is as follows:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If `reduction` is not 'none', then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean". If `reduction` is "mean" or "sum", then output a scalar Tensor, if `reduction` is "none",
            the shape of the output Tensor is the broadcasted shape.

    Inputs:
        - **logits** (Tensor) - Predicted value, Tensor of any dimension.
        - **labels** (Tensor) - Target value, same shape as the `logits` in common cases.
          However, it supports the shape of `logits` is different from the shape of `labels`
          and they should be broadcasted to each other.

    Outputs:
        Tensor, data type is float.

    Raises:
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted to each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # Case 1: logits.shape = labels.shape = (3,)
        >>> loss = nn.L1Loss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.33333334
        >>> # Case 2: logits.shape = (3,), labels.shape = (2, 3)
        >>> loss = nn.L1Loss(reduction='none')
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [[0. 1. 2.]
         [0. 0. 1.]]
    """

    def __init__(self, reduction='mean'):
        """Initialize L1Loss."""
        super(L1Loss, self).__init__(reduction)
        self.reduction = reduction

    def construct(self, logits, labels):
        return F.l1_loss(logits, labels, self.reduction)


class MSELoss(LossBase):
    r"""
    Calculates the mean squared error between the predicted value and the label value.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with} \quad l_n = (x_n - y_n)^2.

    where :math:`N` is the batch size. If `reduction` is not 'none', then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Inputs:
        - **logits** (Tensor) - The predicted value of the input. Tensor of any dimension.
        - **labels** (Tensor) - The input label. Tensor of any dimension, same shape as the `logits` in common cases.
          However, it supports the shape of `logits` is different from the shape of `labels`
          and they should be broadcasted to each other.

    Outputs:
        Tensor, loss of type float, the shape is zero if `reduction` is 'mean' or 'sum',
        while the shape of output is the broadcasted shape if `reduction` is 'none'.

    Raises:
        ValueError: If `reduction` is not one of 'none', 'mean' or 'sum'.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # Case 1: logits.shape = labels.shape = (3,)
        >>> loss = nn.MSELoss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        1.6666667
        >>> # Case 2: logits.shape = (3,), labels.shape = (2, 3)
        >>> loss = nn.MSELoss(reduction='none')
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [[0. 1. 4.]
         [0. 0. 1.]]
    """

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        x = F.square(logits - labels)
        return self.get_loss(x)


@constexpr
def _check_rmseloss_dtype(param_dtype, not_supported_dtype, cls_name):
    """Check RMSELoss not supported data type"""
    if param_dtype in not_supported_dtype:
        raise TypeError(f"For '{cls_name}', the parameters data type must not be in {not_supported_dtype}, "
                        f"but got mindspore.{str(param_dtype).lower()}.")


class RMSELoss(LossBase):
    r"""
    RMSELoss creates a criterion to measure the root mean square error between :math:`x` and :math:`y`
    element-wise, where :math:`x` is the input and :math:`y` is the labels.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the loss of :math:`x` and :math:`y` is given as:

    .. math::
        loss = \sqrt{\frac{1}{N}\sum_{i=1}^{N}{(x_i-y_i)^2}}

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*` means, any number of
          additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as the `logits` in common cases.
          However, it supports the shape of `logits` is different from the shape of `labels`
          and they should be broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is ().

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # Case 1: logits.shape = labels.shape = (3,)
        >>> loss = nn.RMSELoss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.57735026
        >>> # Case 2: logits.shape = (3,), labels.shape = (2, 3)
        >>> loss = nn.RMSELoss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        1.0
    """

    def __init__(self):
        """Initialize RMSELoss."""
        super(RMSELoss, self).__init__()
        self.dtype = P.DType()
        self.MSELoss = MSELoss()

    def construct(self, logits, label):
        logits_dtype = self.dtype(logits)
        label_dtype = self.dtype(label)
        not_supported_dtype = [mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64]
        _check_rmseloss_dtype(logits_dtype, not_supported_dtype, 'RMSELoss')
        _check_rmseloss_dtype(label_dtype, not_supported_dtype, "RMSELoss")

        rmse_loss = F.sqrt(self.MSELoss(logits, label))

        return rmse_loss


class MAELoss(LossBase):
    r"""
    MAELoss creates a criterion to measure the average absolute error between :math:`x` and :math:`y`
    element-wise, where :math:`x` is the input and :math:`y` is the labels.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If `reduction` is not 'none', then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
                         Default: "mean".

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(M, *)` where :math:`*` means, any number of
          additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as the `logits` in common cases.
          However, it supports the shape of `logits` is different from the shape of `labels`
          and they should be broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor, the shape is zero if `reduction` is 'mean' or 'sum',
        while the shape of output is the broadcasted shape if `reduction` is 'none'.

    Raises:
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # Case 1: logits.shape = labels.shape = (3,)
        >>> loss = nn.MAELoss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.33333334
        >>> # Case 2: logits.shape = (3,), labels.shape = (2, 3)
        >>> loss = nn.MAELoss(reduction='none')
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [[0. 1. 2.]
         [0. 0. 1.]]
    """

    def __init__(self, reduction='mean'):
        """Initialize MAELoss."""
        super(MAELoss, self).__init__(reduction)
        self.abs = P.Abs()

    def construct(self, logits, label):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', label, self.cls_name)
        x = self.abs(logits - label)
        return self.get_loss(x)


class MarginRankingLoss(LossBase):
    r"""
    MarginRankingLoss creates a criterion that measures the loss.

    Given two tensors :math:`input1`, :math:`input2` and a Tensor label :math:`target` with values 1 or -1,
    the operation is as follows:

    .. math::
        \text{loss}(input1, input2, target) = \max(0, -target * (input1 - input2) + \text{margin})

    Args:
        margin (float): Specify the adjustment factor of the operation. Default 0.0.
        reduction (str): Specifies which reduction to be applied to the output. It must be one of
          "none", "mean", and "sum", meaning no reduction, reduce mean and sum on output, respectively. Default "mean".

    Inputs:
        - **input1** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions.
        - **input2** (Tensor) - Tensor of shape :math:`(N, *)`, same shape and dtype as `input1`.
        - **target** (Tensor) - Contains value 1 or -1. Suppose the shape of `input1` is
          :math:`(x_1, x_2, x_3, ..., x_R)`, then the shape of `target` must be :math:`(x_1, x_2, x_3, ..., x_R)`.

    Outputs:
        Tensor or Scalar. if `reduction` is "none", its shape is the same as `labels`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If `margin` is not a float.
        TypeError: If `input1`, `input2` or `target` is not a Tensor.
        TypeError: If the types of `input1` and `input2` are inconsistent.
        TypeError: If the types of `input1` and `target` are inconsistent.
        ValueError: If the shape of `input1` and `input2` are inconsistent.
        ValueError: If the shape of `input1` and `target` are inconsistent.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore.ops import Tensor
        >>> import numpy as np
        >>> loss1 = nn.MarginRankingLoss(reduction='none')
        >>> loss2 = nn.MarginRankingLoss(reduction='mean')
        >>> loss3 = nn.MarginRankingLoss(reduction='sum')
        >>> sign = ops.Sign()
        >>> input1 = Tensor(np.array([0.3864, -2.4093, -1.4076]), ms.float32)
        >>> input2 = Tensor(np.array([-0.6012, -1.6681, 1.2928]), ms.float32)
        >>> target = sign(Tensor(np.array([-2, -2, 3]), ms.float32))
        >>> output1 = loss1(input1, input2, target)
        >>> print(output1)
        [0.98759997 0.         2.7003999 ]
        >>> output2 = loss2(input1, input2, target)
        >>> print(output2)
        1.2293333
        >>> output3 = loss3(input1, input2, target)
        >>> print(output3)
        3.6879997
    """

    def __init__(self, margin=0.0, reduction='mean'):
        """Initialize MarginRankingLoss."""
        super(MarginRankingLoss, self).__init__(reduction)
        self.reduction = reduction
        self.margin = margin

    def construct(self, input1, input2, target):
        x = ops.margin_ranking_loss(input1, input2, target, self.margin, self.reduction)
        return x


class SmoothL1Loss(LossBase):
    r"""
    SmoothL1 loss function, if the absolute error element-wise between the predicted value and the target value
    is less than the set threshold `beta`, the square term is used, otherwise the absolute error term is used.

    Given two input :math:`x,\  y`, the SmoothL1Loss can be described as follows:

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\beta}, & \text{if } |x_i - y_i| < {\beta} \\
        |x_i - y_i| - 0.5 {\beta}, & \text{otherwise.}
        \end{cases}

    Where :math:`{\beta}` represents the threshold `beta`.

    If `reduction` is not `none`, then:

    .. math::
        L =
        \begin{cases}
            \operatorname{mean}(L_{i}), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L_{i}),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. note::
        For Ascend platform, the float64 data type of `logits` is not support now.
        SmoothL1Loss can be regarded as modified version of L1Loss or a combination of L1Loss and L2Loss.
        L1Loss computes the element-wise absolute difference between two input tensors while L2Loss computes the
        squared difference between two input tensors. L2Loss often leads to faster convergence but it is less
        robust to outliers, and the loss function has better robustness.

    Args:
        beta (float): The loss function calculates the threshold of the transformation between L1Loss and L2Loss.
            Default: 1.0.
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
                         Default: "none".

    Inputs:
        - **logits** (Tensor) - Predictive value. Tensor of any dimension. Data type must be one of float16,
          float32 and float64.
        - **labels** (Tensor) - Ground truth data, same shape and dtype as the `logits`.

    Outputs:
        Tensor, if `reduction` is 'none', then output is a tensor with the same shape as `logits`.
        Otherwise the shape of output tensor is `()`.

    Raises:
        TypeError: If `beta` is not a float.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        TypeError: If `logits` or `labels` are not Tensor.
        TypeError: If dtype of `logits` or `labels` is neither float16 not float32.
        TypeError: If dtype of `logits` is not the same as `labels`.
        ValueError: If `beta` is less than or equal to 0.
        ValueError: If shape of `logits` is not the same as `labels`.
        TypeError: The float64 data type of `logits` is support on Ascend platform.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = nn.SmoothL1Loss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [0.  0.  0.5]
    """

    def __init__(self, beta=1.0, reduction='none'):
        """Initialize SmoothL1Loss."""
        super(SmoothL1Loss, self).__init__(reduction)
        self.beta = beta
        self.reduction = reduction
        self.smooth_l1_loss = P.SmoothL1Loss(self.beta, self.reduction)

    def construct(self, logits, labels):
        return self.smooth_l1_loss(logits, labels)


class SoftMarginLoss(LossBase):
    r"""
    A loss class for two-class classification problems.

    SoftMarginLoss creates a criterion that optimizes a two-class classification
    logistic loss between input tensor :math:`x` and labels tensor :math:`y`
    (containing 1 or -1).

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    :math:`x.nelement()` represents the number of element of `x` .

    Args:
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum'. Default: "mean".

    Inputs:
        - **logits** (Tensor) - Predict data. Data type must be float16 or float32.
        - **labels** (Tensor) - Ground truth data, with the same type and shape as `logits`.

    Outputs:
        Tensor or Scalar, if `reduction` is "none", its shape is the same as `logits`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If `logits` or `labels` is not a Tensor.
        TypeError: If dtype of `logits` or `labels` is neither float16 nor float32.
        ValueError: If shape of `logits` is not the same as `labels`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> loss = nn.SoftMarginLoss()
        >>> logits = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), mindspore.float32)
        >>> labels = Tensor(np.array([[-1, 1], [1, -1]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.6764238
    """

    def __init__(self, reduction='mean'):
        super(SoftMarginLoss, self).__init__()
        self.soft_margin_loss = P.SoftMarginLoss(reduction)

    def construct(self, logits, labels):
        return self.soft_margin_loss(logits, labels)


class SoftmaxCrossEntropyWithLogits(LossBase):
    r"""
    Computes softmax cross entropy between logits and labels.

    Measures the distribution error between the probabilities of the input (computed with softmax function) and the
    labels where the classes are mutually exclusive (only one class is positive) using cross entropy loss.

    Typical input into this function is unnormalized scores denoted as x whose shape is (N, C),
    and the corresponding targets.

    Typically, the input to this function is the fractional value of each category and the corresponding target value,
    and the input format is (N, C).

    For each instance :math:`x_i`, i ranges from 0 to N-1, the loss is given as:

    .. math::
        \ell(x_i, c) = - \log\left(\frac{\exp(x_i[c])}{\sum_j \exp(x_i[j])}\right)
        =  -x_i[c] + \log\left(\sum_j \exp(x_i[j])\right)

    where :math:`x_i` is a 1D score Tensor, :math:`c` is the index of 1 in one-hot.

    Note:
        While the labels classes are mutually exclusive, i.e., only one class is positive in the labels, the predicted
        probabilities does not need to be exclusive. It is only required that the predicted probability distribution
        of entry is a valid one.

    Args:
        sparse (bool): Specifies whether labels use sparse format or not. Default: False.
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
        - **labels** (Tensor) - Tensor of shape (N, ). If `sparse` is True, The type of
          `labels` is int32 or int64. Otherwise, the type of `labels` is the same as the type of `logits`.

    Outputs:
        Tensor, a tensor of the same shape and type as logits with the component-wise logistic losses.

    Raises:
        TypeError: If `sparse` is not a bool.
        TypeError: If `sparse` is True and dtype of `labels` is neither int32 not int64.
        TypeError: If `sparse` is False and dtype of `labels` is neither float16 not float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: sparse=True
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
        >>> labels_np = np.array([1]).astype(np.int32)
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels)
        >>> print(output)
        [67.]
        >>> # case 2: sparse=False
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        >>> logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
        >>> labels_np = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).astype(np.float32)
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels)
        >>> print(output)
        [30.]
    """

    def __init__(self,
                 sparse=False,
                 reduction='none'):
        """Initialize SoftmaxCrossEntropyWithLogits."""
        super(SoftmaxCrossEntropyWithLogits, self).__init__(reduction)
        self.sparse = validator.check_bool(sparse, "sparse", self.cls_name)
        self.reduction = reduction
        self.softmax_cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0., mstype.float32)
        self.is_cpugpu = context.get_context('device_target') in ["CPU", "GPU"]
        self.sparse_softmax_cross_entropy = P.SparseSoftmaxCrossEntropyWithLogits()

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        if self.sparse:
            if self.reduction == 'mean':
                x = self.sparse_softmax_cross_entropy(logits, labels)
                return x
            labels = self.one_hot(labels, F.shape(logits)[-1], self.on_value, self.off_value)
        x = self.softmax_cross_entropy(logits, labels)[0]
        return self.get_loss(x)


@constexpr
def _check_label_dtype(labels_dtype, cls_name):
    """Internal function, used to check whether the data type of labels meets the requirements."""
    validator.check_type_name("labels", labels_dtype, [mstype.int32, mstype.int64], cls_name)


class DiceLoss(LossBase):
    r"""
    The Dice coefficient is a set similarity loss, which is used to calculate the similarity between two samples. The
    value of the Dice coefficient is 1 when the segmentation result is the best and is 0 when the segmentation result
    is the worst. The Dice coefficient indicates the ratio of the area between two objects to the total area.
    The function is shown as follows:

    .. math::
        dice = 1 - \frac{2 * |pred \bigcap true|}{|pred| + |true| + smooth}

    :math:`pred` represent `logits`, :math:`true` represent `labels` .

    Args:
        smooth (float): A term added to the denominator to improve numerical stability. Should be greater than 0.
                        Default: 1e-5.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*` means, any number of
          additional dimensions. The data type must be float16 or float32.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as the `logits`.
          The data type must be float16 or float32.

    Outputs:
        Tensor, a tensor of shape with the per-example sampled Dice losses.

    Raises:
        ValueError: If the dimension of `logits` is different from `labels`.
        TypeError: If the type of `logits` or `labels` is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = nn.DiceLoss(smooth=1e-5)
        >>> logits = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mstype.float32)
        >>> labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mstype.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.38596618
    """

    def __init__(self, smooth=1e-5):
        """Initialize DiceLoss."""
        super(DiceLoss, self).__init__()
        self.smooth = validator.check_positive_float(smooth, "smooth")
        self.reshape = P.Reshape()

    def construct(self, logits, label):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', label, self.cls_name)
        if logits.dtype == mstype.uint8:
            raise TypeError(f"For '{self.cls_name}', the dtype of 'logits' can not be uint8.")
        if label.dtype == mstype.uint8:
            raise TypeError(f"For '{self.cls_name}', the dtype of 'labels' can not be uint8.")
        intersection = self.reduce_sum(self.mul(logits.view(-1), label.view(-1)))
        unionset = self.reduce_sum(self.mul(logits.view(-1), logits.view(-1))) + \
                   self.reduce_sum(self.mul(label.view(-1), label.view(-1)))

        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = 1 - single_dice_coeff

        return dice_loss


class MultiClassDiceLoss(LossBase):
    r"""
    When there are multiple classifications, label is transformed into multiple binary classifications by one hot.
    For each channel section in the channel, it can be regarded as a binary classification problem, so it can be
    obtained through the binary :class:`mindspore.nn.DiceLoss` losses of each category,
    and then the average value of the binary losses.

    Args:
        weights (Union[Tensor, None]): Tensor of shape :math:`(num\_classes, dim)`. The weight shape[0] should be
            equal to labels shape[1].
            Default: None.
        ignore_indiex (Union[int, None]): Class index to ignore.
            Default: None.
        activation (Union[str, Cell]): Activate function applied to the output of the fully connected layer, eg. 'ReLU'.
            Default: 'softmax'. Choose from: ['softmax', 'logsoftmax', 'relu', 'relu6', 'tanh','Sigmoid']

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, C, *)` where :math:`*` means, any number of additional
          dimensions. The logits dimension should be greater than 1. The data type must be float16 or float32.
        - **labels** (Tensor) - Tensor of shape :math:`(N, C, *)`, same shape as the `logits`.
          The labels dimension should be greater than 1. The data type must be float16 or float32.

    Outputs:
        Tensor, a tensor of shape with the per-example sampled MultiClass Dice Losses.

    Raises:
        ValueError: If the shape of `logits` is different from `labels`.
        TypeError: If the type of `logits` or `labels` is not a tensor.
        ValueError: If the dimension of `logits` or `labels` is less than 2.
        ValueError: If the weights.shape[0] is not equal to labels.shape[1].
        ValueError: If `weights` is a tensor, but its dimension is not 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation="softmax")
        >>> logits = Tensor(np.array([[0.2, 0.5, 0.7], [0.3, 0.1, 0.5], [0.9, 0.6, 0.3]]), mstype.float32)
        >>> labels = Tensor(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), mstype.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.54958105
    """

    def __init__(self, weights=None, ignore_indiex=None, activation="softmax"):
        """Initialize MultiClassDiceLoss."""
        super(MultiClassDiceLoss, self).__init__()
        activation_list = ['softmax', 'logsoftmax', 'relu', 'relu6', 'tanh', 'sigmoid']

        self.binarydiceloss = DiceLoss(smooth=1e-5)
        self.weights = weights if weights is None else validator.check_value_type("weights", weights, [Tensor])
        if isinstance(self.weights, Tensor) and self.weights.ndim != 2:
            raise ValueError(f"For '{self.cls_name}', the dimension of 'weights' must be 2, "
                             f"but got {self.weights.ndim}.")
        self.ignore_indiex = ignore_indiex if ignore_indiex is None else validator.check_value_type("ignore_indiex",
                                                                                                    ignore_indiex,
                                                                                                    [int])
        if isinstance(activation, str) and activation not in activation_list:
            raise ValueError(f"For '{self.cls_name}', the 'activation' must be in {activation_list}, "
                             f"but got {activation}.")

        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if self.activation is not None and not isinstance(self.activation, Cell):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell, "
                            f"but got {type(self.activation)}.")
        self.reshape = P.Reshape()

    def construct(self, logits, label):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', label, self.cls_name)
        total_loss = 0

        if self.activation is not None:
            logits = self.activation(logits)

        for i in range(label.shape[1]):
            if i != self.ignore_indiex:
                dice_loss = self.binarydiceloss(logits[:, i], label[:, i])
                if self.weights is not None:
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / label.shape[1]


class SampledSoftmaxLoss(LossBase):
    r"""
    Computes the sampled softmax training loss. This operator can accelerate the training of the softmax classifier
    over a large number of classes. It is generally an underestimate of the full softmax loss.

    Args:
        num_sampled (int): The number of classes to randomly sample per batch.
        num_classes (int): The number of possible classes.
        num_true (int): The number of labels classes per training example. Default: 1.
        sampled_values (Union[list, tuple]):  List or tuple of (`sampled_candidates`, `true_expected_count`,
            `sampled_expected_count`) returned by a `*CandidateSampler` function.
            Default to None, `UniformCandidateSampler` is applied.
        remove_accidental_hits (bool): Whether to remove "accidental hits"
            where a sampled class equals to one of the labels classes. Default: True.
        seed (int): Random seed for candidate sampling. Default: 0
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".

    Inputs:
        - **weights** (Tensor) - Tensor of shape :math:`(C, dim)`.
        - **bias** (Tensor) - Tensor of shape :math:`(C,)`. The class biases.
        - **labels** (Tensor) - Tensor of shape :math:`(N, num\_true)`, type `int64, int32`. The labels classes.
        - **logits** (Tensor) - Tensor of shape :math:`(N, dim)`. The forward activations of the input network.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor with shape :math:`(N,)`.
        Otherwise, the output is a scalar.

    Raises:
        TypeError: If `sampled_values` is not a list or tuple.
        TypeError: If dtype of `labels` is neither int32 not int64.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If `num_sampled` or `num_true` is greater than `num_classes`.
        ValueError: If length of `sampled_values` is not equal to 3.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> mindspore.set_seed(1)
        >>> loss = nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=1)
        >>> weights = Tensor(np.random.randint(0, 9, [7, 10]), mindspore.float32)
        >>> biases = Tensor(np.random.randint(0, 9, [7]), mindspore.float32)
        >>> labels = Tensor([0, 1, 2])
        >>> logits = Tensor(np.random.randint(0, 9, [3, 10]), mindspore.float32)
        >>> output = loss(weights, biases, labels, logits)
        >>> print(output)
        [4.6051701e+01 1.4000047e+01 6.1989022e-06]
    """

    def __init__(self, num_sampled, num_classes, num_true=1,
                 sampled_values=None, remove_accidental_hits=True, seed=0,
                 reduction='none'):
        """Initialize SampledSoftmaxLoss."""
        super(SampledSoftmaxLoss, self).__init__(reduction)

        if num_true < 1:
            raise ValueError(f"For '{self.cls_name}', the 'num_true' must be greater than or equal to 1, "
                             f"but got {num_true}.")
        if seed < 0:
            raise ValueError(f"For '{self.cls_name}', the 'seed' must be greater than or equal to 0, but got {seed}.")
        if num_sampled > num_classes:
            raise ValueError(f"For '{self.cls_name}', the 'num_sampled' must be smaller than or "
                             f"equal to 'num_classes', but got 'num_sampled': {num_sampled} "
                             f"and 'num_classes': {num_classes}.")
        if num_true > num_classes:
            raise ValueError(f"For '{self.cls_name}', the 'num_true' must be smaller than or equal to 'num_classes', "
                             f"but got 'num_true': {num_true} amd 'num_classes': {num_classes}.")
        if sampled_values is not None:
            if not isinstance(sampled_values, (list, tuple)):
                raise TypeError(f"For '{self.cls_name}', the type of 'sampled_values' must be a list or tuple, "
                                f"but got {type(sampled_values).__name__}.")
            if len(sampled_values) != 3:
                raise ValueError(f"For '{self.cls_name}', the length of 'sampled_values' must be equal to 3,"
                                 f"but got {len(sampled_values)}.")

        self.num_sampled = num_sampled
        self.num_classes = num_classes
        self.num_true = num_true
        self.sampled_values = sampled_values
        self.remove_accidental_hits = remove_accidental_hits
        self.seed = seed
        self.sampler = P.UniformCandidateSampler(
            num_true,
            num_sampled,
            True,
            num_classes,
            seed,
            remove_accidental_hits)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.exp = P.Exp()
        self.log = P.Log()
        self.slice_op = P.Slice()
        self.matmul = P.MatMul(False, True)
        self.gather_v2 = P.Gather()
        self.reduce_max_true = P.ReduceMax(True)
        self.reduce_sum = P.ReduceSum()
        self.reduce_sum_true = P.ReduceSum(True)
        self.concat_dim0 = P.Concat(0)
        self.concat_dim1 = P.Concat(1)
        self.ones_like = P.OnesLike()
        self.zeros_like = P.ZerosLike()
        self.mul = P.Mul()
        self.expand_dims = P.ExpandDims()
        self.dtype = P.DType()

    def construct(self, weights, biases, labels, logits):
        _check_is_tensor('weights', weights, self.cls_name)
        _check_is_tensor('biases', biases, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        _check_is_tensor('logits', logits, self.cls_name)
        _check_label_dtype(self.dtype(labels), self.cls_name)

        logits, labels = self._compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            logits=logits,
            num_true=self.num_true,
            sampled_values=self.sampled_values,
            subtract_log_q=True)

        x = self._softmax_cross_entropy(logits, labels)
        return x

    def _softmax_cross_entropy(self, logits, targets):
        stable_exp_logits = self.exp(logits - self.reduce_max_true(logits, 1))
        pred = stable_exp_logits / self.reduce_sum_true(stable_exp_logits, 1)
        return -self.reduce_sum(targets * self.log(pred + 1.0e-20), 1)

    def _compute_sampled_logits(self, weights,
                                biases,
                                labels,
                                logits,
                                num_true=1,
                                sampled_values=None,
                                subtract_log_q=True):
        """Helper function for SampledSoftmaxLoss functions.

        Computes sampled output training logits and labels suitable

        Note: In the case where num_true > 1, we assign to each labels class
        with the labels probability (1/num_true) so that the labels probabilities
        sum to 1 per-example.

        Args:
            weights (Tensor): Tensor of shape `[num_classes, dim]`.
            biases (Tensor): Tensor of shape `[num_classes]`.
            labels (Tensor): Tensor of shape `[batch_size, num_true]`. The labels classes.
            logits (Tensor): Tensor of shape `[batch_size, dim]`. The forward
                activations of the input network.
            num_true (int): The number of labels classes per training example.
            sampled_values: A tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `UniformCandidateSampler` function.
            subtract_log_q: A `bool`. whether to subtract the log expected count of
                the labels in the sample to get the logits of the true labels. Default: True.
        Returns:
            out_logits: `Tensor` object with shape
                `[batch_size, num_true + num_sampled]`
            out_labels: A tensor object with the same shape as `out_logits`.
        """

        if not labels.dtype == mstype.int32:
            labels = self.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1, num_true))
        labels_flat = self.reshape(labels, (-1,))

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape is [batch_size, 1] tensor
        #   sampled_expected_count shape is [num_sampled] tensor
        if sampled_values is None:
            sampled_values = self.sampler(labels)

        (sampled, true_expected_count, sampled_expected_count) = sampled_values

        if not sampled.dtype == mstype.int32:
            sampled = self.cast(sampled, mstype.int32)
        all_ids = self.concat_dim0((labels_flat, sampled))
        all_w = self.gather_v2(weights, all_ids, 0)

        n_true = self.shape(labels_flat)[0]
        n_sampled = self.shape(sampled)[0]
        n_dim = self.shape(all_w)[1]

        true_w = self.slice_op(all_w, [0, 0], [n_true, n_dim])
        sampled_w = self.slice_op(all_w, [n_true, 0], [n_sampled, n_dim])
        sampled_logits = self.matmul(logits, sampled_w)

        all_b = self.gather_v2(biases, all_ids, 0)
        true_b = self.slice_op(all_b, [0], [n_true])
        sampled_b = self.slice_op(all_b, [n_true], [n_sampled])

        new_true_w_shape = (-1, num_true, n_dim)
        row_wise_dots = self.mul(self.expand_dims(logits, 1),
                                 self.reshape(true_w, new_true_w_shape))

        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = self.reshape(row_wise_dots, (-1, n_dim))
        true_logits = self.reshape(self.reduce_sum(dots_as_matrix, 1), (-1, num_true))
        true_b = self.reshape(true_b, (-1, num_true))
        true_logits += true_b
        sampled_logits += sampled_b

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= self.log(true_expected_count)
            sampled_logits -= self.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = self.concat_dim1((true_logits, sampled_logits))

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = self.concat_dim1((
            self.ones_like(true_logits) / num_true,
            self.zeros_like(sampled_logits)
        ))
        return out_logits, out_labels


class PoissonNLLLoss(LossBase):
    r"""
    Poisson negative log likelihood loss.

    The loss is:

    .. math::
        \mathcal{L}_{D} = \sum_{i = 0}^{|D|}\left( x_{i} - y_{i}\ln x_{i} + \ln{y_{i}!} \right)

    where :math:`\mathcal{L}_{D}` is the loss, :math:`y_{i}` is the `target`,
    :math:`x_{i}` is the `x`.

    If `log_input` is True, use :math:`e^{x_{i}} - y_{i} x_{i}` instead of :math:`x_{i} - y_{i}\ln x_{i}`.
    When calculating logarithms, the lower bound of `x` is set to `eps` to avoid numerical errors.

    If `full` is False, the last term :math:`\ln{y_{i}!}` will be omitted,
    otherwise the last term will be approximated using Stirling formula:

    .. math::
        n! \approx \sqrt{2\pi n}\left( \frac{n}{e} \right)^{n}

    Note:
        Calculating the logarithm of a negative number or the exponent of a large positive number under Ascend
        will have a different range of return values and results different from those under GPU and CPU.

    Args:
        log_input (bool, optional): Whether use log input. Default: True.
        full (bool, optional): Whether include the Stirling approximation term in the loss calculation. Default: False.
        eps (float, optional): Lower bound of `x` when calculating logarithms. Default: 1e-08.
        reduction (str, optional): Apply specific reduction method to the output:
            'none', 'mean', 'sum'. Default: 'mean'.

    Inputs:
        - **x** (Tensor) - The input Tensor. The shape can be any number of dimensions.
        - **target** (Tensor) - The label Tensor which has the same shape as `x`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `x`.
        Otherwise it is a scalar.

    Raises:
        TypeError: If `reduction` is not a str.
        TypeError: If neither `x` nor `target` is a tensor.
        TypeError: If dtype of `x` or `target` is not currently supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0.3, 0.7], [0.5, 0.5]])
        >>> target = Tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> loss = nn.PoissonNLLLoss()
        >>> output = loss(x, target)
        >>> print(output.asnumpy())
        0.3652635
    """

    def __init__(self, log_input=True, full=False, eps=1e-08, reduction="mean"):
        """Initialize PoissonNLLLoss."""
        super(PoissonNLLLoss, self).__init__(reduction=reduction)
        self.log_input = log_input
        self.full = full
        self.eps = eps
        self.maximum = P.Maximum()
        self.cast = P.Cast()

    def construct(self, x, target):
        _check_is_tensor('x', x, self.cls_name)
        _check_is_tensor('target', target, self.cls_name)
        if x.ndim == 0 or target.ndim == 0:
            raise ValueError(
                "For 'PoissonNLLLoss', the inputs must be non-scalar, but got shapes: "
                f"x: {x.shape}, target: {target.shape}"
            )
        target = self.cast(target, x.dtype)
        if self.log_input:
            loss = x.exp() - target * x
        else:
            loss = x - target * ((x + self.eps).log())
        if self.full:
            target = self.maximum(target, self.eps)
            stirling_term = (target > 1) * ((target + 0.5) * target.log() - target + get_half_ln_2_pi())
            loss += F.masked_fill(stirling_term, target <= 1, 0)
        out = self.get_loss(loss)
        return out


@constexpr
def get_half_ln_2_pi():
    return 0.5 * math.log(2 * math.pi)


class MultiLabelSoftMarginLoss(LossBase):
    r"""
    Calculates the MultiLabelSoftMarginLoss.
    Create a criterion for optimizing multi-label one-to-total loss based on maximum entropy.

    .. math::
        \mathcal{L}_{D} = - \frac{1}{|D|}\sum_{i = 0}^{|D|}\left(
        y_{i}\ln\frac{1}{1 + e^{- x_{i}}} + \left( 1 - y_{i}
        \right)\ln\frac{1}{1 + e^{x_{i}}} \right)

    where :math:`\mathcal{L}_{D}` is the loss, :math:`y_{i}` is the `target`,
    :math:`x_{i}` is the `x`. `weight` will multiply to the loss of each class if given.

    Args:
        weight (Union[Tensor, int, float]): The manual rescaling weight given to each class. Default: None.
        reduction (str): Specifies which reduction to be applied to the output. It must be one of
            'none', 'mean', and 'sum', meaning no reduction, reduce mean and sum on output, respectively.
            Default: 'mean'.

    Inputs:
        - **x** (Tensor) - A tensor of shape (N, C), where N is batch size and C is number
          of classes.
        - **target** (Tensor) - The label target Tensor which has the same shape as `x`.

    Outputs:
        Tensor, the data type is the same as x, if the reduction is 'none', its shape is (N), otherwise it is zero.

    Raises:
        ValueError: If the rank of `x` or `target` is not 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0.3, 0.6, 0.6], [0.9, 0.4, 0.2]])
        >>> target = Tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        >>> loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
        >>> out = loss(x, target)
        >>> print(out.asnumpy())
        0.84693956
    """

    def __init__(self, weight=None, reduction="mean"):
        """Initialize MultiLabelSoftMarginLoss."""
        super(MultiLabelSoftMarginLoss, self).__init__(reduction)
        self.weight = weight
        self.reduction = reduction

    def construct(self, x, target):
        return F.multilabel_soft_margin_loss(x, target, self.weight, self.reduction)


class MultiMarginLoss(LossBase):
    r"""
    Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`) and
    output :math:`y` (which is a 1D tensor of target class indices,
    :math:`0 \leq y \leq \text{x.size}(1)-1`):

    For each mini-batch sample, the loss in terms of the 1D input :math:`x` and scalar
    output :math:`y` is:

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p)}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`
    and :math:`i \neq y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D input `weight` tensor w into the constructor.

    Args:
        p (int): Optional. The norm degree for pairwise distance. Should be 1 or 2. Default: 1.
        margin (float): Optional. A parameter to change pairwise distance. Default: 1.0.
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum'. Default: "mean".

    Inputs:
        - **x** (Tensor) - Input x, with shape :math:`(N, C)`. Data type only support float32, float16 or float64.
        - **target** (Tensor) - Ground truth labels, with shape :math:`(N,)`. Data type only support int64. The
          value of target should be non-negative, less than C.
        - **weight** (Tensor, optional) - The rescaling weight to each class with shape :math:`(C,)`. Data type only
          support float32, float16 or float64. Default: None.

    Outputs:
        Tensor, When `reduction` is 'none', the shape is :math:`(N,)`.
        Otherwise, it is a scalar. Has the same data type with `x`.

    Raises:
        TypeError: If dtype of `p` or `target` is not int.
        TypeError: If dtype of `margin` is not float.
        TypeError: If dtype of `reduction` is not str.
        TypeError: If dtype of `x` is not float16, float or float64.
        TypeError: If dtype of `weight` and `x` is not the same.
        ValueError: If 'p' is not 1 or 2.
        ValueError: If 'reduction' is not one of {'none','sum','mean'}.
        ValueError: If shape[0] of `x` is not equal to shape[0] of `target`.
        ValueError: If shape[1] of `x` is not equal to shape[0] of `weight`.
        ValueError: IF rank of `weight` is not 1.
        ValueError: If rank of `x` is not 2 or rank of 'target' is not 1.

    Supported Platforms:
        ``Ascend``  ``CPU``

    Examples:
        >>> x = Tensor(np.ones(shape=[3, 3]), mindspore.float32)
        >>> target = Tensor(np.array([1, 2, 1]), mindspore.int64)
        >>> loss = nn.MultiMarginLoss()
        >>> output = loss(x, target)
        >>> print(output)
        0.6666667
    """

    def __init__(self, p=1, margin=1.0, reduction='mean'):
        """Initialize MultiMarginLoss."""
        super(MultiMarginLoss, self).__init__()
        self.multi_margin_loss = MultiMarginLossOp(p=p, margin=margin, reduction=reduction)
        self.ones = P.Ones()

    def construct(self, x, target, weight=None):
        _check_is_tensor('x', x, self.cls_name)
        _check_is_tensor('target', target, self.cls_name)
        weight_one = weight is None
        if not weight_one:
            _check_is_tensor('weight', weight, self.cls_name)
        else:
            weight = self.ones(x.shape[1], x.dtype)
        loss = self.multi_margin_loss(x, target, weight)
        return loss


class BCELoss(LossBase):
    r"""
    BCELoss creates a criterion to measure the binary cross entropy between the true labels and predicted labels.

    Set the predicted labels as :math:`x`, true labels as :math:`y`, the output loss as :math:`\ell(x, y)`.
    The formula is as follow:

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    where N is the batch size. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Note:
        Note that the predicted labels should always be the output of sigmoid. Because it is a two-class
        classification, the true labels should be numbers between 0 and 1.
        And if input is either 0 or 1, one of the log terms would be mathematically undefined in the above loss
        equation.

    Args:
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            And it must have the same shape and data type as `inputs`. Default: None
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.

    Inputs:
        - **logits** (Tensor) - The input tensor with shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions. The data type must be float16 or float32.
        - **labels** (Tensor) - The label tensor with shape :math:`(N, *)`, the same shape and data type as `logits`.

    Outputs:
        Tensor, has the same dtype as `logits`. if `reduction` is 'none', then it has the same shape as `logits`.
        Otherwise, it is a scalar Tensor.

    Raises:
        TypeError: If dtype of `logits`, `labels` or `weight` (if given) is neither float16 not float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If shape of `logits` is not the same as `labels` or `weight` (if given).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> weight = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 3.3, 2.2]]), mindspore.float32)
        >>> loss = nn.BCELoss(weight=weight, reduction='mean')
        >>> logits = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        1.8952923
    """

    def __init__(self, weight=None, reduction='none'):
        """Initialize BCELoss."""
        super(BCELoss, self).__init__()
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction=reduction)
        self.weight_one = weight is None
        if not self.weight_one:
            self.weight = weight
        else:
            self.ones = P.OnesLike()

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        if self.weight_one:
            weight = self.ones(logits)
        else:
            weight = self.weight
        loss = self.binary_cross_entropy(logits, labels, weight)
        return loss


class CosineEmbeddingLoss(LossBase):
    r"""
    CosineEmbeddingLoss creates a criterion to measure the similarity between two tensors using cosine distance.

    Given two tensors :math:`x1`, :math:`x2`, and a Tensor label :math:`y` with values 1 or -1:

    .. math::
        loss(x_1, x_2, y) = \begin{cases}
        1-cos(x_1, x_2), & \text{if } y = 1\\
        max(0, cos(x_1, x_2)-margin), & \text{if } y = -1\\
        \end{cases}

    Args:
        margin (float): Should be in [-1.0, 1.0]. Default 0.0.
        reduction (str): Specifies which reduction to be applied to the output. It must be one of
          "none", "mean", and "sum", meaning no reduction, reduce mean and sum on output, respectively. Default "mean".

    Inputs:
        - **logits_x1** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions.
        - **logits_x2** (Tensor) - Tensor of shape :math:`(N, *)`, same shape and dtype as `logits_x1`.
        - **labels** (Tensor) - Contains value 1 or -1. Suppose the shape of `logits_x1` is
          :math:`(x_1, x_2, x_3, ..., x_R)`, then the shape of `labels` must be :math:`(x_1, x_3, x_4, ..., x_R)`.

    Outputs:
        Tensor or Scalar, if `reduction` is "none", its shape is the same as `labels`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If `margin` is not a float.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        ValueError: If `margin` is not in range [-1, 1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits_x1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]), mindspore.float32)
        >>> logits_x2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, -1]), mindspore.int32)
        >>> cosine_embedding_loss = nn.CosineEmbeddingLoss()
        >>> output = cosine_embedding_loss(logits_x1, logits_x2, labels)
        >>> print(output)
        0.0003425479
    """

    def __init__(self, margin=0.0, reduction="mean"):
        """Initialize CosineEmbeddingLoss."""
        super(CosineEmbeddingLoss, self).__init__(reduction)
        self.reduce_sum = P.ReduceSum()
        self.maximum = P.Maximum()
        validator.check_value_type("margin", margin, [float], self.cls_name)
        self.margin = validator.check_float_range(margin, -1.0, 1.0, Rel.INC_BOTH, "margin", self.cls_name)

    def construct(self, logits_x1, logits_x2, labels):
        _check_is_tensor('logits_x1', logits_x1, self.cls_name)
        _check_is_tensor('logits_x2', logits_x2, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        inner.same_type_shape_(logits_x1, logits_x2)
        # if labels > 0, 1-cosine(logits_x1, logits_x2)
        # else, max(0, cosine(logits_x1, logits_x2)-margin)
        prod_sum = self.reduce_sum(logits_x1 * logits_x2, (1,))
        square1 = self.reduce_sum(F.square(logits_x1), (1,))
        square2 = self.reduce_sum(F.square(logits_x2), (1,))
        denom = F.sqrt(square1) * F.sqrt(square2)
        cosine = prod_sum / denom

        pos_value = 1.0 - cosine
        neg_value = self.maximum(cosine - self.margin, 0.0)
        zeros = F.zeros_like(cosine)
        pos_part = F.select(labels == 1, pos_value, zeros)
        neg_part = F.select(labels == -1, neg_value, zeros)
        output_unreduced = pos_part + neg_part

        return self.get_loss(output_unreduced)


class MultilabelMarginLoss(LossBase):
    r"""
    MultilabelMarginLoss operation.

    Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input :math:`x` (a 2D mini-batch `Tensor`)
    and output :math:`y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch:

    .. math::
        \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

    where :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`, \
    :math:`y \in \left\{0, \; \cdots , \; \text{y.size}(0) - 1\right\}`, \
    :math:`0 \leq y[j] \leq \text{x.size}(0)-1`, \
    and :math:`i \neq y[j]` for all :math:`i` and :math:`j`.

    :math:`y` and :math:`x` must have the same size.

    The criterion only considers a contiguous block of non-negative targets that
    starts at the front.

    This allows for different samples to have variable amounts of target classes.

    Args:
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', 'sum'. Default: "mean".

    Inputs:
        - **x** (Tensor) - Predict data. Tensor of shape :math:`(C)` or :math:`(N, C)`, where :math:`N`
          is the batch size and :math:`C` is the number of classes. Data type must be float16 or float32.
        - **target** (Tensor) - Ground truth data, with the same shape as `x`, data type must be int32 and
          label targets padded by -1.

    Outputs:
        - **y** (Union[Tensor, Scalar]) - The loss of MultilabelMarginLoss. If `reduction` is "none", its shape
          is :math:`(N)`. Otherwise, a scalar value will be returned.
        - **is_target** (Tensor) - Output tensor for backward input, with the same shape as `target`,
          data type must be int32.

    Raises:
        TypeError: If `x` or `target` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If dtype of `target` is not int32.
        ValueError: If length of shape of `x` is neither 1 nor 2.
        ValueError: If shape of `x` is not the same as `target`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
       >>> loss = nn.MultilabelMarginLoss()
       >>> x = Tensor(np.array([[0.1, 0.2, 0.4, 0.8], [0.2, 0.3, 0.5, 0.7]]), mindspore.float32)
       >>> target = Tensor(np.array([[1, 2, 0, 3], [2, 3, -1, 1]]), mindspore.int32)
       >>> output = loss(x, target)
       >>> print(output)
       (Tensor(shape=[], dtype=Float32, value= 0.325), Tensor(shape=[2, 4], dtype=Int32, value=
       [[1, 1, 1, 1], [0, 0, 1, 1]]))
    """

    def __init__(self, reduction='mean'):
        super(MultilabelMarginLoss, self).__init__()
        self.multilabel_margin_loss = MultilabelMarginLossOp(reduction=reduction)

    def construct(self, x, target):
        return self.multilabel_margin_loss(x, target)


class BCEWithLogitsLoss(LossBase):
    r"""
    Adds sigmoid activation function to input logits, and uses the given logits to compute binary cross entropy
    between the logits and the labels.

    Sets input `logits` as :math:`X`, input `labels` as :math:`Y`, output as :math:`L`. Then,

    .. math::
        p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}}

    .. math::
        L_{ij} = -[Y_{ij} \cdot log(p_{ij}) + (1 - Y_{ij}) \cdot log(1 - p_{ij})]

    Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are 'mean', 'sum', and 'none'.
            If 'none', do not perform reduction. Default: 'mean'.
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            If not None, it can be broadcast to a tensor with shape of `logits`,
            data type must be float16 or float32. Default: None.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the
            number of classes. If not None, it must be broadcast to a tensor with shape of `logits`, data type
            must be float16 or float32. Default: None.

    Inputs:
        - **logits** (Tensor) - Input logits with shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions. The data type must be float16 or float32.
        - **labels** (Tensor) - Ground truth label with shape :math:`(N, *)`, same shape and dtype as `logits`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', its shape is the same as `logits`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If input `logits` or `labels` is not Tensor.
        TypeError: If data type of `logits` or `labels` is neither float16 nor float32.
        TypeError: If `weight` or `pos_weight` is a parameter.
        TypeError: If data type of `weight` or `pos_weight` is neither float16 nor float32.
        TypeError: If data type of `reduction` is not string.
        ValueError: If `weight` or `pos_weight` can not be broadcast to a tensor with shape of `logits`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend``  ``GPU``  ``CPU``

    Examples:
        >>> logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]).astype(np.float32))
        >>> labels = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]).astype(np.float32))
        >>> loss = nn.BCEWithLogitsLoss()
        >>> output = loss(logits, labels)
        >>> print(output)
        0.3463612
    """

    def __init__(self, reduction='mean', weight=None, pos_weight=None):
        """Initialize BCEWithLogitsLoss."""
        super(BCEWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.bce_with_logits_loss = P.BCEWithLogitsLoss(reduction=reduction)
        if isinstance(weight, Parameter):
            raise TypeError(f"For '{self.cls_name}', the 'weight' can not be a Parameter.")
        if isinstance(pos_weight, Parameter):
            raise TypeError(f"For '{self.cls_name}', the 'pos_weight' can not be a Parameter.")
        self.weight = weight
        self.pos_weight = pos_weight
        self.ones = P.OnesLike()

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        ones_input = self.ones(logits)
        if self.weight is not None:
            weight = self.weight
        else:
            weight = ones_input
        if self.pos_weight is not None:
            pos_weight = self.pos_weight
        else:
            pos_weight = ones_input
        loss = self.bce_with_logits_loss(logits, labels, weight, pos_weight)
        return loss


@constexpr
def _check_input_dtype(labels_dtype, cls_name):
    """Internal function, used to check whether the data type of labels meets the requirements."""
    validator.check_type_name("labels", labels_dtype,
                              [mstype.int32, mstype.int64, mstype.float16, mstype.float32], cls_name)


class FocalLoss(LossBase):
    r"""
    It is a loss function to solve the imbalance of categories and the difference of
    classification difficulty.
    The loss function proposed by Kaiming team in their paper
    `Focal Loss for Dense Object Detection <https://arxiv.org/pdf/1708.02002.pdf>`_ improves the
    effect of image object detection.
    The function is shown as follows:

    .. math::
        FL(p_t) = -(1-p_t)^\gamma log(p_t)

    Args:
        gamma (float): Gamma is used to adjust the steepness of weight curve in focal loss. Default: 2.0.
        weight (Union[Tensor, None]): A rescaling weight applied to the loss of each batch element. The dimension of
                                      weight should be 1. If None, no weight is applied. Default: None.
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
                         If "none", do not perform reduction. Default: "mean".

    Inputs:
        - **logits** (Tensor) - Tensor of shape should be :math:`(N, C)` or :math:`(N, C, H)` or :math:`(N, C, H, W)`.
          Where :math:`C` is the number of classes. Its value is greater than 1. If the shape is :math:`(N, C, H, W)`
          or :math:`(N, C, H)`, the :math:`H` or product of :math:`H` and :math:`W` should be the same as labels.
        - **labels** (Tensor) - Tensor of shape should be :math:`(N, C)` or :math:`(N, C, H)` or :math:`(N, C, H, W)`.
          The value of :math:`C` is 1 or it needs to be the same as predict's :math:`C`. If :math:`C` is not 1,
          the shape of target should be the same as that of predict, where :math:`C` is the number of classes.
          If the shape is :math:`(N, C, H, W)` or :math:`(N, C, H)`, the :math:`H` or product of :math:`H`
          and :math:`W` should be the same as logits. The value of `labels` is should be in the
          range [-:math:`C`, :math:`C`). Where :math:`C` is the number of classes in logits.

    Outputs:
        Tensor or Scalar, if `reduction` is "none", its shape is the same as `logits`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If the data type of `gamma` is not a float.
        TypeError: If `weight` is not a Tensor.
        ValueError: If `labels` dim is different from `logits`.
        ValueError: If `labels` channel is not 1 and `labels` shape is different from `logits`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> logits = Tensor([[0.8, 1.4], [0.5, 0.9], [1.2, 0.9]], mstype.float32)
        >>> labels = Tensor([[1], [1], [0]], mstype.int32)
        >>> focalloss = nn.FocalLoss(weight=Tensor([1, 2]), gamma=2.0, reduction='mean')
        >>> output = focalloss(logits, labels)
        >>> print(output)
        0.12516622
    """

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """Initialize FocalLoss."""
        super(FocalLoss, self).__init__(reduction=reduction)

        self.gamma = validator.check_value_type("gamma", gamma, [float])
        if weight is not None and not isinstance(weight, Tensor):
            raise TypeError(f"For '{self.cls_name}', the type of 'weight' must be a Tensor, "
                            f"but got {type(weight).__name__}.")
        if isinstance(weight, Tensor) and weight.ndim != 1:
            raise ValueError(f"For '{self.cls_name}', the dimension of 'weight' must be 1, but got {weight.ndim}.")
        self.weight = weight
        self.expand_dims = P.ExpandDims()
        self.gather_d = P.GatherD()
        self.squeeze = P.Squeeze(axis=1)
        self.tile = P.Tile()
        self.cast = P.Cast()
        self.dtype = P.DType()
        self.logsoftmax = nn.LogSoftmax(1)

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        labelss = labels
        _check_input_dtype(self.dtype(labelss), self.cls_name)

        if logits.ndim > 2:
            logits = logits.view(logits.shape[0], logits.shape[1], -1)
            labelss = labelss.view(labelss.shape[0], labelss.shape[1], -1)
        else:
            logits = self.expand_dims(logits, 2)
            labelss = self.expand_dims(labelss, 2)

        log_probability = self.logsoftmax(logits)

        if labels.shape[1] == 1:
            log_probability = self.gather_d(log_probability, 1, self.cast(labelss, mindspore.int32))
            log_probability = self.squeeze(log_probability)

        probability = F.exp(log_probability)

        if self.weight is not None:
            convert_weight = self.weight[None, :, None]
            convert_weight = self.tile(convert_weight, (labelss.shape[0], 1, labelss.shape[2]))
            if labels.shape[1] == 1:
                convert_weight = self.gather_d(convert_weight, 1, self.cast(labelss, mindspore.int32))
                convert_weight = self.squeeze(convert_weight)
            log_probability = log_probability * convert_weight

        weight = F.pows(-1 * probability + 1.0, self.gamma)
        if labels.shape[1] == 1:
            loss = (-1 * weight * log_probability).mean(axis=1)
        else:
            loss = (-1 * weight * labelss * log_probability).mean(axis=-1)

        return self.get_loss(loss)


class HuberLoss(LossBase):
    r"""
    HuberLoss calculate the error between the predicted value and the target value.
    It has the advantages of both L1Loss and MSELoss.

    Assuming that the :math:`x` and :math:`y` are 1-D Tensor, length :math:`N`, then calculate the loss of :math:`x` and
    :math:`y` without dimensionality reduction (the reduction parameter is set to "none"). The formula is as follows:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top

    with

    .. math::
        l_n = \begin{cases}
            0.5 * (x_n - y_n)^2, & \text{if } |x_n - y_n| < delta; \\
            delta * (|x_n - y_n| - 0.5 * delta), & \text{otherwise. }
        \end{cases}

    where :math:`N` is the batch size. If `reduction` is not "none", then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{"mean";}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{"sum".}
        \end{cases}

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean". If `reduction` is "mean" or "sum", then output a scalar Tensor, if `reduction` is "none",
            the shape of the output Tensor is the broadcasted shape.
        delta (Union[int, float]): The threshold to change between two type of loss.
            The value must be positive. Default: 1.0.

    Inputs:
        - **logits** (Tensor) - Predicted value, Tensor of any dimension. The data type must be float16 or float32.
        - **labels** (Tensor) - Target value, same dtype and shape as the `logits` in common cases.
          However, it supports the shape of `logits` is different from the shape of `labels`
          and they should be broadcasted to each other.

    Outputs:
        Tensor or Scalar, if `reduction` is "none", return a Tensor with same shape and dtype as `logits`.
        Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If data type of `logits` or `labels` is neither float16 nor float32.
        TypeError: If data type of `logits` or `labels` are not the same.
        TypeError: If dtype of `delta` is neither float nor int.
        ValueError: If `delta` is less than or equal to 0.
        ValueError: If `reduction` is not one of "none", "mean", "sum".
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted to each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> # Case 1: logits.shape = labels.shape = (3,)
        >>> loss = nn.HuberLoss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        0.16666667
        >>> # Case 2: logits.shape = (3,), labels.shape = (2, 3)
        >>> loss = nn.HuberLoss(reduction="none")
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [[0.  0.5 1.5]
         [0.  0.  0.5]]
    """

    def __init__(self, reduction='mean', delta=1.0):
        """Initialize HuberLoss."""
        super(HuberLoss, self).__init__(reduction=reduction)
        self.reduction = reduction
        self.delta = delta

    def construct(self, logits, labels):
        return F.huber_loss(logits, labels, self.reduction, self.delta)


class TripletMarginLoss(LossBase):
    r"""
    TripletMarginLoss operation.

    Creates a criterion that measures the triplet loss given an input
    tensors :math:`x`, :math:`positive`, :math:`negative` and a :math:`margin` with a value greater than :math:`0`.
    This is used for measuring a relative similarity between samples.
    A triplet is composed by `a`, `p` and `n` (i.e., `x`, `positive` and `negative` respectively).
    The shapes of all input tensors should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper
    `Learning local feature descriptors with triplets and shallow convolutional neural
    networks <http://158.109.8.37/files/BRP2016.pdf>`_
    by V. Balntas, E. Riba et al.

    The loss function for each sample in the mini-batch is:

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    where

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    Args:
        p (int, optional): The norm degree for pairwise distance. Default: 2.
        eps (float, optional): Add small value to avoid division by zero. Default: 1e-06.
        swap (bool, optional): The distance swap change the negative distance to the distance between positive
            sample and negative sample. Default: "False".
        reduction (str, optional): Apply specific reduction method to the output: 'none', 'mean', 'sum'.
            Default: "mean".

    Inputs:
        - **x** (Tensor) - A sample randomly selected from the training set. Data type must be BasicType.
        - **positive** (Tensor) - A sample belonging to the same category as `x`, with the same type and shape as `x`.
        - **negative** (Tensor) - A sample belonging to the different class from `x`, with the same type and shape
          as `x`.
        - **margin** (Tensor) - Make a margin between the positive pair and the negative pair.

    Outputs:
        Tensor. If `reduction` is "none", its shape is :math:`(N)`. Otherwise, a scalar value will be returned.

    Raises:
        TypeError: If `x` or `positive` or 'negative' or 'margin' is not a Tensor.
        TypeError: If dtype of `x`, `positive` and `negative` is not the same.
        TypeError: If `margin` is not float32.
        TypeError: If `p` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `swap` is not a bool.
        ValueError: If dimensions of input `x`, `positive` and `negative` are less than or equal to 1 at the same time.
        ValueError: If the dimension of input `x` or `positive` or `negative` is bigger than or equal to 8.
        ValueError: If length of shape of `margin` is not 0.
        ValueError: If shape of `x`, `positive` and `negative` cannot broadcast.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> loss = nn.TripletMarginLoss()
        >>> x = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), mindspore.float32)
        >>> positive = Tensor(np.array([[0.4, 0.6], [0.4, 0.6]]), mindspore.float32)
        >>> negative = Tensor(np.array([[0.2, 0.9], [0.3, 0.7]]), mindspore.float32)
        >>> margin = Tensor(1.0, mindspore.float32)
        >>> output = loss(x, positive, negative, margin)
        >>> print(output)
        0.8881968
    """

    def __init__(self, p=2, swap=False, eps=1e-06, reduction='mean'):
        super(TripletMarginLoss, self).__init__()
        self.p = p
        self.swap = swap
        self.eps = eps
        self.reduction = reduction

    def construct(self, x, positive, negative, margin):
        return F.triplet_margin_loss(x, positive, negative, margin=margin, p=self.p,
                                     eps=self.eps, swap=self.swap, reduction=self.reduction)


@constexpr
def _check_nll_loss_inputs(logits_shape, label_shape, logits_dtype, label_dtype, prim_name=None):
    """Internal function, used to check whether the shape of logits and labels meets the requirements."""
    validator.check_type_name('logits', logits_dtype, [mstype.float16, mstype.float32], prim_name)
    validator.check_type_name('labels', label_dtype, [mstype.int32], prim_name)

    logits_shape_new = (logits_shape[0], *logits_shape[2:])
    msg_prefix = f'For \'{prim_name}\', the' if prim_name else "The"
    if logits_shape_new != label_shape:
        raise ValueError(f"{msg_prefix} shape of 'logits' should be (N, C, d_0, d_1, ...), "
                         f"and the shape of 'labels' should be (N, d_0, d_1, ...), "
                         f"but get 'logits' shape: {logits_shape} and 'labels' shape: {label_shape}")


class NLLLoss(LossBase):
    r"""
    Gets the negative log likelihood loss between logits and labels.

    The nll loss with reduction=none can be described as:

    .. math::

        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\}

    where :math:`x` is the logits, :math:`t` is the labels, :math:`w` is the weight,
    :math:`N` is the batch size, :math:`c` belonging to :math:`[0, C-1]` is class index,
    where :math:`C` is the number of classes.

    If `reduction` is not 'none' (default 'mean'), then

    .. math::

        \ell(x, t)=\left\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean', } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right.

    Args:
        weight (Tensor): The rescaling weight to each class. If the value is not None, the shape is :math:`(C,)`.
            The data type only supports float32 or float16. Default: None.
        ignore_index (int): Specifies a target value that is ignored (typically for padding value)
            and does not contribute to the gradient. Default: -100.
        reduction (str):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, C)`
          or :math:`(N, C, d_1, d_2, ..., d_K)` for :math:`K`-dimensional data, where `C = number of classes`.
          Data type must be float16 or float32. `inputs` needs to be logarithmic probability.
        - **labels** (Tensor) -:math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` for :math:`K`-dimensional data.
          Data type must be int32.

    Returns:
        Tensor, the computed negative log likelihood loss value.

    Raises:
        TypeError: If `weight` is not a Tensor.
        TypeError: If `ignore_index` is not an int.
        TypeError: If the data type of `weight` is not float16 or float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        TypeError: If `logits` is not a Tensor.
        TypeError: If `labels` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:

        >>> logits = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> labels = mindspore.Tensor(np.array([1, 0, 4]), mindspore.int32)
        >>> loss = nn.NLLLoss()
        >>> output = loss(logits, labels)
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(reduction)
        validator.check_value_type('ignore_index', ignore_index, int, self.cls_name)
        if weight is not None:
            validator.check_value_type("weight", weight, [Tensor], self.cls_name)
            validator.check_type_name('weight', weight.dtype, [mstype.float16, mstype.float32], self.cls_name)

        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        _check_nll_loss_inputs(logits.shape, labels.shape, logits.dtype, labels.dtype, self.cls_name)
        return F.nll_loss(logits, labels, self.weight, self.ignore_index, self.reduction)


@constexpr
def _check_cross_entropy_inputs(logits_shape, label_shape,
                                logits_rank, label_rank,
                                logits_dtype, label_dtype,
                                prim_name=None):
    """Internal function, used to check whether the shape of logits and labels meets the requirements."""
    validator.check_type_name('logits', logits_dtype, [mstype.float16, mstype.float32], prim_name)

    msg_prefix = f'For \'{prim_name}\', the' if prim_name else "The"
    if logits_rank == label_rank:
        validator.check_type_name('labels', label_dtype, [mstype.float16, mstype.float32], prim_name)
        if logits_shape != label_shape:
            raise ValueError(f"{msg_prefix} shape of 'logits' should be (N, C, d_0, d_1, ...), "
                             f"and the shape of 'labels' should be (N, C, d_0, d_1, ...), "
                             f"but get 'logits' shape: {logits_shape} and 'labels' shape: {label_shape}.")
    elif label_rank == logits_rank - 1:
        validator.check_type_name('labels', label_dtype, [mstype.int32], prim_name)
        if logits_rank != 1:
            logits_shape_new = (logits_shape[0], *logits_shape[2:])
            if logits_shape_new != label_shape:
                raise ValueError(f"{msg_prefix} shape of 'logits' should be (N, C, d_0, d_1, ...), "
                                 f"and the shape of 'labels' should be (N, d_0, d_1, ...), "
                                 f"but get 'logits' shape: {logits_shape} and 'labels' shape: {label_shape}.")
    else:
        raise ValueError(f"{msg_prefix} rank of 'logits' and 'labels' should be:\n"
                         f"1. 'logits.ndim == labels.ndim' for probabilities, \n"
                         f"2. 'logits.ndim - 1 == labels.ndim' for class indices, \n"
                         f"but get 'logits' rank: {logits_rank} and 'labels' rank: {label_rank}.")


@constexpr
def _cross_entropy_ignore_index_warning(prim_name):
    """Internal function, used to warning when ignore_index > 0 for probabilities."""
    log.warning(f"For \'{prim_name}\', 'ignore_index' does not work when 'labels' is Probability.")


class CrossEntropyLoss(LossBase):
    r"""
    The cross entropy loss between input and target.

    The CrossEntropyLoss support two kind of targets:

    - Class indices (int) in the range :math:`[0, C)` where :math:`C` is the number of classes,
      the loss with reduction=none can be described as:

      .. math::

          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      where :math:`x` is the inputs, :math:`t` is the target, :math:`w` is the weight,
      N is the batch size, :math:`c` belonging to [0, C-1] is class index, where :math:`C` is the number of classes.

      If reduction is not 'none' (default 'mean'), then

      .. math::

          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    - Probabilities (float) for each class, useful when labels beyond a single class per minibatch item
      are required, the loss with reduction=none can be described as:

      .. math::

          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the inputs, :math:`t` is the target, :math:`w` is the weight,
      N is the batch size, :math:`c` belonging to [0, C-1] is class index, where :math:`C` is the number of classes.

      If reduction is not 'none' (default 'mean'), then

      .. math::

          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    Args:
        weight (Tensor): The rescaling weight to each class. If the value is not None, the shape is (C,).
            The data type only supports float32 or float16. Default: None.
        ignore_index (int): Specifies a target value that is ignored (typically for padding value)
            and does not contribute to the gradient. Default: -100.
        reduction (str):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing (float): Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default value: 0.0.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)`,
          where `C = number of classes`. Data type must be float16 or float32.
        - **labels** (Tensor) - For class indices, tensor of shape :math:`()`, :math:`(N)` or
          :math:`(N, d_1, d_2, ..., d_K)` , data type must be int32.
          For probabilities, tensor of shape :math:`(C,)` :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` ,
          data type must be float16 or float32.

    Returns:
        Tensor, the computed cross entropy loss value.

    Raises:
        TypeError: If `weight` is not a Tensor.
        TypeError: If `ignore_index` is not an int.
        TypeError: If the data type of `weight` is not float16 or float32.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        TypeError: If `label_smoothing` is not a float.
        TypeError: If `logits` is not a Tensor.
        TypeError: If `labels` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:

        >>> # Case 1: Indices labels
        >>> inputs = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> target = mindspore.Tensor(np.array([1, 0, 4]), mindspore.int32)
        >>> loss = nn.CrossEntropyLoss()
        >>> output = loss(inputs, target)
        >>> # Case 2: Probability labels
        >>> inputs = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> target = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> loss = nn.CrossEntropyLoss()
        >>> output = loss(inputs, target)
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean',
                 label_smoothing=0.0):
        super().__init__(reduction)
        validator.check_value_type('ignore_index', ignore_index, int, self.cls_name)
        validator.check_value_type('label_smoothing', label_smoothing, float, self.cls_name)
        validator.check_float_range(label_smoothing, 0.0, 1.0, Rel.INC_BOTH, 'label_smoothing', self.cls_name)

        if weight is not None:
            validator.check_value_type("weight", weight, [Tensor], self.cls_name)
            validator.check_type_name('weight', weight.dtype, [mstype.float16, mstype.float32], self.cls_name)

        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        _check_cross_entropy_inputs(logits.shape, labels.shape,
                                    logits.ndim, labels.ndim,
                                    logits.dtype, labels.dtype,
                                    self.cls_name)
        if logits.ndim == labels.ndim and self.ignore_index > 0:
            _cross_entropy_ignore_index_warning(self.cls_name)
        return F.cross_entropy(logits, labels, self.weight, self.ignore_index, self.reduction, self.label_smoothing)


class KLDivLoss(LossBase):
    r"""
    Computes the Kullback-Leibler divergence between the logits and the labels.

    For tensors of the same shape :math:`x` and :math:`target`,
    the updating formulas of KLDivLoss algorithm are as follows,

    .. math::
        L(x, target) = target \cdot (\log target - x)

    Then,

    .. math::
        \ell(x, target) = \begin{cases}
        L(x, target), & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L(x, target)), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L(x, target)) / x.\operatorname{shape}[0], & \text{if reduction} = \text{'batchmean';}\\
        \operatorname{sum}(L(x, target)),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`x` represents `logits`,
    :math:`target` represents `labels`, and
    :math:`\ell(x, target)` represents `output`.

    Note:
        - Currently it does not support float64 input on `Ascend`.
        - The output aligns with the mathematical definition of Kullback-Leibler divergence
          only when `reduction` is set to 'batchmean'.

    Args:
        reduction (str): Specifies the reduction to be applied to the output.
            Default: 'mean'.

            - On Ascend, the value of `reduction` must be one of 'batchmean', 'none' or 'sum'.
            - On GPU, the value of `reduction` must be one of 'mean', 'none' or 'sum'.
            - On CPU, the value of `reduction` must be one of 'mean', 'batchmean', 'none' or 'sum'.

    Inputs:
        - **logits** (Tensor) - The input Tensor. The data type must be float16, float32 or float64.
        - **labels** (Tensor) - The label Tensor which has the same shape and data type as `logits`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `logits`.
        Otherwise, it is a scalar.

    Raises:
        TypeError: If `reduction` is not a str.
        TypeError: If neither `logits` nor `labels` is a Tensor.
        TypeError: If dtype of `logits` or `labels` is not currently supported.
        ValueError: If shape of `logits` is not the same as `labels`.
        RuntimeError: If `logits` or `labels` is a scalar when `reduction` is 'batchmean'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
        >>> labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
        >>> loss = nn.KLDivLoss(reduction='mean')
        >>> output = loss(logits, labels)
        >>> print(output)
        -0.23333333
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        return F.kl_div(logits, labels, self.reduction)


class CTCLoss(LossBase):
    """
    Calculates the CTC (Connectionist Temporal Classification) loss.

    For the CTC algorithm, refer to `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with
    Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ .

    Args:
        blank (int): The blank label. Default: 0.
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', or 'sum'. Default: 'mean'.
        zero_infinity (bool): Whether to set infinite loss and correlation gradient to zero. Default: False.

    Inputs:
        - **log_probs** (Tensor) - A tensor of shape (T, N, C) or (T, C), where T is input length, N is batch size and
          C is number of classes (including blank). T, N and C are positive integers.
        - **targets** (Tensor) - A tensor of shape (N, S) or (sum( `target_lengths` )), where S is max target length,
          means the target sequences.
        - **input_lengths** (Union[tuple, Tensor]) - A tuple or Tensor of shape(N). It means the lengths of the input.
        - **target_lengths** (Union[tuple, Tensor]) - A tuple or Tensor of shape(N). It means the lengths of the target.

    Outputs:
        - **neg_log_likelihood** (Tensor) - A loss value which is differentiable with respect to each input node.

    Raises:
        TypeError: If `log_probs` or `targets` is not a Tensor.
        TypeError: If `zero_infinity` is not a bool, `reduction` is not string.
        TypeError: If the dtype of `log_probs` is not float or double.
        TypeError: If the dtype of `targets`, `input_lengths` or `target_lengths` is not int32 or int64.
        ValueError: If `reduction` is not "none", "mean" or "sum".
        ValueError: If the types of `targets`, `input_lengths` or `target_lengths` are different.
        ValueError: If the value of `blank` is not in range [0, C). C is number of classes of `log_probs` .
        RuntimeError: If any value of `input_lengths` is larger than T. T is length of `log_probs` .
        RuntimeError: If any target_lengths[i] is not in range [0, input_length[i]].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from mindspore.nn.loss import CTCLoss
        >>> T = 5      # Input sequence length
        >>> C = 2      # Number of classes
        >>> N = 2      # Batch size
        >>> S = 3      # Target sequence length of longest target in batch (padding length)
        >>> S_min = 2  # Minimum target length, for demonstration purposes
        >>> arr = np.arange(T*N*C).reshape((T, N, C))
        >>> ms_input = Tensor(arr, dtype=mstype.float32)
        >>> input_lengths = np.full(shape=(N), fill_value=T)
        >>> input_lengths = Tensor(input_lengths, dtype=mstype.int32)
        >>> target_lengths = np.full(shape=(N), fill_value=S_min)
        >>> target_lengths = Tensor(target_lengths, dtype=mstype.int32)
        >>> target = np.random.randint(1, C, size=(N, S))
        >>> target = Tensor(target, dtype=mstype.int32)
        >>> ctc_loss = CTCLoss(blank=0, reduction='none', zero_infinity=False)
        >>> loss = ctc_loss(ms_input, target, input_lengths, target_lengths)
        >>> print(loss)
        [-45.79497  -55.794968]
        >>> arr = np.arange(T*C).reshape((T, C))
        >>> ms_input = Tensor(arr, dtype=mstype.float32)
        >>> input_lengths = Tensor([T], dtype=mstype.int32)
        >>> target_lengths = Tensor([S_min], dtype=mstype.int32)
        >>> target = np.random.randint(1, C, size=(S_min,))
        >>> target = Tensor(target, dtype=mstype.int32)
        >>> ctc_loss = CTCLoss(blank=0, reduction='none', zero_infinity=False)
        >>> loss = ctc_loss(ms_input, target, input_lengths, target_lengths)
        >>> print(loss)
        [-25.794968]
    """

    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def construct(self, log_probs, targets, input_lengths, target_lengths):
        _check_is_tensor('log_probs', log_probs, self.cls_name)
        _check_is_tensor('targets', targets, self.cls_name)
        if log_probs.ndim == 2:
            log_probs = log_probs.expand_dims(-2)
            targets = targets.expand_dims(0)
        neg_log_hood, _ = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction,
                                     self.zero_infinity)
        return neg_log_hood


class GaussianNLLLoss(LossBase):
    r"""
    Gaussian negative log likelihood loss.

    The target values are considered to be samples from a Gaussian distribution, where the expectation and variance are
    predicted by a neural network. For `labels` modeled on a Gaussian distribution, `logits` to record expectations,
    and the variance `var` (elements are all positive), the calculated loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{logits} - \text{labels}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where `eps` is used for stability of :math:`log`. When :math:`full=True`, a constant will be added to the loss. If
    the shape of :math:`var` and `logits` are not the same (due to a homoscedastic assumption), their shapes must allow
    correct broadcasting.

    Args:
        full (bool): Whether include the constant term in the loss calculation. When :math:`full=True`,
             the constant term `const.` will be :math:`0.5 * log(2\pi)`. Default: False.
        eps (float): Used to improve the stability of log function. Default: 1e-6.
        reduction (str): Apply specific reduction method to the output: 'none', 'mean', or 'sum'. Default: 'mean'.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` or :math:`(*)` where :math:`*` means any number of
          additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)` or :math:`(*)`, same shape as the logits, or same shape
          as the logits but with one dimension equal to 1 (to allow for broadcasting).
        - **var** - Tensor of shape :math:`(N, *)` or :math:`(*)`, same shape as logits, or same shape as the logits
          but with one dimension equal to 1, or same shape as the logits but with one fewer dimension
          (to allow for broadcasting).

    Returns:
        Tensor or Tensor scalar, the computed loss depending on `reduction`.

    Raises:
        TypeError: If `logits` is not a Tensor.
        TypeError: If `labels` is not a Tensor.
        TypeError: If `full` is not a bool.
        TypeError: If `eps` is not a float.
        ValueError: If `eps` is not a float within [0, inf).
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> import mindspore.common.dtype as mstype
        >>> arr1 = np.arange(8).reshape((4, 2))
        >>> arr2 = np.array([2, 3, 1, 4, 6, 4, 4, 9]).reshape((4, 2))
        >>> logits = Tensor(arr1, mstype.float32)
        >>> labels = Tensor(arr2, mstype.float32)
        >>> loss = nn.GaussianNLLLoss(reduction='mean')
        >>> var = Tensor(np.ones((4, 1)), mstype.float32)
        >>> output = loss(logits, labels, var)
        >>> print(output)
        1.4374993

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """

    def __init__(self, *, full=False, eps=1e-6, reduction='mean'):
        super(GaussianNLLLoss, self).__init__()
        validator.check_float_range(eps, 0, float('inf'), Rel.INC_NEITHER, "eps", self.cls_name)
        validator.check_value_type('full', full, [bool], self.cls_name)
        validator.check_string(reduction, ['none', 'mean', 'sum'], 'reduction', 'gaussian_nll_loss')
        self.full = full
        self.eps = eps
        self.reduction = reduction

    def construct(self, logits, labels, var):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        _check_is_tensor('var', var, self.cls_name)
        return ops.gaussian_nll_loss(logits, labels, var, self.full, self.eps, self.reduction)


class HingeEmbeddingLoss(LossBase):
    r"""
    Hinge Embedding Loss. Compute the output according to the input elements. Measures the loss given an input tensor x
    and a labels tensor y (containing 1 or -1).
    This is usually used for measuring the similarity between two inputs.

    The loss function for :math:`n`-th sample in the mini-batch is

    .. math::
        l_n = \begin{cases}
        x_n, & \text{if}\; y_n = 1,\\
        \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}

    and the total loss functions is

    .. math::
        \ell(x, y) = \begin{cases}
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`L = \{l_1,\dots,l_N\}^\top`.

    Args:
        margin (float): Threshold defined by Hinge Embedding Loss :math:`margin`.
            Represented as :math:`\Delta` in the formula. Default: 1.0.
        reduction (str): Specify the computing method to be applied to the outputs: 'none', 'mean', or 'sum'.
            Default: 'mean'.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(*)` where :math:`*` means any number of dimensions.
        - **labels** (Tensor) - Same shape as the logits, contains -1 or 1.

    Returns:
        Tensor or Tensor scalar, the computed loss depending on `reduction`.

    Raises:
        TypeError: If `logits` is not a Tensor.
        TypeError: If `labels` is not a Tensor.
        TypeError: If `margin` is not a float.
        ValueError: If `labels` does not have the same shape as `logits`.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examplse:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> import mindspore.common.dtype as mstype
        >>> arr1 = np.array([0.9, -1.2, 2, 0.8, 3.9, 2, 1, 0, -1]).reshape((3, 3))
        >>> arr2 = np.array([1, 1, -1, 1, -1, 1, -1, 1, 1]).reshape((3, 3))
        >>> logits = Tensor(arr1, mstype.float32)
        >>> labels = Tensor(arr2, mstype.float32)
        >>> loss = nn.HingeEmbeddingLoss(reduction='mean')
        >>> output = loss(logits, labels)
        >>> print(output)
        0.16666667
    """

    def __init__(self, margin=1.0, reduction='mean'):
        super(HingeEmbeddingLoss, self).__init__()
        validator.check_value_type('margin', margin, [float], self.cls_name)
        validator.check_string(reduction, ['none', 'sum', 'mean'], 'reduction', self.cls_name)
        self.margin = margin
        self.reduction = reduction

    def construct(self, logits, labels):
        loss = ops.hinge_embedding_loss(logits, labels, self.margin, self.reduction)
        return loss
