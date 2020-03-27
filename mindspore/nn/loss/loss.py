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
"""loss"""
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from ... import context


class _Loss(Cell):
    """
    Base class for other losses.
    """
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x):
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))
        return x

    def construct(self, base, target):
        raise NotImplementedError


class L1Loss(_Loss):
    r"""
    L1Loss creates a criterion to measure the mean absolute error (MAE) between :math:`x` and :math:`y` by element,
    where :math:`x` is the input Tensor and :math:`y` is the target Tensor.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        L(x, y) = \{l_1,\dots,l_N\}, \quad \text{with } l_n = \left| x_n - y_n \right|

    When argument reduction is 'mean', the mean value of :math:`L(x, y)` will be returned.
    When argument reduction is 'sum', the sum of :math:`L(x, y)` will be returned. :math:`N` is the batch size.

    Args:
        reduction (str): Type of reduction to apply to loss. The optional values are "mean", "sum", "none".
            Default: "mean".

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **target_data** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, loss float tensor.

    Examples:
        >>> loss = L1Loss()
        >>> input_data = Tensor(np.array([1, 2, 3]), mstype.float32)
        >>> target_data = Tensor(np.array([1, 2, 2]), mstype.float32)
        >>> loss(input_data, target_data)
    """
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__(reduction)
        self.abs = P.Abs()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x)


class MSELoss(_Loss):
    r"""
    MSELoss create a criterion to measures the mean squared error (squared L2-norm) between :math:`x` and :math:`y`
    by element, where :math:`x` is the input and :math:`y` is the target.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        L(x, y) = \{l_1,\dots,l_N\}, \quad \text{with} \quad l_n = (x_n - y_n)^2.

    When argument reduction is 'mean', the mean value of :math:`L(x, y)` will be returned.
    When argument reduction is 'sum', the sum of :math:`L(x, y)` will be returned. :math:`N` is the batch size.

    Args:
        reduction (str): Type of reduction to apply to loss. The optional values are "mean", "sum", "none".
            Default: "mean".

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **target_data** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, weighted loss float tensor.

    Examples:
        >>> loss = MSELoss()
        >>> input_data = Tensor(np.array([1, 2, 3]), mstype.float32)
        >>> target_data = Tensor(np.array([1, 2, 2]), mstype.float32)
        >>> loss(input_data, target_data)
    """
    def construct(self, base, target):
        x = F.square(base - target)
        return self.get_loss(x)


class SmoothL1Loss(_Loss):
    r"""
    A loss class for learning region proposals.

    SmoothL1Loss can be regarded as modified version of L1Loss or a combination of L1Loss and L2Loss.
    L1Loss computes the element-wise absolute difference between two input Tensor while L2Loss computes the
    squared difference between two input Tensor. L2Loss often leads to faster convergence but it is less
    robust to outliers.

    Given two input :math:`x,\  y` of length :math:`N`, the unreduced SmoothL1Loss can be described
    as follows:

    .. math::
        L_{i} =
        \begin{cases}
        0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < \text{sigma}; \\
        |x_i - y_i| - 0.5, & \text{otherwise. }
        \end{cases}

    Here :math:`\text{sigma}` controls the point where the loss function changes from quadratic to linear.
    Its default value is 1.0. :math:`N` is the batch size. This function returns an
    unreduced loss Tensor.

    Args:
        sigma (float): A parameter used to control the point where the function will change from
            quadratic to linear. Default: 1.0.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **target_data** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, loss float tensor.

    Examples:
        >>> loss = SmoothL1Loss()
        >>> input_data = Tensor(np.array([1, 2, 3]), mstype.float32)
        >>> target_data = Tensor(np.array([1, 2, 2]), mstype.float32)
        >>> loss(input_data, target_data)
    """
    def __init__(self, sigma=1.0):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.smooth_l1_loss = P.SmoothL1Loss(self.sigma)

    def construct(self, base, target):
        return self.smooth_l1_loss(base, target)


class SoftmaxCrossEntropyWithLogits(_Loss):
    r"""
    Computes softmax cross entropy between logits and labels.

    Measures the distribution error between the probabilities of the input (computed with softmax function) and the
    target where the classes are mutually exclusive (only one class is positive) using cross entropy loss.

    Typical input into this function is unnormalized scores and target of each class.
    Scores Tensor :math:`x` is of shape :math:`(N, C)` and target Tensor :math:`t` is a
    Tensor of shape :math:`(N, C)` which contains one-hot labels of length :math:`C`.

    For each batch :math:`N_i`, the loss is given as:

    .. math::
        \ell(x_i, t_i) = -w_{t_i} \log\left(\frac{\exp(x_{t_i})}{\sum_j \exp(x_j)}\right)
        = w_{t_i} \left(-x_{t_i} + \log\left(\sum_j \exp(x_i)\right)\right),
    where :math:`x_i` is a 1D score Tensor, :math:`t_i` is the target class and
    :math:`w` is a weight Tensor to generate weighted loss for each class. When not specified,
    weight Tensor is set to be None and weight is the same (:math:`1`) for all class.

    Note:
        While the target classes are mutually exclusive, i.e., only one class is positive in the target, the predicted
        probabilities need not be exclusive. All that is required is that the predicted probability distribution
        of entry is a valid one.

    Args:
        is_grad (bool): Specifies whether calculate grad only. Default: True.
        sparse (bool): Specifies whether labels use sparse format or not. Default: False.
        reduction (Union[str, None]): Type of reduction to apply to loss. Support 'sum' or 'mean' If None,
            do not reduction. Default: None.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **labels** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`. If `sparse` is True, The type of
          `labels` is mstype.int32. If `sparse` is False, the type of `labels` is same as the type of `logits`.

    Outputs:
        Tensor, a tensor of the same shape as logits with the component-wise
        logistic losses.

    Examples:
        >>> loss = SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> logits = Tensor(np.random.randint(0, 9, [1, 10]), mstype.float32)
        >>> labels_np = np.zeros([1, 10]).astype(np.int32)
        >>> labels_np[0][0] = 1
        >>> labels = Tensor(labels_np)
        >>> loss(logits, labels)
    """
    def __init__(self,
                 is_grad=True,
                 sparse=False,
                 reduction=None):
        super(SoftmaxCrossEntropyWithLogits, self).__init__(reduction)
        self.is_grad = is_grad
        self.sparse = sparse
        self.softmax_cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.is_cpugpu = context.get_context('device_target') in ["CPU", "GPU"]

        if self.is_cpugpu:
            self.sparse_softmax_cross_entropy = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=self.is_grad)

    def construct(self, logits, labels):
        if self.is_cpugpu and self.sparse:
            x = self.sparse_softmax_cross_entropy(logits, labels)
            return x

        if self.sparse:
            labels = self.one_hot(labels, F.shape(logits)[-1], self.on_value, self.off_value)
        x = self.softmax_cross_entropy(logits, labels)[0]
        return self.get_loss(x)


class SoftmaxCrossEntropyExpand(Cell):
    r"""
    Computes softmax cross entropy between logits and labels. Implemented by expanded formula.

    This is a wrapper of several functions.

    .. math::
        \ell(x_i, t_i) = -log\left(\frac{\exp(x_{t_i})}{\sum_j \exp(x_j)}\right),
    where :math:`x_i` is a 1D score Tensor, :math:`t_i` is the target class.

    Note:
        When argument sparse is set to True, the format of label is the index
        range from :math:`0` to :math:`C - 1` instead of one-hot vectors.

    Args:
        sparse(bool): Specifies whether labels use sparse format or not. Default: False.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **label** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, a scalar tensor including the mean loss.

    Examples:
        >>> loss = SoftmaxCrossEntropyExpand(sparse=True)
        >>> input_data = Tensor(np.ones([64, 512]), dtype=mstype.float32)
        >>> label = Tensor(np.ones([64]), dtype=mstype.int32)
        >>> loss(input_data, label)
    """
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.reduce_max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()

    def construct(self, logit, label):
        logit_max = self.reduce_max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.reduce_sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_array(-1.0), loss)
        loss = self.reduce_mean(loss, -1)

        return loss
