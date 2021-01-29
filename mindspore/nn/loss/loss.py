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
from mindspore.ops.primitive import constexpr
from mindspore.ops import _selected_ops
from mindspore.nn.cell import Cell
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
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

        self.reduce_mean = _selected_ops.ReduceMean()
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
    L1Loss creates a criterion to measure the mean absolute error (MAE) between :math:`x` and :math:`y` element-wise,
    where :math:`x` is the input Tensor and :math:`y` is the target Tensor.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        L(x, y) = \{l_1,\dots,l_N\}, \quad \text{with } l_n = \left| x_n - y_n \right|

    When argument reduction is 'mean', the mean value of :math:`L(x, y)` will be returned.
    When argument reduction is 'sum', the sum of :math:`L(x, y)` will be returned. :math:`N` is the batch size.

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`. The data type must be float16 or
          float32.
        - **target_data** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`. The data type must be float16 or
          float32.

    Outputs:
        Tensor, loss float tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> loss = nn.L1Loss()
        >>> input_data = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> target_data = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(input_data, target_data)
        >>> print(output)
        0.33333334
    """
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__(reduction)
        self.abs = P.Abs()

    def construct(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x)


class MSELoss(_Loss):
    r"""
    MSELoss creates a criterion to measure the mean squared error (squared L2-norm) between :math:`x` and :math:`y`
    element-wise, where :math:`x` is the input and :math:`y` is the target.

    For simplicity, let :math:`x` and :math:`y` be 1-dimensional Tensor with length :math:`N`,
    the unreduced loss (i.e. with argument reduction set to 'none') of :math:`x` and :math:`y` is given as:

    .. math::
        L(x, y) = \{l_1,\dots,l_N\}, \quad \text{with} \quad l_n = (x_n - y_n)^2.

    When argument reduction is 'mean', the mean value of :math:`L(x, y)` will be returned.
    When argument reduction is 'sum', the sum of :math:`L(x, y)` will be returned. :math:`N` is the batch size.

    Args:
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **target_data** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, weighted loss float tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> loss = nn.MSELoss()
        >>> input_data = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> target_data = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(input_data, target_data)
        >>> print(output)
        0.33333334
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
        \frac{0.5 (x_i - y_i)^{2}}{\text{beta}}, & \text{if } |x_i - y_i| < \text{beta} \\
        |x_i - y_i| - 0.5 \text{beta}, & \text{otherwise. }
        \end{cases}

    Here :math:`\text{beta}` controls the point where the loss function changes from quadratic to linear.
    Its default value is 1.0. :math:`N` is the batch size. This function returns an
    unreduced loss Tensor.

    Args:
        beta (float): A parameter used to control the point where the function will change from
            quadratic to linear. Default: 1.0.

    Inputs:
        - **input_data** (Tensor) - Tensor of shape :math:`(x_1, x_2, ..., x_R)`.
        - **target_data** (Tensor) - Tensor of shape :math:`(y_1, y_2, ..., y_S)`.

    Outputs:
        Tensor, loss float tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = nn.SmoothL1Loss()
        >>> input_data = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> target_data = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(input_data, target_data)
        >>> print(output)
        [0.  0.  0.5]
    """
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.smooth_l1_loss = P.SmoothL1Loss(self.beta)

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

    For each instance :math:`N_i`, the loss is given as:

    .. math::
        \ell(x_i, t_i) = - \log\left(\frac{\exp(x_{t_i})}{\sum_j \exp(x_j)}\right)
        =  -x_{t_i} + \log\left(\sum_j \exp(x_j)\right)

    where :math:`x_i` is a 1D score Tensor, :math:`t_i` is a scalar.

    Note:
        While the target classes are mutually exclusive, i.e., only one class is positive in the target, the predicted
        probabilities need not to be exclusive. It is only required that the predicted probability distribution
        of entry is a valid one.

    Args:
        sparse (bool): Specifies whether labels use sparse format or not. Default: False.
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C).
        - **labels** (Tensor) - Tensor of shape (N, ). If `sparse` is True, The type of
          `labels` is mindspore.int32. If `sparse` is False, the type of `labels` is the same as the type of `logits`.

    Outputs:
        Tensor, a tensor of the same shape as logits with the component-wise
        logistic losses.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> np.random.seed(0)
        >>> logits = Tensor(np.random.randint(0, 9, [1, 10]), mindspore.float32)
        >>> labels_np = np.ones([1,]).astype(np.int32)
        >>> labels = Tensor(labels_np)
        >>> output = loss(logits, labels)
        >>> print(output)
        [7.868383]
    """
    def __init__(self,
                 sparse=False,
                 reduction='none'):
        super(SoftmaxCrossEntropyWithLogits, self).__init__(reduction)
        self.sparse = sparse
        self.reduction = reduction
        self.softmax_cross_entropy = _selected_ops.SoftmaxCrossEntropyWithLogits()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0., mstype.float32)
        self.is_cpugpu = context.get_context('device_target') in ["CPU", "GPU"]
        self.sparse_softmax_cross_entropy = P.SparseSoftmaxCrossEntropyWithLogits()

    def construct(self, logits, labels):
        if self.sparse:
            if self.reduction == 'mean':
                x = self.sparse_softmax_cross_entropy(logits, labels)
                return x
            labels = self.one_hot(labels, F.shape(logits)[-1], self.on_value, self.off_value)
        x = self.softmax_cross_entropy(logits, labels)[0]
        return self.get_loss(x)

@constexpr
def _check_label_dtype(labels_dtype, cls_name):
    validator.check_type_name("labels", labels_dtype, [mstype.int32, mstype.int64], cls_name)


class SampledSoftmaxLoss(_Loss):
    r"""
    Computes the sampled softmax training loss.

    Args:
        num_sampled (int): The number of classes to randomly sample per batch.
        num_classes (int): The number of possible classes.
        num_true (int): The number of target classes per training example.
        sampled_values (Tuple):  Tuple of (`sampled_candidates`, `true_expected_count`,
            `sampled_expected_count`) returned by a `*CandidateSampler` function.
            Default to None, `UniformCandidateSampler` is applied.
        remove_accidental_hits (bool): Whether to remove "accidental hits"
            where a sampled class equals one of the target classes.  Default is True.
        seed (int): Random seed for candidate sampling. Default: 0
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".

    Inputs:
        - **weights** (Tensor) - Tensor of shape (C, dim).
        - **bias** (Tensor) - Tensor of shape (C).  The class biases.
        - **labels** (Tensor) - Tensor of shape (N, num_true), type `int64, int32`. The
          target classes.
        - **inputs** (Tensor) - Tensor of shape (N, dim). The forward activations of
          the input network.

    Outputs:
        Tensor, a tensor of shape (N) with the per-example sampled softmax losses.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> mindspore.set_seed(1)
        >>> loss = nn.SampledSoftmaxLoss(num_sampled=4, num_classes=7, num_true=1)
        >>> weights = Tensor(np.random.randint(0, 9, [7, 10]), mindspore.float32)
        >>> biases = Tensor(np.random.randint(0, 9, [7]), mindspore.float32)
        >>> labels = Tensor([0, 1, 2])
        >>> inputs = Tensor(np.random.randint(0, 9, [3, 10]), mindspore.float32)
        >>> output = loss(weights, biases, labels, inputs)
        >>> print(output)
        [4.6051701e+01 1.4000047e+01 6.1989022e-06]
    """

    def __init__(self, num_sampled, num_classes, num_true=1,
                 sampled_values=None, remove_accidental_hits=True, seed=0,
                 reduction='none'):
        super(SampledSoftmaxLoss, self).__init__(reduction)

        if num_true < 1:
            raise ValueError(f"num_true {num_true} is less than 1.")
        if seed < 0:
            raise ValueError(f"seed {seed} is less than 0.")
        if num_sampled > num_classes:
            raise ValueError(f"num_sampled {num_sampled} is great than num_classes {num_classes}.")
        if num_true > num_classes:
            raise ValueError(f"num_true {num_true} is great than num_classes {num_classes}.")
        if sampled_values is not None:
            if not isinstance(sampled_values, (list, tuple)):
                raise TypeError(f"sampled_values {sampled_values} is not a list.")
            if len(sampled_values) != 3:
                raise ValueError(f"sampled_values size {len(sampled_values)} is not 3.")

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

    def construct(self, weights, biases, labels, inputs):
        _check_label_dtype(self.dtype(labels), self.cls_name)

        logits, labels = self._compute_sampled_logits(
            weights=weights,
            biases=biases,
            labels=labels,
            inputs=inputs,
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
                                inputs,
                                num_true=1,
                                sampled_values=None,
                                subtract_log_q=True):
        """Helper function for SampledSoftmaxLoss functions.

        Computes sampled output training logits and labels suitable

        Note: In the case where num_true > 1, we assign to each target class
        the target probability 1 / num_true so that the target probabilities
        sum to 1 per-example.

        Args:
            weights (Tensor): Tensor of shape `[num_classes, dim]`.
            biases (Tensor): Tensor of shape `[num_classes]`.
            labels (Tensor): Tensor of shape `[batch_size, num_true]`. The target classes.
            inputs (Tensor): Tensor of shape `[batch_size, dim]`.  The forward
                activations of the input network.
            num_true (int): The number of target classes per training example.
            sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `UniformCandidateSampler` function.
            subtract_log_q: A `bool`.  whether to subtract the log expected count of
                the labels in the sample to get the logits of the true labels.
                Default is True.
        Returns:
            out_logits: `Tensor` object with shape
                `[batch_size, num_true + num_sampled]`
            out_labels: A Tensor object with the same shape as `out_logits`.
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

        # true_w shape is [batch_size * num_true, dim]
        true_w = self.slice_op(all_w, [0, 0], [n_true, n_dim])
        sampled_w = self.slice_op(all_w, [n_true, 0], [n_sampled, n_dim])
        sampled_logits = self.matmul(inputs, sampled_w)

        all_b = self.gather_v2(biases, all_ids, 0)
        true_b = self.slice_op(all_b, [0], [n_true])
        sampled_b = self.slice_op(all_b, [n_true], [n_sampled])

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        new_true_w_shape = (-1, num_true, n_dim)
        row_wise_dots = self.mul(self.expand_dims(inputs, 1),
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


class BCELoss(_Loss):
    r"""
    BCELoss creates a criterion to measure the binary cross entropy between the true labels and predicted labels.

    Set the predicted labels as :math:`x`, true labels as :math:`y`, the output loss as :math:`\ell(x, y)`.
    Let,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{`none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    Note:
        Note that the predicted labels should always be the output of sigmoid and the true labels should be numbers
        between 0 and 1.

    Args:
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            And it must have same shape and data type as `inputs`. Default: None
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.

    Inputs:
        - **inputs** (Tensor) - The input Tensor. The data type must be float16 or float32.
        - **labels** (Tensor) - The label Tensor which has same shape and data type as `inputs`.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `inputs`.
        Otherwise, the output is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> weight = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 3.3, 2.2]]), mindspore.float32)
        >>> loss = nn.BCELoss(weight=weight, reduction='mean')
        >>> inputs = Tensor(np.array([[0.1, 0.2, 0.3], [0.5, 0.7, 0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]), mindspore.float32)
        >>> output = loss(inputs, labels)
        >>> print(output)
        1.8952923
    """

    def __init__(self, weight=None, reduction='none'):
        super(BCELoss, self).__init__()
        self.binary_cross_entropy = P.BinaryCrossEntropy(reduction=reduction)
        self.weight_one = weight is None
        if not self.weight_one:
            self.weight = weight
        else:
            self.ones = P.OnesLike()

    def construct(self, inputs, labels):
        if self.weight_one:
            weight = self.ones(inputs)
        else:
            weight = self.weight
        loss = self.binary_cross_entropy(inputs, labels, weight)
        return loss


@constexpr
def _check_reduced_shape_valid(ori_shape, reduced_shape, axis, cls_name):
    validator.check_reduce_shape(ori_shape, reduced_shape, axis, cls_name)

class CosineEmbeddingLoss(_Loss):
    r"""
    Computes the similarity between two tensors using cosine distance.

    Given two tensors `x1`, `x2`, and a Tensor label `y` with values 1 or -1:

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
        - **input_x1** (Tensor) - Input tensor.
        - **input_x2** (Tensor) - Its shape and data type must be the same as `input_x1`'s shape and data type.
        - **y** (Tensor) - Contains value 1 or -1. Suppose the shape of `input_x1` is
          :math:`(x_1, x_2, x_3,..., x_R)`, then the shape of `target` must be :math:`(x_1, x_3, x_4, ..., x_R)`.

    Outputs:
        - **loss** (Tensor) - If `reduction` is "none", its shape is the same as `y`'s shape, otherwise a scalar value
          will be returned.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]), mindspore.float32)
        >>> x2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]), mindspore.float32)
        >>> y = Tensor(np.array([1,-1]), mindspore.int32)
        >>> cosine_embedding_loss = nn.CosineEmbeddingLoss()
        >>> output = cosine_embedding_loss(x1, x2, y)
        >>> print(output)
        0.0003426075
    """
    def __init__(self, margin=0.0, reduction="mean"):
        super(CosineEmbeddingLoss, self).__init__(reduction)
        self.reduce_sum = P.ReduceSum()
        self.maximum = P.Maximum()
        validator.check_value_type("margin", margin, [float], self.cls_name)
        self.margin = validator.check_float_range(margin, -1.0, 1.0, Rel.INC_BOTH, "margin", self.cls_name)

    def construct(self, x1, x2, y):
        F.same_type_shape(x1, x2)
        _check_reduced_shape_valid(F.shape(x1), F.shape(y), (1,), self.cls_name)
        # if target > 0, 1-cosine(x1, x2)
        # else, max(0, cosine(x1, x2)-margin)
        prod_sum = self.reduce_sum(x1 * x2, (1,))
        square1 = self.reduce_sum(F.square(x1), (1,))
        square2 = self.reduce_sum(F.square(x2), (1,))
        denom = F.sqrt(square1 * square2)
        cosine = prod_sum / denom

        pos_value = 1.0 - cosine
        neg_value = self.maximum(cosine - self.margin, 0.0)
        zeros = F.zeros_like(cosine)
        pos_part = F.select(y == 1, pos_value, zeros)
        neg_part = F.select(y == -1, neg_value, zeros)
        output_unreduced = pos_part + neg_part

        return self.get_loss(output_unreduced)
