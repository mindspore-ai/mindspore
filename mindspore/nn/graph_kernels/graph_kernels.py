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
"""
Graph kernels. They are composites of basic primitives and can be compiled into
a fused kernel automaticly when context.set_context(enable_graph_kernel=True).
"""
from ...common import dtype as mstype
from ...ops import operations as P
from ...ops.primitive import PrimitiveWithInfer, prim_attr_register
from ...ops.composite import multitype_ops as C
from ...ops.operations import _grad_ops as G
from ..._checkparam import ParamValidator as validator
from ..cell import Cell, GraphKernel


class InplaceAssign(PrimitiveWithInfer):
    """
    Inplace assign `Parameter` with a value.

    This primitive can only use in graph kernel.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.
        - **value** (Tensor) - The value to assign.
        - **depend** (Tensor) - The dependent tensor to keep this op connected in graph.

    Outputs:
        Tensor, has the same type as original `variable`.

    Examples:
    >>> def construct(self, x):
    >>> val = x - 1.0
    >>> ret = x + 2.0
    >>> return InplaceAssign()(x, val, ret)
    >>> x = Tensor([2.0], mindspore.float32)
    >>> net = Net()
    >>> net(x)
   """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x', 'y', 'z'], outputs=['output'])

    def infer_shape(self, x, y, z):
        return z

    def infer_dtype(self, x, y, z):
        return z

    def get_bprop(self):
        def bprop(x, y, z, out, dout):
            return (x, C.zeros_like(y), dout)
        return bprop


class MaximumGrad(GraphKernel):
    """

    Backprop function for Maximum operator.

    Inputs:
        - **x** (Tensor) - The first input tensor of maximum.
        - **y** (Tensor) - The second input tensor of maximum.
        - **dout** (Tensor) - has the same shape as x and y, next operator's backprop output.

    Outputs:
        dx (Tensor): has the same shape as x and y, returns dout element if
        `x >= y` returns true at the same position, or returns zero at that
        position
        dy (Tensor): has the same shape as x and y, dy = dout - dx

    Examples:
        >>> layer = MaximumGrad()
        >>> output = layer(Tensor([1,2,3], [3, 2, 1], [4, 5, 6]))
    """

    def __init__(self, grad_x=True, grad_y=True):
        super(MaximumGrad, self).__init__()
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.select = P.Select()
        self.greater_equal = P.GreaterEqual()
        self.zeros_like = P.ZerosLike()
        self.sub = P.Sub()

    def construct(self, x, y, dout):
        cmp_result = self.greater_equal(x, y)
        dx = self.select(cmp_result, dout, self.zeros_like(dout))
        dy = dout - dx

        return dx, dy


class MinimumGrad(GraphKernel):
    """
    Backprop function for Minimum operator.

    Compares x and y elementwise, dout should has the same shape with x and y.

    Inputs:
        - **x** (Tensor) - The first input
        - **y** (Tensor) - x and y should have same shape
        - **dout** (Tensor) - Has the same shape as x and y, next operator's backprop output

    Outputs:
        - dx (Tensor) - Has the same shape as x and y, returns dout element if
        `x <= y` returns true at the same position, or returns zero at that
        position
        - dy (Tensor) - Has the same shape as x and y, dy = dout - dx

    Examples:
        >>> layer = MinimumGrad()
        >>> output = layer(Tensor([1,2,3], [3, 2, 1], [4, 5, 6]))
    """

    def __init__(self, grad_x=True, grad_y=True):
        super(MinimumGrad, self).__init__()
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.select = P.Select()
        self.less_equal = P.LessEqual()
        self.zeros_like = P.ZerosLike()
        self.sub = P.Sub()

    def construct(self, x, y, dout):
        cmp_result = self.less_equal(x, y)
        dx = self.select(cmp_result, dout, self.zeros_like(dout))
        # dy = self.select(cmp_result, self.zeros_like(dout), dout)
        dy = dout - dx

        return dx, dy


class AbsGrad(GraphKernel):
    """
    Abs's backprop function.

    Inputs:
        **input_x** (Tensor) - input data of this operator.
        **dout** (Tensor) - output of the next operator's backprop function.

    Outputs:
        Tensor, has the same shape as input_x.

    Examples:
        >>> back = AbsGrad()
        >>> output = back(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
    """

    def __init__(self):
        super(AbsGrad, self).__init__()
        self.mul = P.Mul()
        self.abs = P.Abs()
        self.add = P.TensorAdd()
        self.div = P.RealDiv()
        self.round = P.Round()

    def construct(self, input_x, dout):
        NUM_MAX = 32768
        mul_max = self.mul(input_x, P.Fill()(P.DType()(input_x), (1,), NUM_MAX))
        res_abs = self.abs(mul_max)
        res_div = self.div(mul_max, res_abs)
        res_round = self.round(res_div)
        res = self.mul(res_round, dout)
        return res


class ApplyMomentum(GraphKernel):
    """
    Update parameter according to the ApplyMomentum algorithm.

    Inputs:
        variable (Tensor): mutable tensor var
        accumulation (Tensor): mutable tensor accum
        learning_rate (float32): learning rate
        gradient (float32): The gradient
        momentum (float32): Momentum

    Outputs: updated accumulation and variable
    """

    def __init__(self,
                 use_nesterov=False,
                 use_locking=False,
                 gradient_scale=1.0):
        super(ApplyMomentum, self).__init__()
        self.gradient_scale = validator.check_type('gradient_scale', gradient_scale, [float])
        self.fake_output_assign_1 = InplaceAssign()
        self.fake_output_assign_1.add_prim_attr("fake_output", True)
        self.fake_output_assign_2 = InplaceAssign()
        self.fake_output_assign_2.add_prim_attr("fake_output", True)

    def construct(self, variable, accumulation, learning_rate, gradient, momentum):
        gradient = gradient * self.gradient_scale
        momt_accumulation = accumulation * momentum
        accumulation_inplace = momt_accumulation + gradient

        sum_gradient = accumulation_inplace * learning_rate
        variable_inplace = variable - sum_gradient

        accumulation_inplace = self.fake_output_assign_1(accumulation, accumulation_inplace, accumulation_inplace)
        variable_inplace = self.fake_output_assign_2(variable, variable_inplace, variable_inplace)
        return accumulation_inplace, variable_inplace


class BiasAdd(GraphKernel):
    """
    Return the sum of x and bias.

    Inputs:
        x (Tensor): Tensor of input data.
        bias (Tensor): The bias tensor.

    Output:
        Tensor, the sum of x and bias.

    Example:
    >>> layer = BiasGrad()
    >>> output = BiasAdd(Tensor([1, 2, 3]), Tensor([1,]))
    """

    def __init__(self):
        super(BiasAdd, self).__init__()

    def construct(self, x, bias):
        shape = P.Shape()(x)
        if len(shape) == 4:
            bias_shape = (1, P.Shape()(bias)[0], 1, 1)  # NCHW
        else:
            bias_shape = (1, P.Shape()(bias)[0])
        res = x + P.Reshape()(bias, bias_shape)
        return res

class BiasAddGrad(GraphKernel):
    """
    Computes gradients of BiasAdd.

    Inputs:
        x (Tensor): the gradients of bias add output.

    Output:
        Tensor, the gradients of bias add input.

    Examples:
        >>> dout = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
        >>> bias_add_grad = BiasAddGrad()
        >>> dx = bias_add_grad(dout)
    """
    def __init__(self):
        super(BiasAddGrad, self).__init__()

    def construct(self, x):
        shape_x = P.Shape()(x)
        reduce_axis = [0]
        for i in range(2, len(shape_x)):
            reduce_axis.append(i)

        res = P.ReduceSum()(x, reduce_axis)
        return res


class EqualCount(GraphKernel):
    """
    Computes the number of the same elements of two tensors.

    The two input tensors should have same shape and data type.

    Inputs:
        x (Tensor): the first input tensor.
        y (Tensor): the second input tensor.

    Outputs:
        Tensor, the type is same as input tensor and size as (1,).

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> equal_count = EqualCount()
        >>> equal_count(x, y)
    """
    def __init__(self):
        super(EqualCount, self).__init__()

    def construct(self, x, y):
        equal_bool = P.Equal()(P.Cast()(x, mstype.float32), P.Cast()(y, mstype.float32))
        equal_count = P.Cast()(equal_bool, mstype.float16)

        axes = (0,)
        res = P.ReduceSum()(equal_count, axes)
        res = P.Cast()(res, P.DType()(x))
        return res


class ReduceMean(GraphKernel):
    """
    Reduce a dimension of a tensor by averaging all elements in the dimension.

    The dtype of the tensor to be reduced is number.

    Args:
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
                          If False, don't keep these dimensions. Default : False.

    Inputs:
        - **input_x** (Tensor[Number]) - The input tensor.
        - **axis** (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed.

    Outputs:
        Tensor, has the same dtype as the 'input_x'.

        - If axis is (), and keep_dims is false,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is false,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is false,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = ReduceMean(keep_dims=True)
        >>> output = op(input_x, 1)
    """

    def __init__(self, keep_dims=True):
        super(ReduceMean, self).__init__()
        self.keep_dims = validator.check_type('keep_dims', keep_dims, [bool])
        self.sum = P.ReduceSum(self.keep_dims)

    def construct(self, x, axis):
        shape = P.Shape()(x)
        value_num = 1
        for i in axis:
            value_num *= shape[i]

        data_sum = self.sum(x, axis)
        avg = 1.0 / P.Fill()(P.DType()(x), (1,), value_num)
        res = data_sum * avg
        return res


class ReLU(GraphKernel):
    r"""
    Computes ReLU(Rectified Linear Unit) of input tensor element-wise.

    It returns :math:`\max(x,\  0)` element-wise.

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, with the same type and shape as the `input_x`.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> relu = ReLU()
        >>> result = relu(input_x)
        [[0, 4.0, 0.0], [2.0, 0.0, 9.0]]
    """
    def __init__(self):
        super(ReLU, self).__init__()
        self.max = P.Maximum()

    def construct(self, x):
        return self.max(P.Fill()(P.DType()(x), P.Shape()(x), 0.0), x)


class SoftmaxCrossEntropyWithLogits(GraphKernel):
    r"""
    Gets the softmax cross-entropy value between logits and labels which shoule be one-hot encoding.

    Note:
        Sets input logits as `X`, input label as `Y`, output as `loss`. Then,

        .. math::
            p_{ij} = softmax(X_{ij}) = \frac{exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)}

        .. math::
            loss_{ij} = -\sum_j{Y_{ij} * ln(p_{ij})}

    Inputs:
        - **logits** (Tensor) - Input logits, with shape :math:`(N, C)`.
        - **labels** (Tensor) - Ground truth labels, with shape :math:`(N, C)`.

    Outputs:
        Tuple of 2 Tensor, the loss shape is `(N,)`, and the dlogits with the same shape as `logits`.

    Examples:
        >>> logits = Tensor([[2, 4, 1, 4, 5], [2, 1, 2, 4, 3]], mindspore.float32)
        >>> labels = Tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0]], mindspore.float32)
        >>> softmax_cross = SoftmaxCrossEntropyWithLogits()
        >>> loss, backprop = softmax_cross(logits, labels)
    """

    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()
        self.max = P.ReduceMax(keep_dims=True)
        self.sum_keep_dims = P.ReduceSum(keep_dims=True)

    def construct(self, features, labels):
        data_max = self.max(features, (1,))
        data_sub = features - data_max
        data_exp = P.Exp()(data_sub)
        data_sum = self.sum_keep_dims(data_exp, (1,))
        data_div = data_exp / data_sum
        data_log_tmp = P.Log()(data_sum)
        data_log = data_sub - data_log_tmp
        data_mul = labels * data_log
        data_muls = P.Neg()(data_mul)
        loss = P.ReduceSum()(data_muls, (1,))
        backprop = data_div - labels
        return loss, backprop

    def bprop(self, features, labels, out, dout):
        grad = out[1]
        grad = grad * P.ExpandDims()(dout[0], -1)
        return grad, P.ZerosLike()(labels)


class LayerNormForward(GraphKernel):
    """ Forward function of the LayerNorm operator. """
    def __init__(self, begin_norm_axis=1, begin_params_axis=1):
        super(LayerNormForward, self).__init__()
        self.begin_norm_axis = validator.check_type('begin_norm_axis', begin_norm_axis, [int])
        self.begin_params_axis = validator.check_type('begin_params_axis', begin_params_axis, [int])
        self.mul = P.Mul()
        self.sum_keep_dims = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()
        self.add = P.TensorAdd()
        self.log = P.Log()
        self.exp = P.Exp()
        self.eps = P.Eps()

    def construct(self, input_x, input_gamma, input_beta):
        shape_x = P.Shape()(input_x)

        # Calculate the scaling ratio of the average
        begin_norm_axis = self.begin_norm_axis
        if begin_norm_axis < 0:
            begin_norm_axis += len(shape_x)
        reduce_axis = ()
        for i in range(len(shape_x)):
            if i > begin_norm_axis or i == begin_norm_axis:
                reduce_axis = reduce_axis + (i,)

        reduce_elts = 1.0
        for i in reduce_axis:
            reduce_elts *= shape_x[i]
        mean_cof = 1.0 / reduce_elts

        # Calculate mean
        mean_muls = self.mul(input_x, mean_cof)
        mean = self.sum_keep_dims(mean_muls, reduce_axis)

        # Calculate variance
        variance_sub = self.sub(input_x, mean)
        variance_mul = self.mul(variance_sub, variance_sub)
        variance_muls = self.mul(variance_mul, mean_cof)
        variance = self.sum_keep_dims(variance_muls, reduce_axis)

        # Calculate normalize
        normalize_sub = self.sub(input_x, mean)
        epsilon = self.eps(input_x)
        normalize_add = self.add(variance, epsilon)
        normalize_log = self.log(normalize_add)
        normalize_log_mul = self.mul(normalize_log, -0.5)
        normalize_exp = self.exp(normalize_log_mul)
        normalize_mul = self.mul(normalize_sub, normalize_exp)

        # Calculate scale and translate
        if self.begin_params_axis == 0:
            scale_mul = self.mul(input_gamma, normalize_mul)
            res = self.add(scale_mul, input_beta)
        else:
            scale_mul = self.mul(input_gamma, normalize_mul)
            res = self.add(scale_mul, input_beta)

        return res, mean, variance


class LayerNormXBackprop(GraphKernel):
    r"""
    Together with LayerNormBetaGammaBackprop, to supply the backprop
    functionality for LayerNorm.

    Note:
        Sets input_x as :math:`x_i`, variance as :math:`\sigma^2`, mean as :math:`\mu`,
        input_gamma as :math:`\gamma`. Then,
        .. math::
            \begin{array}{ll} \\
                \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
                \frac {\partial L} {\partial x_i} =
                    \frac{\gamma}{\sqrt{\sigma^2+\epsilon}}
                    ( \frac{\partial L}{\partial y_i}
                    - \frac{1}{m} \cdot \frac{\partial L}{\partial \beta}
                    - \frac{\hat{x_i}}{m} \cdot \frac{\partial L}{\partial \gamma})
            \end{array}

    Inputs:
        - **dy**(Tensor) - The first item of the next operator's backprop's output.
        - **input_x**(Tensor) - The first input of the forward function of LayerNorm.
        - **variance**(Tensor) - The second input of the forward function of LayerNorm.
        - **mean**(Tensor) - The third input of the forward function of LayerNorm.
        - **input_gamma**(Tensor) - The fourth input of the forward function of LayerNorm.

    Outputs:
        Tensor, the output of this operator, will be used as the first item of the result of
            LayerNorm's backprop function, has the same shape and data type as 'input_x'.

    Examples:
        >>> dy = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> variance = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> mean = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> input_gamma = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = LayerNormXBackprop(keep_dims=False)
        >>> output = op(dy, input_x, variance, mean, input_gamma)
    """

    def __init__(self):
        super(LayerNormXBackprop, self).__init__()
        self.sum_keep_dims = P.ReduceSum(keep_dims=True)
        self.log = P.Log()
        self.exp = P.Exp()
        self.eps = P.Eps()

    def construct(self, dy, input_x, variance, mean, input_gamma):
        shape_x = P.Shape()(input_x)
        shape_mean = P.Shape()(mean)
        reduce_axis = ()
        flag = -1
        min_l = 0
        if len(shape_x) > len(shape_mean):
            min_l = len(shape_x)
        else:
            min_l = len(shape_mean)
        for i in range(min_l):
            if (shape_x[i] != shape_mean[i]) and (flag == -1):
                flag = i
        if flag != -1:
            for i in range(flag, len(shape_x)):
                reduce_axis = reduce_axis + (i,)
        else:
            reduce_axis = reduce_axis + (len(shape_x) - 1,)
        mean_num = 1.0
        for i in reduce_axis:
            mean_num *= shape_x[i]
        pd_xl = input_gamma * dy
        epsilon = self.eps(input_x)
        var_elta = variance + epsilon
        var_elta_log = self.log(var_elta)
        var_elta_mul = var_elta_log * -0.5
        var_elta_2 = P.Exp()(var_elta_mul)
        pdvar1_mul = var_elta_2 * var_elta_2
        pd_var_1 = pdvar1_mul * var_elta_2
        sub_x_mean = input_x - mean
        pdvar_mul1 = pd_xl * sub_x_mean
        pdvar_sum = self.sum_keep_dims(pdvar_mul1, reduce_axis)
        pdvar_mul3 = pdvar_sum * pd_var_1
        pd_var = pdvar_mul3 * -0.5
        pdmean1_sum = self.sum_keep_dims(pd_xl, reduce_axis)
        pdmean1_mul = pdmean1_sum * var_elta_2
        pd_mean_1 = pdmean1_mul * -1.0
        pdmean2_mul1 = sub_x_mean * -2.0
        pdmean2_sum = self.sum_keep_dims(pdmean2_mul1, reduce_axis)
        pdmean2_mul3 = pdmean2_sum * (1.0 / mean_num)
        pd_mean_2 = pd_var * pdmean2_mul3
        pd_mean = pd_mean_2 + pd_mean_1
        pd_x_1 = var_elta_2 * pd_xl
        pdx2_mul = pd_var * sub_x_mean
        pd_x_2 = pdx2_mul * (2.0 * (1.0 / mean_num))
        pd_x_3 = pd_mean * (1.0 / mean_num)
        pdx_add = pd_x_1 + pd_x_2
        pd_x = pdx_add + pd_x_3
        return pd_x


class LayerNormBetaGammaBackprop(GraphKernel):
    r"""
    Together with LayerNormXBackprop, to supply the backprop functionality for
    LayerNorm.
    Note:
        Sets input_x as :math:`x_i`, variance as :math:`\sigma^2`, mean as :math:`\mu`,
        input_gamma as :math:`\gamma`. Then,
        .. math::
            \begin{array}{ll} \\
                \hat{x_i} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
                \frac {\partial L} {\partial \beta} =
                    \sum_{i=1}^m \\frac{\\partial L}{\partial y_i} \\
                \frac {\partial L} {\partial \gamma} =
                    \sum_{i=1}^m \\frac{\partial L}{\partial y_i} \cdot \hat{x_i}
            \end{array}

    Inputs:
        - **dy**(Tensor) - The first item of the next operator's backprop's output.
        - **input_x**(Tensor) - The first input of the forward function of LayerNorm.
        - **variance**(Tensor) - The second input of the forward function of LayerNorm.
        - **mean**(Tensor) - The third input of the forward function of LayerNorm.
        - **input_gamma**(Tensor) - The fourth input of the forward function of LayerNorm.

    Outputs:
        Tuple of 2 Tensor, the backprop outputs.

        - **pd_beta**(Tensor) - The first item of return value of this operator, will be used as
                    the second item of the LayerNorm's backprop function.
        - **pd_gamma**(Tensor) - The second item of return value of this operator, will be used as
                    the third item of the LayerNorm's backprop function.

    Examples:
        >>> dy = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> variance = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> mean = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> input_gamma = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = LayerNormBetaGammaBackprop(keep_dims=False)
        >>> pd_beta, pd_gamma = op(dy, input_x, variance, mean, input_gamma)
    """
    def __init__(self):
        super(LayerNormBetaGammaBackprop, self).__init__()
        self.sum_not_keep_dims = P.ReduceSum(keep_dims=False)
        self.log = P.Log()
        self.exp = P.Exp()
        self.eps = P.Eps()

    def construct(self, dy, input_x, variance, mean, shape_gamma):
        shape_x = P.Shape()(input_x)
        params_axis = ()

        if len(shape_x) != len(shape_gamma):
            sub = len(shape_x) - len(shape_gamma)
            for i in range(sub):
                params_axis = params_axis + (i,)

        pd_beta = self.sum_not_keep_dims(dy, params_axis)
        epsilon = self.eps(input_x)
        var_elta = variance + epsilon
        var_elta_log = self.log(var_elta)
        var_elta_mul = var_elta_log * -0.5
        var_elta_2 = P.Exp()(var_elta_mul)
        sub_x_mean = input_x - mean
        var_elta_2_cast = var_elta_2
        xl_mul = var_elta_2_cast * sub_x_mean
        pdga_mul = dy * xl_mul
        pd_gamma = self.sum_not_keep_dims(pdga_mul, params_axis)
        return pd_beta, pd_gamma


class LogSoftmax(GraphKernel):
    r"""
    Log Softmax activation function.

    Applies the Log Softmax function to the input tensor on the specified axis.
    Suppose a slice along the given aixs :math:`x` then for each element :math:`x_i`
    the Log Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \log \left(\frac{exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    where :math:`N` is the length of the Tensor.

    Args:
        axis (int): The axis to do the Log softmax operation. Default: -1.

    Inputs:
        logits (Tensor): The input of Log Softmax.

    Outputs:
        Tensor, with the same type and shape as the logits.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> log_softmax = LogSoftmax()
        >>> log_softmax(input_x)
        [-4.4519143, -3.4519143, -2.4519143, -1.4519144, -0.4519144]
    """

    def __init__(self, axis=-1):
        super(LogSoftmax, self).__init__()
        self.axis = validator.check_type('axis', axis, [int])
        self.max_keep_dims = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.sum_keep_dims = P.ReduceSum(keep_dims=True)
        self.log = P.Log()
        self.mul = P.Mul()

    def construct(self, input_x):
        data_max = self.max_keep_dims(input_x, (self.axis,))
        data_sub = self.sub(input_x, data_max)

        data_exp = self.exp(data_sub)
        data_sum = self.sum_keep_dims(data_exp, (self.axis,))
        data_log = self.log(data_sum)

        res = self.sub(data_sub, data_log)
        return res

    def bprop(self, input_x, out, dout):
        input_x = out
        input_dy = dout

        data_exp = self.exp(input_x)
        data_sum = self.sum_keep_dims(input_dy, (self.axis,))
        data_softmax = self.mul(data_exp, data_sum)

        res = self.sub(input_dy, data_softmax)
        return (res,)


class Tanh(GraphKernel):
    r"""
    Tanh activation function.

    Computes hyperbolic tangent of input element-wise. The Tanh function is defined as:

    .. math::
        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Inputs:
        - **input_x** (Tensor) - The input of Tanh.

    Outputs:
        Tensor, with the same type and shape as the input_x.

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> tanh = Tanh()
        >>> tanh(input_x)
        [0.7615941, 0.9640276, 0.9950548, 0.9993293, 0.99990916]
    """
    def __init__(self):
        super(Tanh, self).__init__()
        self.abs = P.Abs()
        self.add = P.TensorAdd()
        self.div = P.RealDiv()
        self.mul = P.Mul()
        self.mul_fp16 = P.Mul()
        self.mul_fp16.add_prim_attr("output_precision", "float16")
        self.exp = P.Exp()

    def construct(self, input_x):
        input_abs = self.abs(input_x)
        sign_flag = self.div(input_x, input_abs)
        sign_flag_neg = self.mul(sign_flag, -1.0)

        power_val = self.mul(input_abs, -2.0)
        exp_val = self.exp(power_val)
        up_val = self.add(exp_val, -1.0)
        down_val = self.add(exp_val, 1.0)

        div_val = self.div(up_val, down_val)
        res = self.mul(sign_flag_neg, div_val)
        return res

    def bprop(self, input_x, out, dout):
        input_y = out
        input_dy = dout

        data_square = self.mul(input_y, input_y)
        data_mul = self.mul(data_square, -1.0)
        anuminate = self.add(data_mul, 1.0)
        res = self.mul_fp16(anuminate, input_dy)

        return (res,)

class TanhGrad(GraphKernel):
    """
    Backprop function of Tanh

    Mathematical calculating:
        result = Tanh(out)
        result = 1 - result * result
        result = result * dout
    Inputs:
        out (Tensor): Tanh's output
        dout (Tensor): next layer's backward function's output, has same shape as out

    Outputs:
        result (Tensor): result of (1 - tanh(out)^2) * dout

    Examples:
        >>> x_np = np.random.randn(5, 3, 6).astype(np.float16)
        >>> dy_np = np.random.randn(5, 3, 6).astype(np.float16)
        >>> x_ms = Tensor(x_np)
        >>> dy_ms = Tensor(dy_np)
        >>> tanh_grad = TanhGrad()
        >>> out = tanh_grad(x_np, dy_np)
    """
    def __init__(self):
        super(TanhGrad, self).__init__()
        self.add = P.TensorAdd()
        self.mul = P.Mul()
        self.mul_fp16 = P.Mul()
        self.mul_fp16.add_prim_attr("output_precision", "float16")

    def construct(self, out, dout):
        input_y = out
        input_dy = dout

        data_square = self.mul(input_y, input_y)
        data_mul = self.mul(data_square, -1.0)
        anuminate = self.add(data_mul, 1.0)
        res = self.mul_fp16(anuminate, input_dy)

        return res

class Gelu(GraphKernel):
    r"""
    Gaussian Error Linear Units activation function.

    GeLU is described in the paper `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    And also please refer to `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
    <https://arxiv.org/abs/1810.04805>`_.

    Defined as follows:

    .. math::
        \text{output} = 0.5 * x * (1 + erf(x / \sqrt{2})),

    where :math:`erf` is the "Gauss error function" .

    Inputs:
        - **input_x** (Tensor) - Input to compute the Gelu.

    Outputs:
        Tensor, with the same type and shape as input.

    Examples:
        >>> tensor = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> gelu = Gelu()
        >>> result = gelu(tensor)
    """

    def __init__(self):
        super(Gelu, self).__init__()
        self.add = P.TensorAdd()
        self.abs = P.Abs()
        self.exp = P.Exp()
        self.neg = P.Neg()
        self.minimum = P.Minimum()
        self.div = P.RealDiv()
        self.mul = P.Mul()
        self.CSVALUE = 0.044715
        self.CSVALUE_A = 1.59576912
        self.CSVALUE_5 = 0.3989422804
        self.CSVALUE_3B = 0.2140644488

    def construct(self, input_x):
        def _tanh_parameter_compute(data_x):
            """
            compute the parameter of tanh:
            return: result equal (x+0.044715*tf.pow(x,3))
            """
            mul_0 = self.mul(data_x, data_x)
            pow_0 = self.mul(mul_0, data_x)
            mul_1 = self.mul(pow_0, self.CSVALUE)
            result = self.add(data_x, mul_1)

            return result

        tanh_parameter = _tanh_parameter_compute(input_x)
        mul_0 = self.mul(tanh_parameter, 1.5957691)

        mul_0_min = self.minimum(mul_0, 0.0)
        right_mul = self.exp(mul_0_min)

        mul_0_abs = self.abs(mul_0)
        mul_0_abs_neg = self.mul(mul_0_abs, -1.0)
        mul_0_abs_neg_exp = self.exp(mul_0_abs_neg)

        mul_0_abs_neg_exp_add = self.add(mul_0_abs_neg_exp, 1.0)
        left_mul = self.div(input_x, mul_0_abs_neg_exp_add)

        result = self.mul(left_mul, right_mul)
        return result

    def bprop(self, input_x, out, dout):
        """ register backprop function for Gelu """
        data_x = input_x
        data_gelu = out
        data_dy = dout

        def _math_four_compute(data_x):
            """
            return: math_four equal 2*(np(sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))
            """
            datax_pow = data_x * data_x * data_x
            datax_muls_c = self.mul(datax_pow, self.CSVALUE)
            datax_addx = self.add(datax_muls_c, data_x)
            datax_muls_s = self.mul(datax_addx, self.CSVALUE_A)

            return datax_muls_s

        # common part
        math_four = _math_four_compute(data_x)
        math_four_abs = self.abs(math_four)
        math_four_abs_neg = self.mul(math_four_abs, -1.0)
        math_four_abs_neg_exp = self.exp(math_four_abs_neg)
        math_four_min = self.minimum(math_four, 0.0)

        # dividend part
        datax_pow = self.mul(data_x, data_x)
        datax_pow_mul = self.mul(datax_pow, self.CSVALUE_3B)
        datax_pow_mul_add = self.add(datax_pow_mul, self.CSVALUE_A)
        data_gelu_mul = self.mul(data_gelu, datax_pow_mul_add)
        math_four_min_2 = self.mul(math_four_min, 2.0)
        div_right = self.mul(data_gelu_mul, math_four_abs_neg_exp)
        div_left = self.exp(math_four_min_2)
        dividend = self.add(div_left, div_right)

        # divisor part
        div_0 = self.add(math_four_abs_neg_exp, 1.0)
        div_1 = self.exp(math_four_min)
        divisor = self.mul(div_1, div_0)
        res_grad = self.div(dividend, divisor)

        result = self.mul(res_grad, data_dy)
        return (result,)


class Softmax(GraphKernel):
    """
    Operator Softmax
    .. math: `exp(x-max(x)) / sum(exp(x-max(x)))`

    Args:
        axis (int, tuple): Axis along which the softmax normalization is applied

    Inputs:
        x (Tensor): input data for softmax

    Outputs:
        output (Tensor): a tensor with the same shape of the input

    Examples:
        >>> layer = Softmax(1)
        >>> x = Tensor(np.array([1.2, 2.1], [2.2, 3.2]), mindspore.float32)
        >>> output = layer(x)
    """

    def __init__(self, axis):
        super(Softmax, self).__init__()
        validator.check_type("axis", axis, [int, tuple])
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = axis
        for item in self.axis:
            validator.check_type("item of axis", item, [int])
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.mul = P.Mul()

    def construct(self, x):
        max_x = self.max(x, self.axis)
        data_sub = self.sub(x, max_x)
        data_exp = self.exp(data_sub)
        data_expsum = self.sum(data_exp, self.axis)
        output = data_exp / data_expsum
        return output

    def bprop(self, x, out, dout):
        mul_res = self.mul(dout, out)
        sum_res = self.sum(mul_res, self.axis)
        sub_res = self.sub(dout, sum_res)
        res = self.mul(sub_res, out)
        return (res,)


class LayerNorm(Cell):
    r"""
    Applies Layer Normalization over a mini-batch of inputs.

    Layer normalization is widely used in recurrent neural networks. It applies
    normalization over a mini-batch of inputs for each single training case as described
    in the paper `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_. Unlike batch
    normalization, layer normalization performs exactly the same computation at training and
    testing times. It can be described using the following formula. It is applied across all channels
    and pixel but only one batch size.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Args:
        normalized_shape (Union(tuple[int], list[int]): The normalization is performed over axis
            `begin_norm_axis ... R - 1`.
        begin_norm_axis (int): It first normalization dimension: normalization will be performed along dimensions
            `begin_norm_axis: rank(inputs)`, the value should be in [-1, rank(input)). Default: -1.
        begin_params_axis (int): The first parameter(beta, gamma)dimension: scale and centering parameters
            will have dimensions `begin_params_axis: rank(inputs)` and will be broadcast with
            the normalized inputs accordingly, the value should be in [-1, rank(input)). Default: -1.
        gamma_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the gamma weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'ones'.
        beta_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the beta weight.
            The values of str refer to the function `initializer` including 'zeros', 'ones', 'xavier_uniform',
            'he_uniform', etc. Default: 'zeros'.

    Inputs:
        - **input_x** (Tensor) - The shape of 'input_x' is :math:`(x_1, x_2, ..., x_R)`,
          and `input_shape[begin_norm_axis:]` is equal to `normalized_shape`.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `input_x`.

    Examples:
        >>> x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
        >>> shape1 = x.shape()[1:]
        >>> m = G.LayerNorm(shape1,  begin_norm_axis=1, begin_params_axis=1)
        >>> m(x)
    """

    def __init__(self,
                 begin_norm_axis=-1,
                 begin_params_axis=-1
                 ):
        super(LayerNorm, self).__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_params_axis = begin_params_axis
        self.layer_norm = LayerNormForward(begin_norm_axis, begin_params_axis)
        self.layer_norm_x_grad = LayerNormXBackprop()
        self.layer_norm_beta_gamma = LayerNormBetaGammaBackprop()
        self.layer_norm_grad = G.LayerNormGrad(self.begin_norm_axis, self.begin_params_axis)

    def construct(self, input_x, input_gamma, input_beta):
        return self.layer_norm(input_x, input_gamma, input_beta)

    # case 1
    def bprop(self, input_x, input_gamma, input_beta, out, dout):
        dx, d_gamma, d_beta = self.layer_norm_grad(input_x, dout[0], out[2], dout[1], input_gamma)
        return dx, d_gamma, d_beta


class LambUpdateWithLR(GraphKernel):
    r"""
    Part of Lamb optimizer.

    .. math::
        s_1 = select(i_1 \gt y_g, select(i_0 \gt y_g, \frac{i_1}{i_2}, se), se)
        i_5 = i_5 - max(min(s_1, y_m), y_g) \times i_3 \times i_4

    Inputs:
        - **input0** (Tensor) - The first tensor to be computed.
        - **input1** (Tensor) - The second tensor to be computed.
        - **input2** (Tensor) - The third tensor to be computed.
        - **input3** (Tensor) - The fourth tensor to be computed.
        - **input4** (Tensor) - The fifth tensor to be computed.
        - **input5** (Tensor) - The sixth tensor to be computed. It will be updated by result.
        - **greater_y** (Tensor) - The seventh tensor to be computed.
        - **select_e** (Tensor) - The eighth tensor to be computed.
        - **minimum_y** (Tensor) - The ninth tensor to be computed.

    Outputs:
        A fake output tensor.

    Examples:
        >>> lamb_update = LambUpdateWithLR()
        >>> i0 = np.random.normal(0, 1, [1, 16]).astype(np.float32)
        >>> i1 = np.random.normal(0, 1, [1]).astype(np.float32)
        >>> i2 = np.random.normal(0, 1, [1]).astype(np.float32)
        >>> i3 = np.random.normal(0, 1, [1]).astype(np.float32)
        >>> i4 = np.random.normal(0, 1, [1, 16]).astype(np.float32)
        >>> i5 = np.random.normal(0, 1, [1, 16]).astype(np.float32)
        >>> yg = np.random.normal(0, 1, [1]).astype(np.float32)
        >>> se = np.random.normal(0, 1, [1]).astype(np.float32)
        >>> ym = np.random.normal(0, 1, [1]).astype(np.float32)
        >>> lamb_update(i0, i1, i2, i3, i4, i5, yg, se, ym)

    """

    def __init__(self):
        super(LambUpdateWithLR, self).__init__()
        self.greater = P.Greater()
        self.select = P.Select()
        self.div = P.RealDiv()
        self.min = P.Minimum()
        self.max = P.Maximum()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.fake_output_assign = InplaceAssign()
        self.fake_output_assign.add_prim_attr("fake_output", True)

    def construct(self, input0, input1, input2, input3, input4, input5, greater_y, select_e, minimum_y):
        greater0 = self.greater(input0, greater_y)
        greater1 = self.greater(input1, greater_y)
        real_div0 = self.div(input1, input2)
        select0 = self.select(greater0, real_div0, select_e)
        select1 = self.select(greater1, select0, select_e)
        min0 = self.min(select1, minimum_y)
        max0 = self.max(min0, greater_y)
        mul0 = self.mul(max0, input3)
        mul1 = self.mul(mul0, input4)
        sub0 = self.sub(input5, mul1)
        sub0 = self.fake_output_assign(input5, sub0, sub0)
        return sub0

class LambNextMV(GraphKernel):
    r"""
    Part of Lamb optimizer.

    .. math::
        rd_0 = \frac{i_8 \times i_5 + i_9 \times i_4}{i6}
        rd_1 = \frac{x_0 \times i_2 + x_1 \times i_1}{i3}
        y_2 = \frac{rd_0}{\sqrt{rd_1 + x3}} + x_2 \times i_7
        y_3 = \frac{rd_0}{\sqrt{rd_1} + x3}
        i5 = i_8 \times i_5 + i_9 \times i_4
        i2 = x_0 \times i_2 + x_1 \times i_1

    Inputs:
        - **inputs1** (Tensor) - The first input tensor to be computed.
        - **inputs2** (Tensor) - The second input tensor to be computed. It will be updated by result.
        - **inputs3** (Tensor) - The third input tensor to be computed.
        - **inputs4** (Tensor) - The fourth input tensor to be computed.
        - **inputs5** (Tensor) - The fifth input tensor to be computed. It will be updated by result.
        - **inputs6** (Tensor) - The sixth input tensor to be computed.
        - **inputs7** (Tensor) - The seventh input tensor to be computed.
        - **inputs8** (Tensor) - The eighth input tensor to be computed.
        - **inputs9** (Tensor) - The ninth input tensor to be computed.
        - **inputsx0** (Tensor) - The tenth input tensor to be computed.
        - **inputsx1** (Tensor) - The eleventh input tensor to be computed.
        - **inputsx2** (Tensor) - The twelfth input tensor to be computed.
        - **inputsx3** (Tensor) - The thirteenth input tensor to be computed.

    Outputs:
        Tuple of 2 Tensor.

        - **add3** (Tensor) - The shape is same as the shape after broadcasting, and the data type is
                              the one with high precision or high digits among the inputs.
        - **realdiv4** (Tensor) - The shape is same as the shape after broadcasting, and the data type is
                                  the one with high precision or high digits among the inputs.

    Examples:
        >>> lamb_next_mv = LambNextMV()
        >>> i1 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i2 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i3 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i4 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i5 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i6 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i7 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i8 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> i9 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> x0 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> x1 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> x2 = Tensor(np.random.normal(0, 1, [1, 16]).astype(np.float32))
        >>> x3 = Tensor(np.ones([1, 16]).astype(np.float32) * 1e-6)
        >>> lamb_next_mv(i1, i2, i3, i4, i5, i6, i7, i8, i9, x0, x1, x2, x3)

    """

    def __init__(self):
        super(LambNextMV, self).__init__()
        self.mul = P.Mul()
        self.add = P.TensorAdd()
        self.div = P.RealDiv()
        self.sqrt = P.Sqrt()
        self.rsqrt = P.Rsqrt()
        self.fake_output_assign_1 = InplaceAssign()
        self.fake_output_assign_1.add_prim_attr("fake_output", False)
        self.fake_output_assign_2 = InplaceAssign()
        self.fake_output_assign_2.add_prim_attr("fake_output", False)


    def construct(self, input1, input2, input3, input4, input5, input6, input7,
                  input8, input9, inputx0, inputx1, inputx2, inputx3):
        mul3 = self.mul(inputx1, input1)
        mul2 = self.mul(inputx0, input2)
        add1 = self.add(mul2, mul3)
        realdiv1 = self.div(add1, input3)
        add2 = self.add(realdiv1, inputx3)
        sqrt0 = self.rsqrt(add2)
        sqrt1 = self.sqrt(realdiv1)
        add4 = self.add(sqrt1, inputx3)
        mul1 = self.mul(input9, input4)
        mul0 = self.mul(input8, input5)
        add0 = self.add(mul0, mul1)
        realdiv0 = self.div(add0, input6)
        realdiv2 = self.mul(realdiv0, sqrt0)
        realdiv4 = self.div(realdiv0, add4)
        mul4 = self.mul(inputx2, input7)
        add3 = self.add(realdiv2, mul4)

        add3 = self.fake_output_assign_1(input5, add0, add3)
        add3 = self.fake_output_assign_2(input2, add1, add3)

        return add3, realdiv4
