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
"""adafactor"""
from __future__ import absolute_import

from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.log import logging
from mindspore.common.initializer import initializer
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore import _checkparam as validator
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore.nn.optim.optimizer import Optimizer


def _rms(update_tensor):
    """calculate rms"""
    return F.sqrt(P.ReduceMean(False)(F.square(update_tensor)))


def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    """Approximation of exponential moving average of square of gradient"""
    reduce_mean = P.ReduceMean(keep_dims=True)(exp_avg_sq_row, -1)
    div_val = 1.0 / P.Sqrt()(P.Div()(exp_avg_sq_row, reduce_mean))
    r_factor = (P.ExpandDims()(div_val, -1))

    exp_avg_sq_col = P.ExpandDims()(exp_avg_sq_col, -2)
    c_factor = 1.0 / P.Sqrt()(exp_avg_sq_col)
    return P.Mul()(r_factor, c_factor)


reduce_mean_keep_alive = P.ReduceMean().add_prim_attr("keep_alive", True)
_adafactor_opt = C.MultitypeFuncGraph("adafactor_opt")


@_adafactor_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool", "Bool", "Bool", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _run_opt_with_one_number(eps, clip_threshold, beta1, beta2t, weight_decay, scale_parameter,
                             compression, use_first_moment, weight_decay_flag, learning_rate,
                             grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq):
    """Apply ada factor optimizer to the weight parameter using Tensor."""
    grad_dtype = F.dtype(grad)
    grad_shape = F.shape(grad)

    if grad_dtype == mstype.float16:
        grad = F.cast(grad, mstype.float32)
    p_data_fp32 = param
    if F.dtype(p_data_fp32) == mstype.float16:
        p_data_fp32 = F.cast(p_data_fp32, mstype.float32)

    factored = len(grad_shape) >= 2

    if scale_parameter:
        rms = _rms(p_data_fp32)
        param_scale = P.Maximum()(eps[1], rms)
        learning_rate_update = learning_rate * param_scale * F.ones_like(rms)
    else:
        learning_rate_update = learning_rate

    update = (grad ** 2) + eps[0]

    if factored:
        exp_avg_sq_row_update = F.cast(exp_avg_sq_row, grad_dtype)
        exp_avg_sq_row_update = P.Mul()(exp_avg_sq_row_update, beta2t)
        update_mean = reduce_mean_keep_alive(update, -1) * (1.0 - beta2t)
        exp_avg_sq_row_update = P.Add()(exp_avg_sq_row_update, update_mean)
        F.assign(exp_avg_sq_row, F.cast(exp_avg_sq_row_update, F.dtype(exp_avg_sq_row)))
        exp_avg_sq_row_update = exp_avg_sq_row

        exp_avg_sq_col_update = F.cast(exp_avg_sq_col, grad_dtype)
        exp_avg_sq_col_update = P.Mul()(exp_avg_sq_col_update, beta2t)
        update_mean = reduce_mean_keep_alive(update, -2) * (1.0 - beta2t)
        exp_avg_sq_col_update = P.Add()(exp_avg_sq_col_update, update_mean)
        F.assign(exp_avg_sq_col, F.cast(exp_avg_sq_col_update, F.dtype(exp_avg_sq_col)))
        exp_avg_sq_col_update = exp_avg_sq_col

        update = _approx_sq_grad(exp_avg_sq_row_update, exp_avg_sq_col_update)
        update = P.Mul()(update, grad)
    else:
        exp_avg_sq_update = F.cast(exp_avg_sq, grad_dtype)
        update = update * (1.0 - beta2t)
        exp_avg_sq_update = P.Add()(P.Mul()(exp_avg_sq_update, beta2t), update)
        F.assign(exp_avg_sq, F.cast(exp_avg_sq_update, F.dtype(exp_avg_sq)))
        exp_avg_sq_update = exp_avg_sq
        exp_avg_sq_update = 1.0 / P.Sqrt()(exp_avg_sq_update)
        update = P.Mul()(exp_avg_sq_update, grad)

    update_rms_thres = _rms(update) / clip_threshold
    update_coff = P.Maximum()(update_rms_thres, P.OnesLike()(update_rms_thres))
    update = P.Mul()(P.Div()(update, update_coff), learning_rate_update)

    if use_first_moment:
        exp_avg_update = exp_avg
        if compression:
            exp_avg_update = F.cast(exp_avg, grad_dtype)
        exp_avg_update = P.Add()(P.Mul()(exp_avg_update, beta1), update * (1 - beta1))
        F.assign(exp_avg, F.cast(exp_avg_update, F.dtype(exp_avg)))
        update = exp_avg

    if weight_decay_flag:
        p_data_fp32_coff = p_data_fp32 * -weight_decay * learning_rate_update
        p_data_fp32 = P.Add()(p_data_fp32, p_data_fp32_coff)
    p_data_fp32 = P.Sub()(p_data_fp32, update)
    P.Assign()(param, F.cast(p_data_fp32, F.dtype(param)))
    return True


@_adafactor_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor", "Tensor")
def _run_fused_ada_factor(fused_ada_factor, eps, clip_threshold, beta1, beta2t, weight_decay, learning_rate,
                          grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq):
    fused_ada_factor(eps, clip_threshold, beta1, beta2t, weight_decay, learning_rate,
                     grad, param, exp_avg, exp_avg_sq_row, exp_avg_sq_col, exp_avg_sq)
    return True


def trans_to_tensor(param, is_tuple=False, fp32=True):
    """
    Transform params to tensor.
    """
    if param is None or isinstance(param, bool):
        return param
    data_type = mstype.float32 if fp32 else mstype.float16
    if is_tuple:
        new_param = [Tensor(ele, data_type) for ele in param]
        return tuple(new_param)
    return Tensor(param, data_type)


class AdaFactor(Optimizer):
    r"""
    Updates gradients by the Adaptive Learning Rates with Sublinear Memory Cost (Adafactor) algorithm.

    The Adafactor algorithm is proposed in `Adafactor: Adafactor: Adaptive Learning Rates with Sublinear Memory
    Cost <https://arxiv.org/abs/1804.04235>`_.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Adafactor for weight vector are as follows,

    .. math::
        \begin{array}{l} \\
        \alpha_{t}=\max \left(\epsilon_{2}, \operatorname{RMS}\left(X_{t-1}\right)\right) \rho_{t} \\
        G_{t}=\nabla f_{t}\left(X_{t-1}\right) \\
        \hat{V}_{t}=\hat{\beta}_{2} \hat{V}_{t-1}+\left(1-\hat{\beta}_{2_{t}}\right)\left(G_{t}^{2}+ \\
        \epsilon_{1} 1_{n}\right) \\
        U_{t}=G_{t} / \sqrt{\hat{V}_{t}} \\
        \hat{U}_{t}=U_{t} / \max \left(1, \operatorname{RMS}\left(U_{t}\right) / d\right) \\
        X_{t}=X_{t-1}-\alpha_{t} \hat{U}_{t}
        \end{array}

    Adafactor for weight matrices are as follows,

    .. math::
        \begin{array}{l} \\
        \alpha_{t}=\max \left(\epsilon_{2}, \operatorname{RMS}\left(X_{t-1}\right)\right) \rho_{t} \\
        G_{t}=\nabla f_{t}\left(X_{t-1}\right) \\
        R_{t}=\hat{\beta}_{2 t} R_{t-1}+\left(1-\hat{\beta}_{2 t}\right)\left(G_{t}^{2}+ \\
        \epsilon_{1} 1_{n} 1_{m}^{\top}\right) 1_{m} \\
        C_{t}=\hat{\beta}_{2 t} C_{t-1}+\left(1-\hat{\beta}_{2 t}\right) 1_{n}^{\top}\left(G_{t}^{2}+ \\
        \epsilon_{1} 1_{n} 1_{m}^{\top}\right) \\
        \hat{V}_{t}=R_{t} C_{t} / 1_{n}^{\top} R_{t} \\
        U_{t}=G_{t} / \sqrt{\hat{V}_{t}} \\
        \hat{U}_{t}=U_{t} / \max \left(1, \operatorname{RMS}\left(U_{t}\right) / d\right) \\
        X_{t}=X_{t-1}-\alpha_{t} U_{t}
        \end{array}

    Where RMS is:

    .. math::
        \operatorname{RMS}\left(U_{t}\right)=\operatorname{RMS}_{x \in X}\left(u_{x t}\right)= \\
        \sqrt{\operatorname{Mean}_{x \in X}\left(\frac{\left(g_{x t}\right)^{2}}{\hat{v}_{x t}}\right)}

    :math:`x` is each individual parameter,
    :math:`t` is assumed to be the current number of steps,
    :math:`a_{t}` is the learning rate,
    :math:`f(X)` is the loss function,
    :math:`\epsilon1` and :math:`\epsilon2` is a small positive number to prevent errors,
    :math:`d` is the clipping threshold,
    :math:`\beta_{2}` is the moment decay,
    :math:`\rho` is the relative step size,
    :math:`R` is the running averages of the row sums of the squared gradient,
    :math:`C` is the running averages of the column sums of the squared gradient.

    Note:
        The learning rate depending of this optimizer will be control by the *scale_parameter*, *relative_step* and
        *warmup_init* options. To use a manual (external) learning rate schedule, it should be
        set `scale_parameter=False` and `relative_step=False`.

        If parameters is not used in the network, please do not add it to the optimizer,
        otherwise the calculation result will be abnormal.

        To improve parameter groups performance, the customized order of parameters is supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`.

        learning_rate (Union[float, Tensor]): A value or a graph for the learning rate.
            When the learning_rate is a Tensor in a 1D dimension.
            If the type of `learning_rate` is int, it will be converted to float. Default: ``None`` .
        eps (tuple): The regularization constans for square gradient and parameter scale respectively.
            default: ``(1e-30, 1e-3)`` .
        clip_threshold (Union[float, Tensor]): The threshold of root mean square of final gradient update.
            default: ``1.0``.
        decay_rate (Union[float, Tensor]): The coefficient used to compute running averages of square gradient.
            default: ``0.8`` .
        beta1 (float): The coefficient to computing running averages of gradient. Should be in range (0.0, 1.0).
            Default: ``None`` .
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: ``0.0`` .
        scale_parameter (bool): If True, learning rate is scaled by root mean square of parameter.
            default: ``True`` .
        relative_step (bool): If True, time-dependent learning rate is computed instead of external learning rate.
            default: ``True`` .
        warmup_init (bool): The time-dependent learning rate computation depends on whether warm-up
            initialization is being used. default: ``False`` .
        compression (bool): If True, the data type of the running averages exponent will be compression to float16.
            default: ``False`` .
        loss_scale (float): A floating point value for the loss scale. Should be greater than 0. In general, use the
            default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to ``False`` , then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.amp.FixedLossScaleManager` for more details.
            Default: ``1.0`` .

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        TypeError: If `learning_rate` is not one of int, float, Tensor, Iterable, LearningRateSchedule.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `beta1`, `beta2`, `eps` or `loss_scale` is not a float.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_locking` or `use_nesterov` is not a bool.
        ValueError: If `loss_scale` or `eps` is less than or equal to 0.
        ValueError: If `beta1`, `beta2` is not in range (0.0, 1.0).
        ValueError: If `weight_decay` is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> #1) Parameters use the default learning rate with None and weight decay with 0.
        >>> optim = nn.AdaFactor(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups
        >>> all_params = net.trainable_params()
        >>> group_params = [{'params': [all_params[0]]}, {'params': [all_params[1]]}]
        >>> optim = nn.AdaFactor(group_params, learning_rate=0.1, weight_decay=0.0, relative_step=False)
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = ms.Model(net, loss_fn=loss, optimizer=optim)
    """
    _support_parallel_optimizer = True

    @opt_init_args_register
    def __init__(self,
                 params,
                 learning_rate=None,
                 eps=(1e-30, 1e-3),
                 clip_threshold=1.0,
                 decay_rate=0.8,
                 beta1=0.9,
                 weight_decay=0.0,
                 scale_parameter=True,
                 relative_step=True,
                 warmup_init=False,
                 compression=False,
                 loss_scale=1.0):

        if learning_rate is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options", learning_rate)
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        if learning_rate is None and not relative_step:
            raise ValueError("Cannot learning_rate is None and relative_step=False")
        if learning_rate is None:
            learning_rate = 0.0
        if beta1 is None:
            beta1 = 0.0
        self.scale_lr = True
        if not isinstance(learning_rate, (float, int)) and learning_rate is not None:
            self.scale_lr = False
            if relative_step or scale_parameter:
                logging.warning("When learning_rate is learning scheduler, it not support update learning rate!")

        super(AdaFactor, self).__init__(learning_rate, params, weight_decay, loss_scale)
        validator.check_value_type("eps", eps, [list, tuple], self.cls_name)
        if len(eps) != 2:
            raise ValueError("eps must have 2 value: (eps1, eps2).")
        for i, ele in enumerate(eps):
            validator.check_value_type("eps{}".format(i), ele, [float], self.cls_name)
            validator.check_non_negative_float(ele, "eps{}".format(i), self.cls_name)
        validator.check_value_type("clip_threshold", clip_threshold, [float], self.cls_name)
        validator.check_non_negative_float(clip_threshold, "clip_threshold", self.cls_name)
        validator.check_value_type("decay_rate", decay_rate, [float], self.cls_name)
        validator.check_float_range(decay_rate, 0, 1, validator.INC_NEITHER, "decay_rate", self.cls_name)
        validator.check_float_range(weight_decay, 0, 1, validator.INC_LEFT, "weight_decay", self.cls_name)
        validator.check_value_type("scale_parameter", scale_parameter, [bool], self.cls_name)
        validator.check_value_type("relative_step", relative_step, [bool], self.cls_name)
        validator.check_value_type("compression", compression, [bool], self.cls_name)
        validator.check_value_type("beta1", beta1, [int, float], self.cls_name)
        validator.check_non_negative_float(float(beta1), "beta1", self.cls_name)
        self.eps = trans_to_tensor(eps)
        self.clip_threshold = trans_to_tensor(clip_threshold)
        self.decay_rate = trans_to_tensor(-decay_rate)
        self.beta1 = trans_to_tensor(beta1)
        self.weight_decay = trans_to_tensor(weight_decay)
        self.weight_decay_flag = bool(weight_decay)

        self.scale_parameter = scale_parameter
        self.relative_step = relative_step
        self.warmup_init = warmup_init
        self.compression = compression
        if not self.scale_lr:
            self.scale_parameter = False
        self.init_ada_factor_state(beta1)
        self.step = Parameter(initializer(0, [1], mstype.float32), name='afactor_step')
        self.fused_ada_factor = P.FusedAdaFactor(enable_scale_parameter=self.scale_parameter,
                                                 enable_first_moment=self.use_first_moment,
                                                 enable_weight_decay=self.weight_decay_flag)
        if context.get_context("device_target") == "CPU":
            self.use_fused_ada_factor = True
        else:
            self.use_fused_ada_factor = False
        logging.info("AdaFactor init completed %s.", self.learning_rate)

    def init_ada_factor_state(self, beta1):
        """init adafactor variables"""
        if beta1 > 0:
            self.use_first_moment = True
            self.exp_avg = self._parameters.clone(prefix="exp_avg", init='zeros')
        else:
            self.use_first_moment = False
            self.exp_avg = ParameterTuple([Parameter(Tensor(0.0))] * len(self._parameters))

        self.exp_avg_sq = []
        self.exp_avg_sq_col = []
        self.exp_avg_sq_row = []
        for param in self._parameters:
            param_dtype = param.dtype
            param_shape = param.shape
            param_name = param.name
            if len(param_shape) > 1:
                self.exp_avg_sq_row.append(Parameter(initializer(0, shape=param_shape[:-1], dtype=param_dtype),
                                                     name="exp_avg_sq_row_{}".format(param_name)))
                self.exp_avg_sq_col.append(Parameter(initializer(0, shape=param_shape[:-2] + param_shape[-1:],
                                                                 dtype=param_dtype),
                                                     name="exp_avg_sq_col_{}".format(param_name)))
                self.exp_avg_sq.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                 name="exp_avg_sq_{}".format(param_name)))

            else:
                self.exp_avg_sq_row.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                     name="exp_avg_sq_row_{}".format(param_name)))
                self.exp_avg_sq_col.append(Parameter(initializer(0, shape=(1,), dtype=param_dtype),
                                                     name="exp_avg_sq_col_{}".format(param_name)))

                if self.compression:
                    self.exp_avg_sq.append(Parameter(initializer(0, shape=param_shape, dtype=mstype.float16),
                                                     name="exp_avg_sq_{}".format(param_name)))
                else:
                    self.exp_avg_sq.append(Parameter(initializer(0, shape=param_shape, dtype=param_dtype),
                                                     name="exp_avg_sq_{}".format(param_name)))

        self.exp_avg_sq_row = ParameterTuple(self.exp_avg_sq_row)
        self.exp_avg_sq_col = ParameterTuple(self.exp_avg_sq_col)
        self.exp_avg_sq = ParameterTuple(self.exp_avg_sq)

    @property
    def supports_memory_efficient_fp16(self):
        """
        Support memory efficient for fp16
        """
        return True

    @property
    def supports_flat_params(self):
        """
        Support flatten params
        """
        return False

    @jit
    def construct(self, gradients):
        gradients = self.flatten_gradients(gradients)
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)
        step = F.assign_add(self.step, 1)
        if self.scale_lr and self.relative_step:
            if self.warmup_init:
                min_step = 1e-6 * step
            else:
                min_step = 1e-2
            lr = P.Minimum()(min_step, 1.0 / P.Sqrt()(step * 1.0))
        beta2t = 1.0 - P.Pow()(step, self.decay_rate)

        if self.use_fused_ada_factor:
            success = self.hyper_map(F.partial(_adafactor_opt, self.fused_ada_factor, self.eps, self.clip_threshold,
                                               self.beta1, beta2t, self.weight_decay, lr),
                                     gradients, self._parameters, self.exp_avg, self.exp_avg_sq_row,
                                     self.exp_avg_sq_col, self.exp_avg_sq)
        else:
            success = self.hyper_map(F.partial(_adafactor_opt, self.eps, self.clip_threshold, self.beta1, beta2t,
                                               self.weight_decay, self.scale_parameter, self.compression,
                                               self.use_first_moment, self.weight_decay_flag, lr),
                                     gradients, self._parameters, self.exp_avg, self.exp_avg_sq_row,
                                     self.exp_avg_sq_col, self.exp_avg_sq)

        return success

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        self._set_base_target(value)
        if value == 'CPU':
            self.fused_ada_factor.set_device("CPU")
            self.use_fused_ada_factor = True
        else:
            self.use_fused_ada_factor = False
