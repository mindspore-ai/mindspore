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
"""lars optimizer"""
from __future__ import absolute_import

from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore._checkparam import Validator as validator
from mindspore.common import Tensor, Parameter, dtype as mstype
from mindspore.common.api import jit
from mindspore.nn.optim.optimizer import _grad_scale, Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register

_lars_opt = C.MultitypeFuncGraph("lars_opt")


@_lars_opt.register("Function", "Number", "Tensor", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt(lars, loss_scale, learning_rate, weight_decay, gradient, weight, decay_flag, lars_flag):
    """Apply lars optimizer to the weight parameter."""
    if lars_flag:
        op_reduce_sum = P.SquareSumAll()
        w_square_sum, grad_square_sum = op_reduce_sum(weight, gradient)
        if decay_flag:
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, weight_decay / loss_scale, learning_rate)
        else:
            num_zero = 0.0
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, num_zero, learning_rate)
        return grad_t

    return gradient


def _check_param_value(optimizer, epsilon, coefficient, use_clip, prim_name):
    validator.check_value_type("optimizer", optimizer, Optimizer, prim_name)
    validator.check_value_type("epsilon", epsilon, [float], prim_name)
    validator.check_value_type("coefficient", coefficient, [float], prim_name)
    validator.check_value_type("use_clip", use_clip, [bool], prim_name)


class LARS(Optimizer):
    r"""
    Implements the LARS algorithm.

    LARS is an optimization algorithm employing a large batch optimization technique. Refer to paper `LARGE BATCH
    TRAINING OF CONVOLUTIONAL NETWORKS <https://arxiv.org/abs/1708.03888>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            &\newline
            &\hline \\
            &\textbf{Parameters}: \text{base learning rate } \gamma_{0} , \text{ momentum  m}, \text{ weight decay }
             \lambda , \\
            &\hspace{5mm}\text{ LARS coefficient } \eta , \text{ number of steps } T \\
            &\textbf{Init}: \text{ t=0, v=0, init weight }  w_{0}^{l}  \text{ for each layer } l \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{while} \text{ t<T  for each layer } l \textbf{ do} \\
            &\hspace{5mm}g_{t}^{l} \leftarrow \nabla L\left(w_{t}^{l}\right) \\
            &\hspace{5mm}\gamma_{t} \leftarrow \gamma_{0} *\left(1-\frac{t}{T}\right)^{2} \\
            &\hspace{5mm}\gamma^{l} \leftarrow \eta *\frac{\left\|w_{t}^{l}\right\|}{\left\|g_{t}^{l}\right\|+
             \lambda\left\|w_{t}^{l}\right\|} \text{(compute the local LR } \gamma^{ l)} \\
            &\hspace{5mm}v_{t+1}^{l} \leftarrow m v_{t}^{l}+\gamma_{t+1} * \gamma^{l} *\left(g_{t}^{l}+\lambda
             w_{t}^{l}\right) \\
            &\hspace{5mm}w_{t+1}^{l} \leftarrow w_{t}^{l}-v_{t+1}^{l} \\
            &\textbf{ end while } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`w` represents the network parameters, :math:`g` represents `gradients`,
    :math:`t` represents the current step, :math:`\lambda` represents `weight_decay` in `optimizer`,
    :math:`\gamma` represents `learning_rate` in `optimizer`, :math:`\eta` represents `coefficient`.

    Args:
        optimizer (Optimizer): MindSpore optimizer for which to wrap and modify gradients.
        epsilon (float): Term added to the denominator to improve numerical stability. Default: 1e-05.
        coefficient (float): Trust coefficient for calculating the local learning rate. Default: 0.001.
        use_clip (bool): Whether to use clip operation for calculating the local learning rate. Default: False.
        lars_filter (Function): A function to determine which of the network parameters to use LARS algorithm. Default:
                                lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params` in the optimizer, the shape is the
          as same as the `params` in the optimizer.

    Outputs:
        Union[Tensor[bool], tuple[Parameter]], it depends on the output of `optimizer`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>>
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = nn.Momentum(net.trainable_params(), 0.1, 0.9)
        >>> opt_lars = nn.LARS(opt, epsilon=1e-08, coefficient=0.02)
        >>> model = ms.Model(net, loss_fn=loss, optimizer=opt_lars, metrics=None)
    """

    @opt_init_args_register
    def __init__(self, optimizer, epsilon=1e-05, coefficient=0.001, use_clip=False,
                 lars_filter=lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name):
        super(LARS, self).__init__(0.0, [Parameter(Tensor(0.0), name="fake_param")])
        _check_param_value(optimizer, epsilon, coefficient, use_clip, self.cls_name)
        self.opt = optimizer
        self.dynamic_decay_flags = optimizer.dynamic_decay_flags
        self.dynamic_weight_decay = optimizer.dynamic_weight_decay
        self.weight_decay = optimizer.weight_decay
        self.global_step = optimizer.global_step
        self.parameters = optimizer.parameters
        if optimizer._use_flattened_params:  # pylint: disable=W0212
            self.opt._use_flattened_params = False  # pylint: disable=W0212
        self._user_parameters += [param.name for param in self.parameters]
        self.use_clip = use_clip
        self.lars_flag = tuple(lars_filter(x) for x in self.parameters)
        self.is_group = optimizer.is_group
        self.learning_rate = Parameter(Tensor(0.0, dtype=mstype.float32), name="fake_lr")
        self.decay_flags = optimizer.decay_flags
        self.reciprocal_scale = optimizer.reciprocal_scale
        self.need_scale = optimizer.need_scale
        self.lars = P.LARSUpdate(epsilon, coefficient, use_clip)
        self.cast = P.Cast()
        self.loss_scale = optimizer.loss_scale

        if use_clip:
            self.is_group_lr = optimizer.is_group_lr
            self.dynamic_lr = optimizer.dynamic_lr
            self.origin_learning_rate = optimizer.learning_rate
            if self.is_group_lr and self.dynamic_lr:
                raise ValueError("For 'LARS', if the argument 'use_clip' is set to True, then the dynamic "
                                 "learning rate and group learning rate cannot both be true.")

        if self.is_group:
            optimizer.dynamic_decay_flags = tuple(map(lambda x: False, self.dynamic_decay_flags))
        else:
            optimizer.dynamic_decay_flags = False
        optimizer.decay_flags = tuple(map(lambda x: False, self.decay_flags))
        optimizer.dynamic_weight_decay = False
        optimizer.reciprocal_scale = 1.0
        optimizer.exec_weight_decay = False

    def _get_lr(self):
        """Get the learning rate of current step."""
        lr = self.origin_learning_rate
        if self.dynamic_lr:
            if self.is_group_lr:
                lr = ()
                for learning_rate in self.origin_learning_rate:
                    current_dynamic_lr = learning_rate(self.global_step)
                    lr += (current_dynamic_lr,)
            else:
                lr = self.origin_learning_rate(self.global_step)

        return lr

    @jit
    def construct(self, gradients):
        params = self.parameters
        gradients = self.flatten_gradients(gradients)
        if self.use_clip:
            lr = self._get_lr()
        else:
            lr = self.learning_rate
        weight_decay = self.get_weight_decay()

        if self.need_scale:
            gradients = self.hyper_map(F.partial(_grad_scale, self.reciprocal_scale), gradients)

        if self.is_group:
            if self.is_group_lr:
                gradients = self.hyper_map(F.partial(_lars_opt, self.lars, self.loss_scale), lr, weight_decay,
                                           gradients, params, self.decay_flags, self.lars_flag)
            else:
                gradients = self.hyper_map(F.partial(_lars_opt, self.lars, self.loss_scale, lr), weight_decay,
                                           gradients, params, self.decay_flags, self.lars_flag)
        else:
            gradients = self.hyper_map(F.partial(_lars_opt, self.lars, self.loss_scale, lr, weight_decay),
                                       gradients, params, self.decay_flags, self.lars_flag)
        success = self.opt(gradients)
        if self._is_dynamic_lr_or_weight_decay() and not self.opt.dynamic_lr and not self.opt.dynamic_weight_decay:
            self.assignadd(self.global_step, self.global_step_increase_tensor)
        return success
