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
"""lars optimizer"""
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore._checkparam import Validator as validator
from mindspore.common import Tensor, Parameter, dtype as mstype
from .optimizer import _grad_scale, Optimizer

_lars_opt = C.MultitypeFuncGraph("lars_opt")


@_lars_opt.register("Function", "Tensor", "Number", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt(lars, learning_rate, weight_decay, gradient, weight, decay_flag, lars_flag):
    """Apply lars optimizer to the weight parameter."""
    if lars_flag:
        op_reduce_sum = P.SquareSumAll()
        w_square_sum, grad_square_sum = op_reduce_sum(weight, gradient)
        if decay_flag:
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, weight_decay, learning_rate)
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
    Implements the LARS algorithm with LARSUpdate Operator.

    LARS is an optimization algorithm employing a large batch optimization technique. Refer to paper `LARGE BATCH
    TRAINING OF CONVOLUTIONAL NETWORKS <https://arxiv.org/abs/1708.03888>`_.

    The updating formulas are as follows,

    .. math::

        \begin{array}{ll} \\
            \lambda  = \frac{\theta  \text{ * } || \omega  ||  }{|| g_{t} || \text{ + } \delta \text{ * } || \omega  || }  \\
            \lambda  =
            \begin{cases}
                \min(\frac{\lambda}{\alpha }, 1)
                    & \text{ if } clip = True \\
                \lambda
                    & \text{ otherwise }
            \end{cases}\\
            g_{t+1} = \lambda * (g_{t} + \delta * \omega)
        \end{array}

    :math:`\theta` represents `coefficient`, :math:`\omega` represents `parameters`, :math:`g` represents `gradients`,
    :math:`t` represents updateing step, :math:`\delta` represents `weight_decay`,
    :math:`\alpha` represents `learning_rate`, :math:`clip` represents `use_clip`.

    Args:
        optimizer (Optimizer): MindSpore optimizer for which to wrap and modify gradients.
        epsilon (float): Term added to the denominator to improve numerical stability. Default: 1e-05.
        coefficient (float): Trust coefficient for calculating the local learning rate. Default: 0.001.
        use_clip (bool): Whether to use clip operation for calculating the local learning rate. Default: False.
        lars_filter (Function): A function to determine whether apply the LARS algorithm. Default:
                                lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params` in the optimizer, the shape is the
          as same as the `params` in the optimizer.

    Outputs:
        Union[Tensor[bool], tuple[Parameter]], it depends on the output of `optimizer`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = nn.Momentum(net.trainable_params(), 0.1, 0.9)
        >>> opt_lars = nn.LARS(opt, epsilon=1e-08, coefficient=0.02)
        >>> model = Model(net, loss_fn=loss, optimizer=opt_lars, metrics=None)
    """

    def __init__(self, optimizer, epsilon=1e-05, coefficient=0.001, use_clip=False,
                 lars_filter=lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name):
        super(LARS, self).__init__(0.0, [Parameter(Tensor(0.0), name="fake_param")])
        _check_param_value(optimizer, epsilon, coefficient, use_clip, self.cls_name)
        self.opt = optimizer
        self.parameters = optimizer.parameters
        self.use_clip = use_clip
        self.lars_flag = tuple(lars_filter(x) for x in self.parameters)
        self.is_group = optimizer.is_group
        self.learning_rate = Parameter(Tensor(0.0, dtype=mstype.float32), name="fake_lr")
        self.decay_flags = optimizer.decay_flags
        self.reciprocal_scale = optimizer.reciprocal_scale
        self.need_scale = optimizer.need_scale
        self.hyper_map = C.HyperMap()
        self.lars = P.LARSUpdate(epsilon, coefficient, use_clip)
        self.cast = P.Cast()

        if use_clip:
            self.is_group_lr = optimizer.is_group_lr
            self.dynamic_lr = optimizer.dynamic_lr
            self.origin_learning_rate = optimizer.learning_rate
            self.global_step = optimizer.global_step
            if self.is_group_lr and self.dynamic_lr:
                raise ValueError('Grouped dynamic learning rate is currently not supported for the inputs optimizer ' \
                                 'of lars.')

        if self.is_group:
            self.weight_decay = tuple(map(lambda x: x / optimizer.loss_scale, optimizer.weight_decay))
            optimizer.weight_decay = tuple(map(lambda x: 0.0, optimizer.weight_decay))
        else:
            self.weight_decay = optimizer.weight_decay / optimizer.loss_scale
            optimizer.weight_decay = 0.0

        optimizer.decay_flags = tuple(map(lambda x: False, self.decay_flags))
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

    def construct(self, gradients):
        params = self.parameters
        if self.use_clip:
            lr = self._get_lr()
        else:
            lr = self.learning_rate

        if self.need_scale:
            gradients = self.hyper_map(F.partial(_grad_scale, self.reciprocal_scale), gradients)

        if self.is_group:
            if self.is_group_lr:
                gradients = self.hyper_map(F.partial(_lars_opt, self.lars), lr, self.weight_decay,
                                           gradients, params, self.decay_flags, self.lars_flag)
            else:
                gradients = self.hyper_map(F.partial(_lars_opt, self.lars, lr), self.weight_decay,
                                           gradients, params, self.decay_flags, self.lars_flag)
        else:
            gradients = self.hyper_map(F.partial(_lars_opt, self.lars, lr, self.weight_decay),
                                       gradients, params, self.decay_flags, self.lars_flag)
        success = self.opt(gradients)

        return success
