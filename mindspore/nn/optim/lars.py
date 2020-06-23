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
    if "Adam" in optimizer.cls_name or "Lamb" in optimizer.cls_name:
        raise TypeError("LARS can not be used with ", optimizer.cls_name)
    validator.check_value_type("epsilon", epsilon, [float], prim_name)
    validator.check_value_type("coefficient", coefficient, [float], prim_name)
    validator.check_value_type("use_clip", use_clip, [bool], prim_name)

class LARS(Optimizer):
    """
    Implements the LARS algorithm with LARSUpdate Operator.

    LARS is an optimization algorithm employing a large batch optimization technique. Refer to paper `LARGE BATCH
    TRAINING OF CONVOLUTIONAL NETWORKS <https://arxiv.org/abs/1708.03888>`_.

    Args:
        optimizer (Optimizer): MindSpore optimizer for which to wrap and modify gradients.
        epsilon (float): Term added to the denominator to improve numerical stability. Default: 1e-05.
        coefficient (float): Trust coefficient for calculating the local learning rate. Default: 0.001.
        use_clip (bool): Whether to use clip operation for calculating the local learning rate. Default: False.
        lars_filter (Function): A function to determine whether apply lars algorithm. Default:
                                lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params` in optimizer, the shape is
          as same as the `params` in optimizer.

    Outputs:
        Union[Tensor[bool], tuple[Parameter]], it depends on the output of `optimizer`.

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
        self.lars = P.LARSUpdate(epsilon, coefficient, use_clip)
        self.cast = P.Cast()
        self.parameters = optimizer.parameters
        if use_clip is True:
            self.learning_rate = optimizer.learning_rate
            self.dynamic_lr = optimizer.dynamic_lr
            self.gather = optimizer.gather
            self.assignadd = optimizer.assignadd
            self.global_step = optimizer.global_step
        else:
            self.learning_rate = Parameter(Tensor(0.0, dtype=mstype.float32), name="fake_lr")
        self.reciprocal_scale = optimizer.reciprocal_scale
        optimizer.reciprocal_scale = 1.0
        self.is_group = optimizer.is_group
        if self.is_group:
            self.weight_decay = tuple(map(lambda x: x / optimizer.loss_scale, optimizer.weight_decay))
        else:
            self.weight_decay = optimizer.weight_decay / optimizer.loss_scale
        optimizer.exec_weight_decay = False
        optimizer.weight_decay = 0.0
        self.decay_flags = optimizer.decay_flags
        self.lars_flag = tuple(lars_filter(x) for x in self.parameters)
        self.hyper_map = C.HyperMap()

    def construct(self, gradients):
        params = self.parameters
        if self.dynamic_lr:
            lr = self.gather(self.learning_rate, self.global_step, 0)
            F.control_depend(lr, self.assignadd(self.global_step, 1))
        else:
            lr = self.learning_rate
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(_grad_scale, self.reciprocal_scale), gradients)
        if self.is_group:
            grad_t = self.hyper_map(F.partial(_lars_opt, self.lars, lr), self.weight_decay,
                                    gradients, params, self.decay_flags, self.lars_flag)
        else:
            grad_t = self.hyper_map(F.partial(_lars_opt, self.lars, lr, self.weight_decay),
                                    gradients, params, self.decay_flags, self.lars_flag)
        success = self.opt(grad_t)

        return success
