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
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.nn.cell import Cell
from .optimizer import grad_scale

lars_opt = C.MultitypeFuncGraph("lars_opt")


@lars_opt.register("Function", "Number", "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt(lars, weight_decay, learning_rate, gradient, weight, decay_flag, lars_flag):
    """Apply lars optimizer to the weight parameter."""
    if lars_flag:
        op_reduce = P.ReduceSum()
        w_square_sum = op_reduce(F.square(weight))
        grad_square_sum = op_reduce(F.square(gradient))
        if decay_flag:
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, weight_decay, learning_rate)
        else:
            num_zero = 0.0
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, num_zero, learning_rate)
        return grad_t

    return gradient


@lars_opt.register("Function", "Number", "Number", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt_v2(lars, weight_decay, learning_rate, gradient, weight, decay_flag, lars_flag):
    """Apply lars optimizer to the weight parameter."""
    if lars_flag:
        op_reduce = P.ReduceSum()
        w_square_sum = op_reduce(F.square(weight))
        grad_square_sum = op_reduce(F.square(gradient))
        if decay_flag:
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, weight_decay, learning_rate)
        else:
            num_zero = 0.0
            grad_t = lars(weight, gradient, w_square_sum, grad_square_sum, num_zero, learning_rate)
        return grad_t

    return gradient


class LARS(Cell):
    """
    Implements the LARS algorithm with LARSUpdate Operator.

    LARS is an optimization algorithm employing a large batch optimization technique. Refer to paper `LARGE BATCH
    TRAINING OF CONVOLUTIONAL NETWORKS <https://arxiv.org/abs/1708.03888>`_.

    Args:
        optimizer (Optimizer): MindSpore optimizer for which to wrap and modify gradients.
        epsilon (float): Term added to the denominator to improve numerical stability. Default: 1e-05.
        hyperpara (float): Trust coefficient for calculating the local learning rate. Default: 0.001.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        use_clip (bool): Whether to use clip operation for calculating the local learning rate. Default: False.
        decay_filter (Function): A function to determine whether apply weight decay on parameters. Default:
                                 lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.
        lars_filter (Function): A function to determine whether apply lars algorithm. Default:
                                lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.
        loss_scale (float): A floating point value for the loss scale. Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params` in optimizer, the shape is
          as same as the `params` in optimizer.

    Outputs:
        Union[Tensor[bool], tuple[Parameter]], it depends on the output of `optimizer`.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = nn.Momentum(net.trainable_params(), 0.1, 0.9)
        >>> opt_lars = nn.LARS(opt, epsilon=1e-08, hyperpara=0.02)
        >>> model = Model(net, loss_fn=loss, optimizer=opt_lars, metrics=None)
    """

    def __init__(self, optimizer, epsilon=1e-05, hyperpara=0.001, weight_decay=0.0, use_clip=False,
                 decay_filter=lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name,
                 lars_filter=lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name, loss_scale=1.0):
        super(LARS, self).__init__(auto_prefix=False)
        self.opt = optimizer
        self.parameters = optimizer.parameters
        self.learning_rate = optimizer.learning_rate
        self.lars = P.LARSUpdate(epsilon, hyperpara, use_clip)
        self.reciprocal_scale = 1.0 / loss_scale
        self.weight_decay = weight_decay * loss_scale
        self.cast = P.Cast()
        self.decay_flag = tuple(decay_filter(x) for x in self.parameters)
        self.lars_flag = tuple(lars_filter(x) for x in self.parameters)
        self.hyper_map = C.HyperMap()
        self.dynamic_lr = False
        self.gather = None
        self.global_step = None
        self.axis = None
        if not isinstance(self.learning_rate, float):
            self.dynamic_lr = True
            self.assignadd = P.AssignAdd()
            self.gather = P.GatherV2()
            self.global_step = Parameter(initializer(0, [1], mstype.int32), name="lars_global_step")
            self.axis = 0

    def construct(self, gradients):
        params = self.parameters
        if self.dynamic_lr:
            lr = self.gather(self.learning_rate, self.global_step, self.axis)
            F.control_depend(lr, self.assignadd(self.global_step, 1))
        else:
            lr = F.scalar_to_array(self.learning_rate)
        if self.reciprocal_scale != 1.0:
            gradients = self.hyper_map(F.partial(grad_scale, self.reciprocal_scale), gradients)

        grad_t = self.hyper_map(F.partial(lars_opt, self.lars, self.weight_decay, lr),
                                gradients, params, self.decay_flag, self.lars_flag)
        success = self.opt(grad_t)

        return success
