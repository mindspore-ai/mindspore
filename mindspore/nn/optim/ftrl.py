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
"""FTRL"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from .optimizer import Optimizer, _apply_decay, _grad_scale

_ftrl_opt = C.MultitypeFuncGraph("ftrl_opt")


@_ftrl_opt.register("Function", "Function", "Tensor", "Number", "Number", "Number", "Tensor", "Tuple", "Tensor",
                    "Tensor")
def _tensor_run_opt_with_sparse(opt, spars_opt, learning_rate, l1, l2, lr_power, linear, gradient, weight, moment):
    """Apply sparse ftrl optimizer to the weight parameter when the gradient is sparse."""
    success = True
    success = F.depend(success, spars_opt(weight, moment, linear, gradient[1], gradient[0]))
    return success


@_ftrl_opt.register("Function", "Function", "Tensor", "Number", "Number", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor")
def _tensor_run_opt(opt, spars_opt, learning_rate, l1, l2, lr_power, linear, gradient, weight, moment):
    """Apply ftrl optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, moment, linear, gradient, learning_rate, l1, l2, lr_power))
    return success


def _check_param(initial_accum, lr_power, l1, l2, use_locking, weight_decay=0.0, prim_name=None):
    """Check param."""
    validator.check_value_type("initial_accum", initial_accum, [float], prim_name)
    validator.check_number("initial_accum", initial_accum, 0.0, Rel.GE, prim_name)

    validator.check_value_type("lr_power", lr_power, [float], prim_name)
    validator.check_number("lr_power", lr_power, 0.0, Rel.LE, prim_name)

    validator.check_value_type("l1", l1, [float], prim_name)
    validator.check_number("l1", l1, 0.0, Rel.GE, prim_name)

    validator.check_value_type("l2", l2, [float], prim_name)
    validator.check_number("l2", l2, 0.0, Rel.GE, prim_name)

    validator.check_value_type("use_locking", use_locking, [bool], prim_name)

    validator.check_value_type("weight_decay", weight_decay, [float], prim_name)
    validator.check_number("weight_decay", weight_decay, 0.0, Rel.GE, prim_name)


class FTRL(Optimizer):
    """
    Implement the FTRL algorithm with ApplyFtrl Operator.

    FTRL is an online convex optimization algorithm that adaptively chooses its regularization function
    based on the loss functions. Refer to paper `Adaptive Bound Optimization for Online Convex Optimization
    <https://arxiv.org/abs/1002.4908>`_. Refer to paper `Ad Click Prediction: a View from the Trenches
    <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_ for engineering document.

    Note:
        The sparse strategy is applied while the SparseGatherV2 operator being used for forward network and the
        `sparse_grad` of `Parameter` being set as True. The sparse feature is under continuous development. The sparse
        behavior is currently performed on the CPU, weight decay is not supported.

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
            should be Parameter.
        initial_accum (float): The starting value for accumulators, must be zero or positive values. Default: 0.1.
        learning_rate (float): The learning rate value, should be positive. Default: 0.001.
        lr_power (float): Learning rate power controls how the learning rate decreases during training, must be less
            than or equal to zero. Use fixed learning rate if lr_power is zero. Default: -0.5.
        l1 (float): l1 regularization strength, must be greater than or equal to zero. Default: 0.0.
        l2 (float): l2 regularization strength, must be greater than or equal to zero. Default: 0.0.
        use_locking (bool): If True use locks for update operation. Default: False.
        loss_scale (float): Value for the loss scale. It should be equal to or greater than 1.0. Default: 1.0.
        wegith_decay (float): Weight decay value to multiply weight, must be zero or positive value. Default: 0.0.

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of `params` in optimizer, the shape is as same as the `params`
          in optimizer.

    Outputs:
        tuple[Parameter], the updated parameters, the shape is the same as `params`.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = nn.FTRL(net.trainable_params())
        >>> model = Model(net, loss_fn=loss, optimizer=opt, metrics=None)
    """
    def __init__(self, params, initial_accum=0.1, learning_rate=0.001, lr_power=-0.5, l1=0.0, l2=0.0,
                 use_locking=False, loss_scale=1.0, weight_decay=0.0):
        super(FTRL, self).__init__(learning_rate, params, loss_scale=loss_scale)
        if self.is_group:
            raise RuntimeError(f"The {self.cls_name} optimizer cannot support group setting.")
        _check_param(initial_accum, lr_power, l1, l2, use_locking, weight_decay, self.cls_name)
        self.moments = self.parameters.clone(prefix="moments", init=initial_accum)
        self.linear = self.parameters.clone(prefix="linear", init='zeros')
        self.l1 = l1
        self.l2 = l2
        self.lr_power = lr_power
        self.weight_decay = weight_decay
        self.decay_tf = tuple((lambda: True)() for x in self.parameters)
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyFtrl(use_locking=use_locking)
        self.sparse_opt = P.SparseApplyFtrl(learning_rate, l1, l2, lr_power, use_locking=use_locking)

    def construct(self, grads):
        params = self.parameters
        moments = self.moments
        linear = self.linear
        lr = self.learning_rate
        if self.weight_decay > 0.0:
            grads = self.hyper_map(F.partial(_apply_decay, self.weight_decay), self.decay_tf, params, grads)

        grads = self.scale_grad(grads)
        success = self.map_(F.partial(_ftrl_opt, self.opt, self.sparse_opt, lr, self.l1, self.l2, self.lr_power),
                            linear, grads, params, moments)
        return success
