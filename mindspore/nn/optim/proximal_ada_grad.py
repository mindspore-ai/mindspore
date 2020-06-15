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
"""PROXIMAL_ADA_GRAD"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from .optimizer import Optimizer

_proximal_ada_grad_opt = C.MultitypeFuncGraph("proximal_ada_grad_opt")


@_proximal_ada_grad_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt(opt, learning_rate, l1, l2, gradient, weight, accum):
    """Apply proximal_ada_grad optimizer to the weight parameter."""
    success = True
    success = F.depend(success, opt(weight, accum, learning_rate, l1, l2, gradient))
    return success


def _check_param_value(accum, l1, l2, use_locking, prim_name=None):
    """Check inputs param."""
    validator.check_value_type("accum", accum, [float], prim_name)
    validator.check_value_type("l1", l1, [float], prim_name)
    validator.check_value_type("l2", l2, [float], prim_name)
    validator.check_value_type("use_locking", use_locking, [bool], prim_name)
    validator.check_number_range("accum", accum, 0.0, float("inf"), Rel.INC_LEFT, prim_name)
    validator.check_number_range("l1", l1, 0.0, float("inf"), Rel.INC_LEFT, prim_name)
    validator.check_number_range("l2", l2, 0.0, float("inf"), Rel.INC_LEFT, prim_name)


class ProximalAdagrad(Optimizer):
    """
    Implement the ProximalAdagrad algorithm with ApplyProximalAdagrad Operator.

    ProximalAdagrad is an online Learning and Stochastic Optimization.
    Refer to paper `Efficient Learning using Forward-Backward Splitting
    <http://papers.nips.cc//paper/3793-efficient-learning-using-forward-backward-splitting.pdf>`_.

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
            should be Parameter.
        accum (float): The starting value for accumulators, must be zero or positive values. Default: 0.1.
        learning_rate (float): The learning rate value, must be greater than or equal to zero. Default: 0.001.
        l1 (float): l1 regularization strength, must be greater than or equal to zero. Default: 0.0.
        l2 (float): l2 regularization strength, must be greater than or equal to zero. Default: 0.0.
        use_locking (bool): If True use locks for update operation. Default: False.
        loss_scale (float): Value for the loss scale. It should be equal to or greater than 1.0. Default: 1.0.
        wegith_decay (float): Weight decay value to multiply weight, must be zero or positive value. Default: 0.0.

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of `params` in optimizer, the shape is as same as the `params`
          in optimizer.

    Outputs:
        Tensor[bool], the value is True.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> opt = nn.ProximalAdagrad(net.trainable_params())
        >>> model = Model(net, loss_fn=loss, optimizer=opt, metrics=None)
    """

    def __init__(self, params, accum=0.1, learning_rate=0.001, l1=0.0, l2=0.0,
                 use_locking=False, loss_scale=1.0, weight_decay=0.0):
        super(ProximalAdagrad, self).__init__(learning_rate, params, weight_decay, loss_scale)
        if self.is_group:
            raise RuntimeError(f"The {self.cls_name} optimizer cannot support group setting.")
        _check_param_value(accum, l1, l2, use_locking, self.cls_name)
        self.accum = self.parameters.clone(prefix="accum", init=accum)
        self.l1 = Tensor(l1, mstype.float32)
        self.l2 = Tensor(l2, mstype.float32)
        self.weight_decay = weight_decay
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyProximalAdagrad(use_locking=use_locking)

    def construct(self, grads):
        params = self.parameters
        accum = self.accum
        grads = self.decay_weight(grads)
        grads = self.scale_grad(grads)
        lr = self.learning_rate
        success = self.hyper_map(F.partial(_proximal_ada_grad_opt, self.opt, lr, self.l1, self.l2),
                                 grads, params, accum)
        return success
