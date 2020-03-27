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
"""lamb"""
import numpy as np
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore._checkparam import ParamValidator as validator
from mindspore._checkparam import Rel
from .optimizer import Optimizer
from .. import layer

num_one = Tensor(np.ones([1]), mstype.float32)

lamb_opt = C.MultitypeFuncGraph("lamb_opt")

@lamb_opt.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                   "Tensor", "Bool")
def _update_run_op(beta1, beta2, eps, lr, weight_decay_tensor, global_step, param, m, v,
                   gradient, decay_flag):
    """
    Update parameters.

    Args:
        beta1 (Tensor): The exponential decay rate for the 1st moment estimates. Should be in range (0.0, 1.0).
        beta2 (Tensor): The exponential decay rate for the 2nd moment estimates. Should be in range (0.0, 1.0).
        eps (Tensor): Term added to the denominator to improve numerical stability. Should be greater than 0.
        lr (Tensor): Learning rate.
        weight_decay_tensor (Tensor): Weight decay. Should be equal to or greater than 0.
        global_step (Tensor): Global step.
        param (Tensor): Parameters.
        m (Tensor): m value of parameters.
        v (Tensor): v value of parameters.
        gradient (Tensor): Gradient of parameters.
        decay_flag (bool): Specifies whether param update with weight decay.

    Returns:
        Tensor, the new value of v after updating.
    """
    op_mul = P.Mul()
    op_sqrt = P.Sqrt()
    op_rsqrt = P.Rsqrt()
    op_square = P.Square()
    op_cast = P.Cast()
    op_reshape = P.Reshape()
    op_shape = P.Shape()
    op_pow = P.Pow()
    op_norm = layer.Norm()
    op_select = P.Select()
    op_greater = P.Greater()
    op_fill = P.Fill()
    op_dtype = P.DType()

    param = op_cast(param, mstype.float32)
    m = op_cast(m, mstype.float32)
    v = op_cast(v, mstype.float32)
    gradient = op_cast(gradient, mstype.float32)

    next_m = op_mul(beta1, m) + op_mul(op_cast(num_one, mstype.float32) - beta1, gradient)

    next_v = op_mul(beta2, v) + op_mul(op_cast(num_one, mstype.float32) - beta2, op_square(gradient))

    next_mm = next_m / (op_cast(num_one, mstype.float32)
                        - op_pow(beta1, op_cast(global_step + num_one, mstype.float32)))
    next_vv = next_v / (op_cast(num_one, mstype.float32) -
                        op_pow(beta2, op_cast(global_step + num_one, mstype.float32)))
    w_norm = op_norm(param)
    g_norm = op_norm(gradient)

    g_norm_hat = op_norm(op_mul(next_mm, op_rsqrt(next_vv + eps)) + weight_decay_tensor * param)
    zeros = F.zeros_like_tensor(w_norm)
    ones = op_fill(op_dtype(w_norm), op_shape(w_norm), 1.0)
    trust_ratio = op_select(
        op_greater(w_norm, zeros),
        op_select(op_greater(g_norm, zeros), w_norm / g_norm_hat, ones),
        ones)
    tens = op_fill(op_dtype(trust_ratio), op_shape(trust_ratio), 10.0)
    trust_ratio = C.clip_by_value(trust_ratio, zeros, tens)
    update = next_mm / (op_sqrt(next_vv) + eps)

    if decay_flag:
        update = update + op_mul(weight_decay_tensor, param)

    update_with_lr = op_mul(op_mul(trust_ratio, lr), update)

    next_param = param - op_reshape(update_with_lr, op_shape(param))

    next_v = F.depend(next_v, F.assign(param, next_param))
    next_v = F.depend(next_v, F.assign(m, next_m))
    next_v = F.depend(next_v, F.assign(v, next_v))

    return next_v


def _check_param_value(decay_steps, warmup_steps, start_learning_rate,
                       end_learning_rate, power, beta1, beta2, eps, weight_decay):

    """Check the type of inputs."""
    validator.check_type("decay_steps", decay_steps, [int])
    validator.check_type("warmup_steps", warmup_steps, [int])
    validator.check_type("start_learning_rate", start_learning_rate, [float])
    validator.check_type("end_learning_rate", end_learning_rate, [float])
    validator.check_type("power", power, [float])
    validator.check_type("beta1", beta1, [float])
    validator.check_type("beta2", beta2, [float])
    validator.check_type("eps", eps, [float])
    validator.check_type("weight_dacay", weight_decay, [float])
    validator.check_number_range("decay_steps", decay_steps, 1, float("inf"), Rel.INC_LEFT)
    validator.check_number_range("beta1", beta1, 0.0, 1.0, Rel.INC_NEITHER)
    validator.check_number_range("beta2", beta2, 0.0, 1.0, Rel.INC_NEITHER)
    validator.check_number_range("eps", eps, 0.0, float("inf"), Rel.INC_NEITHER)
    validator.check_number_range("weight_decay", weight_decay, 0.0, float("inf"), Rel.INC_LEFT)


class Lamb(Optimizer):
    """
    Lamb Dynamic LR.

    LAMB is an optimization algorithm employing a layerwise adaptive large batch
    optimization technique. Refer to the paper `LARGE BATCH OPTIMIZATION FOR DEEP LEARNING: TRAINING BERT IN 76
    MINUTES <https://arxiv.org/abs/1904.00962>`_.

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
                                  should be class mindspore.Parameter.
        decay_steps (int): The steps of the lr decay. Should be equal to or greater than 1.
        warmup_steps (int): The steps of lr warm up. Default: 0.
        start_learning_rate (float): A floating point value for the learning rate. Default: 0.1.
        end_learning_rate (float): A floating point value for the end learning rate. Default: 0.0001.
        power (float): The power of the polynomial. Default: 1.0.
        beta1 (float): The exponential decay rate for the 1st moment estimates. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimates. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0. Should be equal to or greater than 0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default:
            lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[Parameter], the updated velocity value, the shape is the same as `params`.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = Lamb(params=net.trainable_params(), decay_steps=10)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """

    def __init__(self,
                 params,
                 decay_steps,
                 warmup_steps=0,
                 start_learning_rate=0.1,
                 end_learning_rate=0.0001,
                 power=1.0,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-6,
                 weight_decay=0.0,
                 decay_filter=lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name):

        super(Lamb, self).__init__(start_learning_rate, params)
        _check_param_value(decay_steps, warmup_steps, start_learning_rate, end_learning_rate,
                           power, beta1, beta2, eps, weight_decay)

        # turn them to scalar when me support scalar/tensor mix operations
        self.global_step = Parameter(initializer(0, [1]), name="global_step")

        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
        self.decay_steps = Tensor(np.array([decay_steps]).astype(np.float32))
        self.start_learning_rate = Tensor(np.array([start_learning_rate]).astype(np.float32))
        self.end_learning_rate = Tensor(np.array([end_learning_rate]).astype(np.float32))
        self.diff_learning_rate = Tensor(np.array([start_learning_rate - end_learning_rate]).astype(np.float32))
        self.power = power
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.weight_decay_tensor = Tensor(np.array([weight_decay]).astype(np.float32))
        self.params = self.parameters
        self.moments1 = self.params.clone(prefix="lamb_m", init='zeros')
        self.moments2 = self.params.clone(prefix="lamb_v", init='zeros')
        self.decay_flag = tuple(decay_filter(x) for x in self.params)

        self.hyper_map = C.HyperMap()
        self.min = P.Minimum()
        self.pow = P.Pow()
        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, gradients):
        step = self.min(self.global_step, self.decay_steps)
        p = step / self.decay_steps
        lr = self.diff_learning_rate * self.pow(self.one - p, self.power) + self.end_learning_rate
        if self.warmup_flag:
            warmup_percent = self.global_step / self.warmup_steps
            warmup_lr = self.start_learning_rate * warmup_percent
            is_warmup = self.cast(self.greater(self.warmup_steps, self.global_step), mstype.float32)
            lr = (self.one - is_warmup) * lr + is_warmup * warmup_lr
        updated_velocity = self.hyper_map(F.partial(lamb_opt, self.beta1, self.beta2, self.eps, lr,
                                                    self.weight_decay_tensor, self.global_step),
                                          self.params, self.moments1, self.moments2, gradients, self.decay_flag)

        added_global_step = self.global_step + self.one
        F.control_depend(lr, added_global_step)
        self.global_step = added_global_step

        return updated_velocity
