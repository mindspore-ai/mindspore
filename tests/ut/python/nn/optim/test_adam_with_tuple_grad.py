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
""" test adam """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Parameter, context
from mindspore.common.api import _executor
from mindspore.common import dtype as mstype
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Optimizer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel


adam_opt_for_map = C.MultitypeFuncGraph("adam_opt_for_map")
@adam_opt_for_map.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                           "Tensor", "Tensor", "Tensor", "Bool")
def _update_run_op_for_map(beta1, beta2, eps, lr, weight_decay_tensor, param, m, v, gradient, decay_flag):
    op_mul = P.Mul()
    op_square = P.Square()
    op_sqrt = P.Sqrt()
    op_cast = P.Cast()
    op_reshape = P.Reshape()
    op_shape = P.Shape()

    param_fp32 = op_cast(param, mstype.float32)
    m_fp32 = op_cast(m, mstype.float32)
    v_fp32 = op_cast(v, mstype.float32)
    gradient_fp32 = op_cast(gradient, mstype.float32)

    next_m = op_mul(beta1, m_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32) - beta1, gradient_fp32)

    next_v = op_mul(beta2, v_fp32) + op_mul(op_cast(F.tuple_to_array((1.0,)), mstype.float32)
                                            - beta2, op_square(gradient_fp32))

    update = next_m / (op_sqrt(next_v) + eps)
    if decay_flag:
        update = update + op_mul(weight_decay_tensor, param_fp32)

    update_with_lr = op_mul(lr, update)
    next_param = param_fp32 - op_reshape(update_with_lr, op_shape(param_fp32))

    next_v = F.depend(next_v, F.assign(param, next_param))
    next_v = F.depend(next_v, F.assign(m, next_m))
    next_v = F.depend(next_v, F.assign(v, next_v))
    return next_v


@adam_opt_for_map.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                           "Tensor", "Tensor", "Tuple", "Bool")
def _update_run_op_sparse_for_map(beta1, beta2, eps, lr, weight_decay_tensor, param, m, v, gradient, decay_flag):
    return gradient[2][2]

def _check_param_value(beta1, beta2, eps, weight_decay, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_value_type("weight_dacay", weight_decay, [float], prim_name)
    validator.check_number_range("beta1", beta1, 0.0, 1.0, Rel.INC_NEITHER, prim_name)
    validator.check_number_range("beta2", beta2, 0.0, 1.0, Rel.INC_NEITHER, prim_name)
    validator.check_number_range("eps", eps, 0.0, float("inf"), Rel.INC_NEITHER, prim_name)
    validator.check_number_range("weight_decay", weight_decay, 0.0, float("inf"), Rel.INC_LEFT, prim_name)


class AdamWeightDecaySparse(Optimizer):
    """
    Implements Adam algorithm weight decay fix.

    Args:
        params (list[Parameter]): A list of parameter, which will be updated. The element in `params`
                                  should be class mindspore.Parameter.
        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimates. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimates. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        decay_filter (Function): A function to determine whether to apply weight decay on parameters. Default:
                                 lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`,
          and might be in sparse format.

    Outputs:
        tuple[Parameter], the updated velocity value, the shape is the same as `params`.

    Examples:
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.AdamWeightDecay(params=net.trainable_params())
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
   """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0,
                 decay_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name):
        super(AdamWeightDecaySparse, self).__init__(learning_rate, params)
        if self.is_group:
            raise RuntimeError(f"The {self.cls_name} optimizer cannot support group setting.")
        _check_param_value(beta1, beta2, eps, weight_decay, self.cls_name)
        self.beta1 = Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = Tensor(np.array([beta2]).astype(np.float32))
        self.eps = Tensor(np.array([eps]).astype(np.float32))
        self.weight_decay_tensor = Tensor(np.array([weight_decay]).astype(np.float32))

        self.params = self.parameters
        self.moments1 = self.params.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.params.clone(prefix="adam_v", init='zeros')
        self.decay_flag = tuple(decay_filter(x) for x in self.params)

        self.map = C.Map()

    def construct(self, gradients):
        lr = self.get_lr()
        updated_velocity = self.map(F.partial(adam_opt_for_map, self.beta1, self.beta2, self.eps, lr,
                                              self.weight_decay_tensor),
                                    self.params, self.moments1, self.moments2, gradients, self.decay_flag)

        return updated_velocity


def test_AdamWeightDecaySparse():
    """ test_AdamWeightDecaySparse """
    context.set_context(mode=context.GRAPH_MODE)
    class Loss(nn.Cell):
        def __init__(self):
            super(Loss, self).__init__()
        def construct(self, base, target):
            return base
    class NetWithSparseGatherV2(nn.Cell):
        def __init__(self):
            super(NetWithSparseGatherV2, self).__init__()
            self.w1 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="w1", sparse_grad=True)
            self.w2 = Parameter(Tensor(np.ones([2, 1, 2]).astype(np.float32)), name="w2")
            self.gatherv2 = P.SparseGatherV2()
            self.axis = 0
        def construct(self, indices):
            return self.gatherv2(self.w1, indices, self.axis) * self.w2

    inputs = Tensor(np.array([0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()
    net.set_train()
    loss = Loss()
    optimizer = AdamWeightDecaySparse(net.trainable_params())

    net_with_loss = WithLossCell(net, loss)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
    _executor.compile(train_network, inputs, label)
