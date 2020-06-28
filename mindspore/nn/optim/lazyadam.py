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
"""lazy adam"""
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from .optimizer import Optimizer

_lazy_adam_opt = C.MultitypeFuncGraph("lazy_adam_opt")


@_lazy_adam_opt.register("Function", "Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tuple",
                         "Tensor", "Tensor", "Tensor")
def _run_opt_with_sparse(opt, sparse_opt, beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, params,
                         moment1, moment2):
    """Apply sparse lazy adam optimizer to the weight parameter when the gradient is sparse."""
    success = True
    success = F.depend(success, sparse_opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                           eps, gradient[1], gradient[0]))
    return success


@_lazy_adam_opt.register("Function", "Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor",
                         "Tensor", "Tensor", "Tensor")
def _run_opt_with_one_number(opt, sparse_opt, beta1_power, beta2_power, beta1, beta2, eps, lr, gradient, params,
                             moment1, moment2):
    """Apply adam optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(params, moment1, moment2, beta1_power, beta2_power, lr, beta1, beta2,
                                    eps, gradient))
    return success


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


class LazyAdam(Optimizer):
    r"""
    Updates gradients by Adaptive Moment Estimation (Adam) algorithm.

    The Adam algorithm is proposed in `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_.

    The updating formulas are as follows,

    .. math::
        \begin{array}{ll} \\
            m = \beta_1 * m + (1 - \beta_1) * g \\
            v = \beta_2 * v + (1 - \beta_2) * g * g \\
            l = \alpha * \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \\
            w = w - l * \frac{m}{\sqrt{v} + \epsilon}
        \end{array}

    :math:`m` represents the 1st moment vector `moment1`, :math:`v` represents the 2nd moment vector `moment2`,
    :math:`g` represents `gradients`, :math:`l` represents scaling factor `lr`, :math:`\beta_1, \beta_2` represent
    `beta1` and `beta2`, :math:`t` represents updating step while :math:`beta_1^t` and :math:`beta_2^t` represent
    `beta1_power` and `beta2_power`, :math:`\alpha` represents `learning_rate`, :math:`w` represents `params`,
    :math:`\epsilon` represents `eps`.

    Note:
        The LazyAdam optimizer supports separating parameter groups. Different parameter groups can set different
        `learning_rate` and `weight_decay`.

        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        value of weight_decay > 0. When not separating parameter groups, the `weight_decay` in the API will be
        applied on the parameters if `weight_decay` > 0 and the 'beta' and 'gamma' are not in the name of parameters.

        The sparse strategy is applied while the SparseGatherV2 operator being used for forward network and the
        `sparse_grad` of `Parameter` being set. The sparse behavior, to be notice, is not equivalent to the
        original Adam algorithm, as only the current indices parames will be updated. The sparse feature is under
        continuous development. The sparse behavior is currently performed on the CPU.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` should be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr" and "weight_decay" are the keys can be parsed.

            - params: Required. The value should be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

        learning_rate (Union[float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                        Iterable or a Tensor and the dims of the Tensor is 1,
                                                        use dynamic learning rate, then the i-th step will
                                                        take the i-th value as the learning rate.
                                                        When the learning_rate is float or learning_rate is a Tensor
                                                        but the dims of the Tensor is 0, use fixed learning rate.
                                                        Other cases are not supported. Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimates. Should be in range (0.0, 1.0). Default:
                       0.9.
        beta2 (float): The exponential decay rate for the 2nd moment estimates. Should be in range (0.0, 1.0). Default:
                       0.999.
        eps (float): Term added to the denominator to improve numerical stability. Should be greater than 0. Default:
                     1e-8.
        use_locking (bool): Whether to enable a lock to protect updating variable tensors.
            If True, updating of the var, m, and v tensors will be protected by a lock.
            If False, the result is unpredictable. Default: False.
        use_nesterov (bool): Whether to use Nesterov Accelerated Gradient (NAG) algorithm to update the gradients.
            If True, updates the gradients using NAG.
            If False, updates the gradients without using NAG. Default: False.
        weight_decay (float): Weight decay (L2 penalty). Default: 0.0.
        loss_scale (float): A floating point value for the loss scale. Should be equal to or greater than 1. Default:
                            1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.LazyAdam(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': 0.01},
        >>>                 {'params': no_conv_params}]
        >>> opt = nn.LazyAdam(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # the conv_params's parameters will use a learning rate of 0.01 and a weight decay of 0.01
        >>> # the no_cov_params's parameters don't set learning and weight decay. So they will use a
        >>> # learning rate of 0.1 and a weight decay of 0.0.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False,
                 use_nesterov=False, weight_decay=0.0, loss_scale=1.0):
        super(LazyAdam, self).__init__(learning_rate, params, weight_decay, loss_scale)
        _check_param_value(beta1, beta2, eps, weight_decay, self.cls_name)
        validator.check_value_type("use_locking", use_locking, [bool], self.cls_name)
        validator.check_value_type("use_nesterov", use_nesterov, [bool], self.cls_name)

        self.beta1 = Tensor(beta1, mstype.float32)
        self.beta2 = Tensor(beta2, mstype.float32)
        self.beta1_power = Parameter(initializer(1, [1], mstype.float32), name="beta1_power")
        self.beta2_power = Parameter(initializer(1, [1], mstype.float32), name="beta2_power")
        self.eps = eps
        self.use_nesterov = use_nesterov
        self.use_locking = use_locking

        self.moment1 = self.parameters.clone(prefix="moment1", init='zeros')
        self.moment2 = self.parameters.clone(prefix="moment2", init='zeros')

        self.hyper_map = C.HyperMap()
        self.opt = P.Adam(use_locking, use_nesterov)
        self.sparse_opt = P.SparseApplyLazyAdam(use_locking, use_nesterov)

    def construct(self, gradients):
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()

        self.beta1_power = self.beta1_power * self.beta1
        self.beta2_power = self.beta2_power * self.beta2

        if self.is_group_lr:
            success = self.map_(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self.beta1_power,
                                          self.beta2_power, self.beta1, self.beta2, self.eps),
                                lr, gradients, self.parameters, self.moment1, self.moment2)
        else:
            success = self.map_(F.partial(_lazy_adam_opt, self.opt, self.sparse_opt, self.beta1_power,
                                          self.beta2_power, self.beta1, self.beta2, self.eps, lr),
                                gradients, self.parameters, self.moment1, self.moment2)
        return success
