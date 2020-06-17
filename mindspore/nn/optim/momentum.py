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
"""momentum"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import check_bool
from mindspore._checkparam import Validator as validator
from .optimizer import Optimizer

momentum_opt = C.MultitypeFuncGraph("momentum_opt")


@momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success


class Momentum(Optimizer):
    """
    Implements the Momentum algorithm.

    Refer to the paper on the importance of initialization and momentum in deep learning for more details.

    Note:
        The Momentum optimizer supports separating parameter groups. Different parameter groups can set different
        `learning_rate` and `weight_decay`.

        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        value of weight_decay > 0. When not separating parameter groups, the `weight_decay` in the API will be
        applied on the parameters if `weight_decay` > 0 and the 'beta' and 'gamma' are not in the name of parameters.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` should be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value should be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" in the keys, the value should be the order of parameters and
              the order will be followed in optimizer. There are no other keys in the `dict` and the parameters which
              in the value of 'order_params' but not in any group will use default learning rate and default weight
              decay.

        learning_rate (Union[int, float, Tensor, Iterable]): A value for the learning rate. When the learning_rate is
                                                             Iterable or a Tensor and the dims of the Tensor is 1,
                                                             use dynamic learning rate, then the i-th step will
                                                             take the i-th value as the learning rate.
                                                             When the learning_rate is float or learning_rate is a
                                                             Tensor but the dims of the Tensor is 0, use fixed learning
                                                             rate. Other cases are not supported. It should be equal to
                                                             or greater than 0.0.
        momentum (float): Hyperparameter of type float, means momentum for the moving average.
            It should be at least 0.0.
        weight_decay (int, float): Weight decay (L2 penalty). It should be equal to or greater than 0.0. Default: 0.0.
        loss_scale (int, float): A floating point value for the loss scale. It should be greater than 0.0. Default: 1.0.
        use_nesterov (bool): Enable Nesterov momentum. Default: False.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Raises:
        ValueError: If the momentum is less than 0.0.

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> bias_params = list(filter(lambda x: 'bias' in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        >>>                 {'params': bias_params, 'lr': 0.01},
        >>>                 {'order_params': net.trainable_params()}]
        >>> opt = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
        >>> # The conv_params's parameters will use a learning rate of default value 0.1 and a weight decay of 0.01.
        >>> # The bias_params's parameters will use a learning rate of 0.01 and a weight decay of default value 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>> # The parameters which in the value of 'order_params' but not in any group will use a learning rate
        >>> # of default value 0.1 and a weight decay of default value 0.0.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(Momentum, self).__init__(learning_rate, params, weight_decay, loss_scale)
        validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.params = self.parameters
        self.use_nesterov = check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map(F.partial(momentum_opt, self.opt, self.momentum), lr, gradients, params, moments)
        else:
            success = self.hyper_map(F.partial(momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)
        return success
