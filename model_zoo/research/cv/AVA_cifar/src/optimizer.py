# Copyright 2021 Huawei Technologies Co., Ltd
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
"""optimizer"""

from mindspore.nn import SGD
from mindspore.ops import functional as F, composite as C, operations as P

_sgd_opt = C.MultitypeFuncGraph("sgd_opt")


@_sgd_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, accum, stat):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, gradient, learning_rate, accum, momentum, stat))
    return success


class SGD_(SGD):
    """SGD optimizer"""
    def __init__(self, params, learning_rate=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False,
                 loss_scale=1.0):
        super(SGD_, self).__init__(params, learning_rate, momentum, dampening, weight_decay, nesterov, loss_scale)
        self.sm_scalar = P.ScalarSummary()

    def construct(self, gradients):
        """forward function"""
        params = self.parameters
        accum = self.accum
        stat = self.stat
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map(F.partial(_sgd_opt, self.opt, self.momentum), lr, gradients, params, accum, stat)
        else:
            success = self.hyper_map(F.partial(_sgd_opt, self.opt, self.momentum, lr), gradients, params, accum, stat)

        # Record learning rate here
        self.sm_scalar("learning_rate", lr)
        return success
