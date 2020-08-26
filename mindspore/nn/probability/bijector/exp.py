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
"""Power Bijector"""
from .power_transform import PowerTransform

class Exp(PowerTransform):
    r"""
    Exponential Bijector.
    This Bijector performs the operation: Y = exp(x).

    Args:
        name (str): name of the bijector. Default: 'Exp'.

    Examples:
        >>> # To initialize a Exp bijector
        >>> import mindspore.nn.probability.bijector as msb
        >>> n = msb.Exp()
        >>>
        >>> # To use Exp distribution in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.e1 = msb.Exp()
        >>>
        >>>     def construct(self, value):
        >>>         # Similar calls can be made to other probability functions
        >>>         # by replacing 'forward' with the name of the function
        >>>         ans1 = self.s1.forward(value)
        >>>         ans2 = self.s1.inverse(value)
        >>>         ans3 = self.s1.forward_log_jacobian(value)
        >>>         ans4 = self.s1.inverse_log_jacobian(value)
    """
    def __init__(self,
                 name='Exp'):
        param = dict(locals())
        super(Exp, self).__init__(name=name, param=param)
