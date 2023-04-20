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
"""Exp Bijector"""
from .power_transform import PowerTransform


class Exp(PowerTransform):
    r"""
    Exponential Bijector.
    This Bijector performs the operation:

    .. math::
        Y = \exp(x).

    Args:
        name (str): The name of the Bijector. Default: ``'Exp'`` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>>
        >>> # To initialize an Exp bijector.
        >>> exp_bijector = nn.probability.bijector.Exp()
        >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
        >>> ans1 = exp_bijector.forward(value)
        >>> print(ans1.shape)
        (3,)
        >>> ans2 = exp_bijector.inverse(value)
        >>> print(ans2.shape)
        (3,)
        >>> ans3 = exp_bijector.forward_log_jacobian(value)
        >>> print(ans3.shape)
        (3,)
        >>> ans4 = exp_bijector.inverse_log_jacobian(value)
        >>> print(ans4.shape)
        (3,)
    """

    def __init__(self,
                 name='Exp'):
        super(Exp, self).__init__(name=name)

    def extend_repr(self):
        """Display instance object as string."""
        if self.is_scalar_batch:
            str_info = 'exp'
        else:
            str_info = 'batch_shape = {}'.format(self.batch_shape)
        return str_info
