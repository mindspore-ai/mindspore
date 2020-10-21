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
"""Bijector"""
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P
from mindspore._checkparam import Validator as validator
from ..distribution._utils.utils import CheckTensor, cast_to_tensor
from ..distribution import Distribution
from ..distribution import TransformedDistribution


class Bijector(Cell):
    """
    Bijecotr class.

    Args:
        is_constant_jacobian (bool): Whether the Bijector has constant derivative. Default: False.
        is_injective (bool): Whether the Bijector is a one-to-one mapping. Default: True.
        name (str): The name of the Bijector. Default: None.
        dtype (mindspore.dtype): The type of the distributions that the Bijector can operate on. Default: None.
        param (dict): The parameters used to initialize the Bijector. Default: None.
    """

    def __init__(self,
                 is_constant_jacobian=False,
                 is_injective=True,
                 name=None,
                 dtype=None,
                 param=None):
        """
        Constructor of Bijector class.
        """
        super(Bijector, self).__init__()
        validator.check_value_type('name', name, [str], type(self).__name__)
        validator.check_value_type(
            'is_constant_jacobian', is_constant_jacobian, [bool], name)
        validator.check_value_type('is_injective', is_injective, [bool], name)
        self._name = name
        self._dtype = dtype
        self._parameters = {}
        # parsing parameters
        for k in param.keys():
            if k == 'param':
                continue
            if not(k == 'self' or k.startswith('_')):
                self._parameters[k] = param[k]
        self._is_constant_jacobian = is_constant_jacobian
        self._is_injective = is_injective

        self.context_mode = context.get_context('mode')
        self.checktensor = CheckTensor()

        # ops needed for the base class
        self.cast_base = P.Cast()
        self.dtype_base = P.DType()
        self.shape_base = P.Shape()
        self.fill_base = P.Fill()

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def parameters(self):
        return self._parameters

    @property
    def is_constant_jacobian(self):
        return self._is_constant_jacobian

    @property
    def is_injective(self):
        return self._is_injective

    def _add_parameter(self, value, name):
        """
        Cast `value` to a tensor and add it to `self.default_parameters`.
        Add `name` into  and `self.parameter_names`.
        """
        # initialize the attributes if they do not exist yet
        if not hasattr(self, 'default_parameters'):
            self.default_parameters = []
            self.parameter_names = []
        # cast value to a tensor if it is not None
        value_t = None if value is None else cast_to_tensor(value, self.parameter_type)
        self.default_parameters += [value_t,]
        self.parameter_names += [name,]
        return value_t

    def _calc_event_shape(self):
        """
        Calculate event_shape based on parameters.
        """
        broadcast_shape = None
        for param in self.default_parameters:
            if broadcast_shape is None:
                broadcast_shape = self.shape_base(param)
                broadcast_shape_tensor = self.fill_base(self.parameter_type, broadcast_shape, 0.0)
            else:
                broadcast_shape = self.shape_base(param + broadcast_shape_tensor)
                broadcast_shape_tensor = self.fill_base(self.parameter_type, broadcast_shape, 0.0)
        return broadcast_shape


    def _check_value(self, value, name):
        """
        Check availability of `value` as a Tensor.
        """
        self.checktensor(value, name)
        return value

    def cast_param_by_value(self, value, para):
        local = self.cast_base(para, self.dtype_base(value))
        return local

    def forward(self, *args, **kwargs):
        """
        Forward transformation: transform the input value to another distribution.
        """
        return self._forward(*args, **kwargs)

    def inverse(self, *args, **kwargs):
        """
        Inverse transformation: transform the input value back to the original distribution.
        """
        return self._inverse(*args, **kwargs)

    def forward_log_jacobian(self, *args, **kwargs):
        """
        Logarithm of the derivative of the forward transformation.
        """
        return self._forward_log_jacobian(*args, **kwargs)

    def inverse_log_jacobian(self, *args, **kwargs):
        """
        Logarithm of the derivative of the inverse transformation.
        """
        return self._inverse_log_jacobian(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Call Bijector directly.
        This __call__ may go into two directions:
        If args[0] is a distribution instance, the call will generate a new distribution derived from
        the input distribution.
        Otherwise, input[0] must be the name of a Bijector function, e.g. "forward", then this call will
        go in the construct and invoke the correstpoding Bijector function.

        Args:
            *args: args[0] shall be either a distribution or the name of a Bijector function.
        """
        if isinstance(args[0], Distribution):
            return TransformedDistribution(self, args[0], self.distribution.dtype)
        return super(Bijector, self).__call__(*args, **kwargs)

    def construct(self, name, *args, **kwargs):
        """
        Override `construct` in Cell.

        Note:
            Names of supported functions include:
            'forward', 'inverse', 'forward_log_jacobian', and 'inverse_log_jacobian'.

        Args:
            name (str): The name of the function.
            *args (list): A list of positional arguments that the function needs.
            **kwargs (dictionary): A dictionary of keyword arguments that the function needs.
        """
        if name == 'forward':
            return self.forward(*args, **kwargs)
        if name == 'inverse':
            return self.inverse(*args, **kwargs)
        if name == 'forward_log_jacobian':
            return self.forward_log_jacobian(*args, **kwargs)
        if name == 'inverse_log_jacobian':
            return self.inverse_log_jacobian(*args, **kwargs)
        return None
