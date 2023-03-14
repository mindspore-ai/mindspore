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
from mindspore.ops.operations import _inner_ops as inner
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from ..distribution._utils.utils import CheckTensor, cast_to_tensor, raise_type_error
from ..distribution import Distribution
from ..distribution import TransformedDistribution


class Bijector(Cell):
    """
    Bijecotr class. A bijector perform a mapping from one distribution to the other via some function.
    If :math:`X` is a random variable following the original distribution,
    and :math:`g(x)` is the mapping function,
    then :math:`Y = g(X)` is the random variable following the transformed distribution.

    Args:
        is_constant_jacobian (bool): Whether the Bijector has constant derivative. Default: False.
        is_injective (bool): Whether the Bijector is a one-to-one mapping. Default: True.
        name (str): The name of the Bijector. Default: None.
        dtype (mindspore.dtype): The type of the distributions that the Bijector can operate on. Default: None.
        param (dict): The parameters used to initialize the Bijector. Default: None.

    Note:
        `dtype` of bijector represents the type of the distributions that the bijector could operate on.
        When `dtype` is None, there is no enforcement on the type of input value except that the input value
        has to be float type. During initialization, when `dtype` is None, there is no enforcement on the dtype
        of the parameters. All parameters should have the same float type, otherwise a TypeError will be raised.
        Specifically, the parameter type will follow the dtype of the input value, i.e. parameters of the bijector
        will be casted into the same type as input value when `dtype` is None.
        When `dtype` is specified, it is forcing the parameters and input value to be the same dtype as `dtype`.
        When the type of parameters or the type of the input value is not the same as `dtype`, a TypeError will be
        raised. Only subtype of mindspore.float_type can be used to specify bijector's `dtype`.

    Supported Platforms:
        ``Ascend`` ``GPU``
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
        if dtype is not None:
            validator.check_type_name(
                "dtype", dtype, mstype.float_type, type(self).__name__)
        self._name = name
        self._dtype = dtype
        self._parameters = {}
        # parsing parameters
        for k in param.keys():
            if k == 'param':
                continue
            if not(k == 'self' or k.startswith('_')):
                self._parameters[k] = param[k]

        # if no bijector is used as an argument during initialization
        if 'bijector' not in param.keys():
            self._batch_shape = self._calc_batch_shape()
            self._is_scalar_batch = self._check_is_scalar_batch()

        self._is_constant_jacobian = is_constant_jacobian
        self._is_injective = is_injective

        self.context_mode = context.get_context('mode')
        self.checktensor = CheckTensor()

        # ops needed for the base class
        self.cast_base = P.Cast()
        self.dtype_base = P.DType()
        self.shape_base = P.Shape()
        self.fill_base = P.Fill()
        self.sametypeshape_base = inner.SameTypeShape()
        self.issubclass_base = inner.IsSubClass()

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

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def is_scalar_batch(self):
        return self._is_scalar_batch

    def _check_value_dtype(self, value):
        """
        Firstly check if the input value is Tensor. Then, if `self.dtype` is None, check
        if the input tensor is or can be directly cast into a float tensor.
        If `self.dtype` is not None, check if the input tensor's dtype is `self.dtype`.
        """
        self.checktensor(value, 'input value of bijector')
        value_type = self.dtype_base(value)
        if self.dtype is None:
            if self.issubclass_base(value_type, mstype.float_):
                return value
            return raise_type_error('input value of bijector', value_type, mstype.float_)
        dtype_tensor = self.fill_base(self.dtype, self.shape_base(value), 0.0)
        self.sametypeshape_base(value, dtype_tensor)
        return value

    def _shape_mapping(self, shape):
        shape_tensor = self.fill_base(self.parameter_type, shape, 0.0)
        dist_shape_tensor = self.fill_base(
            self.parameter_type, self.batch_shape, 0.0)
        return (shape_tensor + dist_shape_tensor).shape

    def shape_mapping(self, shape):
        return self._shape_mapping(shape)

    def _add_parameter(self, value, name):
        """
        Cast `value` to a tensor and add it to `self.default_parameters`.
        Add `name` into  and `self.parameter_names`.
        """
        # initialize the attributes if they do not exist yet
        if not hasattr(self, 'default_parameters'):
            self.default_parameters = []
            self.parameter_names = []
            self.common_dtype = None
        # cast value to a tensor if it is not None
        if isinstance(value, bool) or value is None:
            raise TypeError("{} cannot be type {}".format(name, type(value)))
        value_t = Tensor(value)
        # if the bijector's dtype is not specified
        if self.dtype is None:
            if self.common_dtype is None:
                self.common_dtype = value_t.dtype
            elif value_t.dtype != self.common_dtype:
                raise TypeError(
                    f"{name} should have the same dtype as other arguments.")
            # check if the parameters are casted into float-type tensors
            validator.check_type_name(
                f"dtype of {name}", value_t.dtype, mstype.float_type, type(self).__name__)
        # check if the dtype of the input_parameter agrees with the bijector's dtype
        elif value_t.dtype != self.dtype:
            raise TypeError(
                f"{name} should have the same dtype as the bijector's dtype.")
        self.default_parameters += [value,]
        self.parameter_names += [name,]
        return value_t

    def _calc_batch_shape(self):
        """
        Calculate batch_shape based on parameters.
        """
        if 'param_dict' not in self.parameters.keys():
            return None
        param_dict = self.parameters.get('param_dict')
        broadcast_shape_tensor = None
        for value in param_dict.values():
            if value is None:
                return None
            if broadcast_shape_tensor is None:
                broadcast_shape_tensor = cast_to_tensor(value)
            else:
                value = cast_to_tensor(value)
                broadcast_shape_tensor = (value + broadcast_shape_tensor)
        return broadcast_shape_tensor.shape

    def _check_is_scalar_batch(self):
        """
        Check if the parameters used during initialization are scalars.
        """
        if 'param_dict' not in self.parameters.keys():
            return False
        param_dict = self.parameters.get('param_dict')
        for value in param_dict.values():
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                return False
        return True

    def _check_value(self, value, name):
        """
        Check availability of `value` as a Tensor.
        """
        self.checktensor(value, name)
        return value

    def cast_param_by_value(self, value, para):
        """
        Cast the parameter(s) of the bijector to be the same type of input_value.

        Args:
            value (Tensor): input value.
            para (Tensor): parameter(s) of the bijector.

        Returns:
            Tensor, the value of parameters after casting.
        """
        local = self.cast_base(para, self.dtype_base(value))
        return local

    def forward(self, value, *args, **kwargs):
        """
        Forward transformation: transform the input value to another distribution.

        Args:
            value (Tensor): the value of the input variables.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Returns:
            Tensor, the value of the transformed random variable.
        """
        return self._forward(value, *args, **kwargs)

    def inverse(self, value, *args, **kwargs):
        """
        Inverse transformation: transform the input value back to the original distribution.

        Args:
            value (Tensor): the value of the transformed variables.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Returns:
            Tensor, the value of the input random variable.
        """
        return self._inverse(value, *args, **kwargs)

    def forward_log_jacobian(self, value, *args, **kwargs):
        """
        Logarithm of the derivative of the forward transformation.

        Args:
            value (Tensor): the value of the input variables.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Returns:
            Tensor, the value of logarithm of the derivative of the forward transformation.
        """
        return self._forward_log_jacobian(value, *args, **kwargs)

    def inverse_log_jacobian(self, value, *args, **kwargs):
        """
        Logarithm of the derivative of the inverse transformation.

        Args:
            value (Tensor): the value of the transformed variables.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Returns:
            Tensor, the value of logarithm of the derivative of the inverse transformation.
        """
        return self._inverse_log_jacobian(value, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Call Bijector directly.
        This __call__ may go into two directions:
        If args[0] is a distribution instance, the call will generate a new distribution derived from
        the input distribution.
        Otherwise, input[0] must be the name of a Bijector function, e.g. "forward", then this call will
        go in the construct and invoke the corresponding Bijector function.

        Args:
            *args: args[0] shall be either a distribution or the name of a Bijector function.
        """
        if isinstance(args[0], Distribution):
            return TransformedDistribution(self, args[0])
        return super(Bijector, self).__call__(*args, **kwargs)

    def construct(self, name, *args, **kwargs):
        """
        Override `construct` in Cell.

        Note:
            Names of supported functions include:
            'forward', 'inverse', 'forward_log_jacobian', and 'inverse_log_jacobian'.

        Args:
            name (str): The name of the function.
            *args (list): the list of positional arguments forwarded to subclasses.
            **kwargs (dict): the dictionary of keyword arguments forwarded to subclasses.

        Returns:
            Tensor, the result of the function corresponding to name.
        """
        if name == 'forward':
            return self.forward(*args, **kwargs)
        if name == 'inverse':
            return self.inverse(*args, **kwargs)
        if name == 'forward_log_jacobian':
            return self.forward_log_jacobian(*args, **kwargs)
        if name == 'inverse_log_jacobian':
            return self.inverse_log_jacobian(*args, **kwargs)
        raise Exception('Invalid name')
