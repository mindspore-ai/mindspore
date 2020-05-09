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

"""Parameter for cell."""
import numbers
from copy import copy, deepcopy
from .initializer import initializer, Initializer
from .tensor import Tensor, MetaTensor
from .._checkparam import _check_str_by_regular
from ..parallel._utils import _set_clone_info, _CloneInfo

__all__ = ['Parameter', 'ParameterTuple']

PARAMETER_NAME_DEFAULT = "Parameter"
PARAMETER_NAME_PREFIX_MAX_LEN = 1024


def _check_type(x):
    """Check input data type"""
    if not isinstance(x, Parameter):
        raise ValueError("Should be `Parameter` collection.")
    return True


class Parameter:
    """
    Parameter types of cell models.

    Note:
        Each parameter of Cell is represented by Parameter class.

    Args:
        default_input (Union[Tensor, Initializer]): Parameter data, when `default_input` is` Initializer`,
            the data stored by Parameter is `MetaTensor`, otherwise it is `Tensor`.
        name (str): Name of the child parameter.
        requires_grad (bool): True if the parameter requires gradient. Default: True.
        layerwise_parallel (bool): A kind of model parallel mode. When layerwise_parallel is true in paralle mode,
            broadcast and gradients communication would not be applied on parameters. Default: False.
    """
    def __init__(self, default_input, name, requires_grad=True, layerwise_parallel=False):
        self.set_parameter_data(default_input)
        self.name = name
        self.requires_grad = requires_grad
        self.layerwise_parallel = layerwise_parallel
        self._is_init = False
        self.clone_info = _CloneInfo()

    def __repr__(self):
        format_str = 'Parameter (name={name})'
        return format_str.format(name=self._name)

    def __parameter__(self):
        """For parse check."""

    @property
    def name(self):
        """Get the name of the parameter."""
        return self._name

    @name.setter
    def name(self, name_):
        """
        Define a name for the parameter.

        Args:
            name_ (`str` or `None`): The name of the parameter. When the parameter is None or an empty string,
                the default value `PARAMETER_NAME_DEFAULT` is used.
        """
        if name_ is None:
            name_ = PARAMETER_NAME_DEFAULT
        elif isinstance(name_, str):
            name_ = name_.strip()
            if name_ == '':
                name_ = PARAMETER_NAME_DEFAULT
            if len(name_) > PARAMETER_NAME_PREFIX_MAX_LEN:
                raise ValueError("The length of the '{}' name should be less than {}.".
                                 format(name_, PARAMETER_NAME_PREFIX_MAX_LEN))
        else:
            raise ValueError("The type of the name should be `str` or `None`.")
        self._name = name_

    @property
    def is_init(self):
        """Get init status of the parameter."""
        return self._is_init

    @is_init.setter
    def is_init(self, is_init_):
        """
        Set init status of the parameter.

        Args:
            is_init_ (bool): The init status of the parameter.
        """
        self._is_init = is_init_

    def clone(self, prefix, init='same'):
        """
        Clone the parameter.

        Args:
            prefix (str): Namespace of parameter.
            init (Union[Tensor, str, Initializer, numbers.Number]): Initialize the shape of the parameter.
                Default: 'same'.

        Returns:
            Parameter, a new parameter.
        """
        _check_str_by_regular(prefix)
        x = copy(self)
        x.name = prefix + '.' + x.name
        x.is_init = False
        if init != 'same':
            shape = self.default_input.shape()
            dtype = self.default_input.dtype()
            if isinstance(init, (str, Initializer, numbers.Number)):
                x.init_mode = initializer(init, shape=shape, dtype=dtype)
                x.default_input = MetaTensor(dtype, shape)
            else:
                x.default_input = initializer(init, shape=shape, dtype=dtype)

        x.clone_info = copy(self.clone_info)
        _set_clone_info(self.clone_info, x.clone_info)
        return x

    @property
    def layerwise_parallel(self):
        return self._layerwise_parallel

    @layerwise_parallel.setter
    def layerwise_parallel(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("`layerwise_parallel` parameter must be bool type")
        self._layerwise_parallel = value

    @property
    def requires_grad(self):
        """Return whether the parameter requires gradient."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("`requires_grad` parameter must be bool type")
        self._requires_grad = value

    @property
    def data(self):
        return self.default_input

    def __add__(self, other):
        res = deepcopy(self)
        res.default_input = res.default_input + other
        return res

    def __sub__(self, other):
        res = deepcopy(self)
        res.default_input = res.default_input - other
        return res

    def __mul__(self, other):
        res = deepcopy(self)
        res.default_input = res.default_input * other
        return res

    def __truediv__(self, other):
        res = deepcopy(self)
        res.default_input = res.default_input / other
        return res

    def set_parameter_data(self, data):
        """Set `default_input` of current `Parameter`."""
        if isinstance(data, bool):
            raise ValueError('Parameter data can not be `bool`')
        if isinstance(data, Tensor):
            # make a copy of Tensor to init the parameter
            data = Tensor(data.asnumpy().copy())
        elif isinstance(data, Initializer):
            self.init_mode = data
            data = MetaTensor(self.init_mode.dtype, self.init_mode.shape)
        else:
            data = Tensor(data)
        self.default_input = data


    def init_data(self):
        if not isinstance(self.default_input, MetaTensor):
            return
        self.default_input = self.init_mode.to_tensor()
        self.init_mode = None


class ParameterTuple(tuple):
    """
    Class for storing tuple of parameters.

    Note:
        Used to store the parameters of the network into the parameter tuple collection.
    """
    def __new__(cls, iterable):
        """Create instance object of ParameterTuple."""
        g = (x for x in iterable if _check_type(x))
        return tuple.__new__(ParameterTuple, g)

    def clone(self, prefix, init='same'):
        """
        Clone the parameter.

        Args:
            prefix (str): Namespace of parameter.
            init (str): Initialize the shape of the parameter. Default: 'same'.

        Returns:
            Tuple, the new Parameter tuple.
        """
        _check_str_by_regular(prefix)
        new = []
        for x in self:
            x1 = x.clone(prefix, init)
            new.append(x1)
        return ParameterTuple(new)

    def __parameter_tuple__(self):
        """For parse check."""
