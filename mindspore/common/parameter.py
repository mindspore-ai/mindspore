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
from copy import copy
from mindspore import context
from .._c_expression import ParamValue
from . import dtype as mstype
from .initializer import initializer, Initializer
from .tensor import Tensor, MetaTensor
from .._checkparam import _check_str_by_regular
from ..parallel._tensor import _get_slice_index

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
        self._value = ParamValue()
        self.set_parameter_data(default_input)
        self.name = name
        self.requires_grad = requires_grad
        self.layerwise_parallel = layerwise_parallel
        self._is_init = False
        self._sliced = False
        self.is_param_ps = False
        if context.get_context("mode") == context.PYNATIVE_MODE:
            self.init_data()

    def __repr__(self):
        format_str = 'Parameter (name={name})'
        return format_str.format(name=self._value.name)

    def __parameter__(self):
        """For parse check."""

    def set_param_ps(self):
        self.is_param_ps = True

    @property
    def name(self):
        """Get the name of the parameter."""
        return self._value.name

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
        self._value.name = name_

    @property
    def sliced(self):
        """Get slice status of the parameter."""
        return self._sliced

    @sliced.setter
    def sliced(self, sliced_):
        self._sliced = sliced_

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
        # pylint: disable=protected-access
        x._value = self._value.clone()
        x._value.name = prefix + '.' + self._value.name
        x.is_init = False
        if init != 'same':
            shape = self.default_input.shape
            dtype = self.default_input.dtype
            if isinstance(init, (str, Initializer, numbers.Number)):
                x.init_mode = initializer(init, shape=shape, dtype=dtype)
                x.default_input = MetaTensor(dtype, shape)
                if context.get_context("mode") == context.PYNATIVE_MODE:
                    x.init_data()
            else:
                x.default_input = initializer(init, shape=shape, dtype=dtype)
        return x

    @property
    def layerwise_parallel(self):
        return self._value.layerwise_parallel

    @layerwise_parallel.setter
    def layerwise_parallel(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("`layerwise_parallel` parameter must be bool type")
        self._value.layerwise_parallel = value

    @property
    def requires_grad(self):
        """Return whether the parameter requires gradient."""
        return self._value.requires_grad

    @requires_grad.setter
    def requires_grad(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("`requires_grad` parameter must be bool type")
        self._value.requires_grad = value

    @property
    def data(self):
        return self.default_input

    @property
    def default_input(self):
        return self._data

    @default_input.setter
    def default_input(self, data):
        self._data = data
        self._value.data = data

    def __add__(self, other):
        return self.default_input + other

    def __sub__(self, other):
        return self.default_input - other

    def __mul__(self, other):
        return self.default_input * other

    def __truediv__(self, other):
        return self.default_input / other

    def __setitem__(self, index, value):
        default_input = self.default_input
        default_input[index] = value
        return self

    def set_parameter_data(self, data):
        """Set `default_input` of current `Parameter`."""
        self.init_mode = None
        if isinstance(data, bool):
            raise ValueError('Parameter data can not be `bool`')
        if isinstance(data, Tensor):
            # make a copy of Tensor to init the parameter
            data = Tensor(data.asnumpy())
            data.init_flag = False
        elif isinstance(data, Initializer):
            self.init_mode = data
            data = MetaTensor(self.init_mode.dtype, self.init_mode.shape)
        elif isinstance(data, int):
            data = Tensor(data, dtype=mstype.int32)
        elif isinstance(data, float):
            data = Tensor(data, dtype=mstype.float32)
        else:
            data = Tensor(data)
            data.init_flag = False

        self.default_input = data

    def init_data(self, layout=None, set_sliced=False):
        """
        Init data of the parameter.

        Args:
            layout (list[list[int]]): Parameter slice layout [dev_mat, tensor_map, slice_shape].

                - dev_mat (list[int]): Device matrix.
                - tensor_map (list[int]): Tensor map.
                - slice_shape (list[int]): Shape of slice.
            set_sliced (bool): True if should set parameter sliced after init the data of initializer.
                Default: False.
        """
        if self.init_mode is None:
            return
        if layout is not None:
            if not isinstance(layout, list):
                raise TypeError("The layout should be list! layout is {}."
                                .format(layout))
            if len(layout) < 3:
                raise ValueError("The length of layout must be larger than 3! layout is {}."
                                 .format(layout))
            slice_index = int(_get_slice_index(layout[0], layout[1]))
            self.default_input = self.init_mode.to_tensor(slice_index, layout[2])
        else:
            self.default_input = self.init_mode.to_tensor()

        self.init_mode = None
        if set_sliced:
            self.sliced = True


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
