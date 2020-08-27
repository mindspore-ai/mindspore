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
from copy import copy
from .._c_expression import ParamInfo
from . import dtype as mstype
from .initializer import initializer, Initializer
from .tensor import Tensor, MetaTensor
from .._checkparam import _check_str_by_regular
from ..parallel._tensor import _get_slice_index
from ..parallel._auto_parallel_context import auto_parallel_context

__all__ = ['Parameter', 'ParameterTuple']

PARAMETER_NAME_DEFAULT = "Parameter"
PARAMETER_NAME_PREFIX_MAX_LEN = 1024

def _is_in_parallel_mode():
    """Get parallel mode."""
    return auto_parallel_context().get_parallel_mode() in ["semi_auto_parallel", "auto_parallel"]


class Parameter(MetaTensor):
    """
    Parameter types of cell models.

    After initialized `Parameter` is a subtype of `Tensor`.

    In auto_parallel mode of  "semi_auto_parallel" and "auto_parallel", if init `Parameter` by
    an `Initializer`, the type of Parameter will be `MetaTensor` not `Tensor`. `MetaTensor`
    only saves the shape and type info of a tensor with no memory usage. The shape can be changed while
    compile for auto-parallel. Call `init_data` will return a Tensor Parameter with initialized data.

    Note:
        Each parameter of Cell is represented by Parameter class.

    Args:
        default_input (Union[Tensor, Initializer, Number]): Parameter data, to be set initialized.
        name (str): Name of the child parameter.
        requires_grad (bool): True if the parameter requires gradient. Default: True.
        layerwise_parallel (bool): A kind of model parallel mode. When layerwise_parallel is true in parallel mode,
            broadcast and gradients communication would not be applied to parameters. Default: False.

    Example:
        >>> from mindspore import Parameter, Tensor
        >>> from mindspore.common import initializer as init
        >>> from mindspore.ops import operations as P
        >>> from mindspore.nn import Cell
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import context
        >>>
        >>> class Net(Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.matmul = P.MatMul()
        >>>         self.weight = Parameter(Tensor(np.ones((1,2))), name="w", requires_grad=True)
        >>>
        >>>     def construct(self, x):
        >>>         out = self.matmul(self.weight, x)
        >>>         return out
        >>> context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        >>> net = Net()
        >>> x = Tensor(np.ones((2,1)))
        >>> net(x)
        [[2.]]
        >>> net.weight.set_parameter_data(Tensor(np.zeros((1,2))))
        >>> net(x)
        [[0.]]
    """
    __base_type__ = {}

    def __new__(cls, default_input, name, *args, **kwargs):
        input_class, *class_init_args = Parameter._get_parameter_new_args(default_input)
        new_type = Parameter._get_base_class(input_class)
        obj = input_class.__new__(new_type)
        input_class.__init__(obj, *class_init_args)
        # it's better to make the Initializer a kind of metatensor.
        obj.init_mode = None
        obj.is_default_input_initializer = False
        if isinstance(default_input, Initializer):
            obj.is_default_input_initializer = True
        if not isinstance(obj, Tensor):
            obj.init_mode = default_input
        return obj

    def __reduce_ex__(self, _):
        data = self
        if self.init_mode is not None:
            data = self.init_mode
        else:
            # cast to break deep infinit loop while deepcopy
            data = Tensor(self)
        return (
            Parameter, (data, self.name, self.requires_grad, self.layerwise_parallel))

    def __init__(self, default_input, name, requires_grad=True, layerwise_parallel=False):
        self._value = ParamInfo()
        self.name = name
        self.requires_grad = requires_grad
        self.layerwise_parallel = layerwise_parallel
        # this flag for tensor copy data.
        self.init_flag = False
        # this flag is for ge variable copy data.
        self._is_init = False
        self._inited_param = None
        self._sliced = False
        self.is_param_ps = False
        self._cast_type = None
        self.init_in_server = False
        self.is_in_parallel = _is_in_parallel_mode()

    @staticmethod
    def _get_base_class(input_class):
        input_class_name = f'Parameter{input_class.__name__}'
        if input_class_name in Parameter.__base_type__:
            new_type = Parameter.__base_type__[input_class_name]
        else:
            new_type = type(input_class_name, (Parameter, input_class), {})
            Parameter.__base_type__[input_class_name] = new_type
        return new_type

    @staticmethod
    def _get_parameter_new_args(data):
        """Set `default_input` of current `Parameter`."""
        if isinstance(data, bool):
            raise ValueError('Parameter data can not be `bool`')
        if isinstance(data, Initializer):
            if _is_in_parallel_mode():
                # do not init data while in auto parallel.
                return (MetaTensor, data.dtype, data.shape)
            data = data.to_tensor()
        if isinstance(data, Tensor):
            # make a copy of Tensor to init the parameter
            return (Tensor, data.asnumpy(),)
        if isinstance(data, int):
            return (Tensor, data, mstype.int32)
        if isinstance(data, float):
            return (Tensor, data, mstype.float32)
        return (Tensor, data)

    def __str__(self):
        value_str = MetaTensor.__str__(self)
        if isinstance(self, Tensor):
            value_str = Tensor.__str__(self)
        return f'Parameter (name={self._value.name}, value={value_str})'

    def __repr__(self):
        value_str = MetaTensor.__repr__(self)
        if isinstance(self, Tensor):
            value_str = Tensor.__repr__(self)
        return f'Parameter (name={self._value.name}, value={value_str})'

    def __parameter__(self):
        """For parse check."""

    def set_param_ps(self, init_in_server=False):
        self.is_param_ps = True
        self.init_in_server = init_in_server


    @property
    def inited_param(self):
        """Get the new parameter after call the init_data."""
        return self._inited_param


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
    def cast_type(self):
        return self._cast_type

    @cast_type.setter
    def cast_type(self, dst_type):
        if dst_type not in (mstype.float16, mstype.float32, None):
            raise ValueError("The type of the name should be type of [float32, float16] or `None`.")
        self._cast_type = dst_type

    @property
    def sliced(self):
        """Get slice status of the parameter."""
        return self._sliced

    @sliced.setter
    def sliced(self, sliced_):
        self._sliced = sliced_

    @property
    def is_init(self):
        """Get the initialization status of the parameter."""
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
            shape = self.shape
            dtype = self.dtype
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
        return self

    @default_input.setter
    def default_input(self, data):
        self.set_parameter_data(data)

    def _update_tensor_data(self, data):
        "Update the parameter by a Tensor."
        if isinstance(self, Tensor):
            # for Tensor same shape:
            self.init_flag = False
            return self.assign_value(data)
        # create a new tensor
        return Parameter(data, self.name, self.requires_grad)

    def set_parameter_data(self, data, slice_shape=False):
        """
        Set `default_input` of current `Parameter`.

        Args:
            data (Union[Tensor, Initializer, int, float]): new data.
            slice_shape (bool): If slice the Parameter, will not check if shape is match. Default: False.

        Retruns:
            Parameter, the parameter after set data.
        """
        def raise_type_error(incoming):
            raise TypeError(f"Incoming Parameter dtype can not be converted to current dtype implicitly. "
                            f"Current dtype is {self.dtype}, and incoming is {incoming}. "
                            f"Use .set_dtype(xxx) to change the dtype.")

        if not isinstance(data, (MetaTensor, Initializer, int, float)):
            raise TypeError(f"Parameter data must be [`Initializer`, `int`, `float`] or a kind of `MetaTensor` "
                            f"(like `Tensor` or `MetaTensor`). But with type {type(data)}.")
        if isinstance(data, (int, float)):
            if self.dtype in mstype.int_type and isinstance(data, float):
                raise_type_error(mstype.float_)
            data = Tensor(data, self.dtype)
        # both not init.
        is_incoming_tensor = isinstance(data, Tensor)
        is_current_tensor = isinstance(self, Tensor)

        if is_incoming_tensor and not is_current_tensor:
            raise TypeError("Parameter is a `MetaTensor` and not initializered, `data` for `set_parameter_data`"
                            "should be a Initializer. If you want to update it by Tensor, call method"
                            "`init_parameters_data` of `Cell` to init and replace all the Parameter of"
                            "network, then call this method.")
        if tuple(self.shape) != tuple(data.shape):
            # If Slice create Parameter shape can be change.
            if not slice_shape:
                raise ValueError(f"Can not change the shape of Parameter which has been initialized."
                                 f" Current shape is {self.shape}, and incoming is {data.shape}.")
        if self.dtype != data.dtype:
            if mstype.implicit_conversion_seq[self.dtype] < mstype.implicit_conversion_seq[data.dtype]:
                raise_type_error(data.dtype)
            else:
                data = Tensor(data, self.dtype)
        if isinstance(data, Initializer):
            # The parameter has been initializered, directly update by the data
            if is_current_tensor:
                self._update_tensor_data(data.to_tensor())
            else:
                # also update the related inited parameter data
                if self.inited_param is not None:
                    self.inited_param.set_parameter_data(data)
                self.init_mode = data
        elif is_incoming_tensor or is_current_tensor:
            self._update_tensor_data(data)
        else:
            raise ValueError(f"Not support to update the Parameter by {data}")
        self.sliced = slice_shape
        return self

    def init_data(self, layout=None, set_sliced=False):
        """
        Initialize the parameter data.

        Args:
            layout (list[list[int]]): Parameter slice layout [dev_mat, tensor_map, slice_shape].

                - dev_mat (list[int]): Device matrix.
                - tensor_map (list[int]): Tensor map.
                - slice_shape (list[int]): Shape of slice.
            set_sliced (bool): True if the parameter is set sliced after initializing the data.
                Default: False.

        Raises:
            RuntimeError: If it is from Initializer, and parallel mode has changed after the Initializer created.

        Returns:
            Parameter, the `Parameter` after initializing data. If current `Parameter` was already initialized before,
            returns the same initialized `Parameter`.
        """
        if self.is_default_input_initializer:
            is_current_in_parallel = _is_in_parallel_mode()
            if self.is_in_parallel != is_current_in_parallel:
                raise RuntimeError("Must set or change parallel mode before any Initializer created.")
        if self.init_mode is None:
            return self
        if self.inited_param is not None:
            return self.inited_param
        if layout is not None:
            if not isinstance(layout, list):
                raise TypeError("The layout should be list! layout is {}.".format(layout))
            if len(layout) < 3:
                raise ValueError("The length of layout must be larger than 3! layout is {}.".format(layout))
            slice_index = int(_get_slice_index(layout[0], layout[1]))
            if (self.init_in_server and self.is_param_ps and isinstance(self.init_mode, Initializer)):
                data = self.init_mode.to_tensor(0, [1])
            else:
                data = self.init_mode.to_tensor(slice_index, layout[2])
        else:
            if (self.init_in_server and self.is_param_ps and isinstance(self.init_mode, Initializer)):
                data = self.init_mode.to_tensor(0, [1])
            else:
                data = self.init_mode.to_tensor()

        obj = self._update_tensor_data(data)
        if id(obj) != id(self):
            self._inited_param = obj
        obj.init_mode = None
        obj.sliced = set_sliced
        return obj


class ParameterTuple(tuple):
    """
    Class for storing tuple of parameters.

    Note:
        It is used to store the parameters of the network into the parameter tuple collection.
    """
    def __new__(cls, iterable):
        """Create instance object of ParameterTuple."""
        data = tuple(iterable)
        for x in data:
            if not isinstance(x, Parameter):
                raise TypeError(f"ParameterTuple input should be `Parameter` collection."
                                f"But got a {type(iterable)}, {iterable}")
        return tuple.__new__(ParameterTuple, tuple(data))

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
