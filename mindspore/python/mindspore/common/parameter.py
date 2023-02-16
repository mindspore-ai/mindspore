# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

from copy import copy
import sys
import math
import numbers
import numpy as np
from mindspore import log as logger
from mindspore.log import _LogActionOnce
from mindspore._c_expression import ParamInfo
from mindspore.common import dtype as mstype
from mindspore import context
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.common._utils import get_slice_num, get_slice_shape
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator
from mindspore._check_jit_forbidden_api import jit_forbidden_register
from mindspore._c_expression import Tensor as Tensor_
from mindspore.parallel._tensor import _get_slice_index
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.parallel._ps_context import _is_role_worker, _is_role_pserver, _is_role_sched, _clone_hash_table, \
                                           _is_ps_mode
from mindspore.parallel._ps_context import _reinsert_hash_table_size, _insert_accumu_init_info, _cache_enable
import mindspore.common._monad as monad

__all__ = ['Parameter', 'ParameterTuple']

PARAMETER_NAME_DEFAULT = "Parameter"
PARAMETER_NAME_PREFIX_MAX_LEN = 1024

# Global variable for parameter unique key.
_GLOBAL_PARAMETER_KEY = -1


def _is_in_parallel_mode():
    """Get parallel mode."""
    return auto_parallel_context().get_parallel_mode() in ["semi_auto_parallel", "auto_parallel"]


def init_to_value(init):
    """
    Get value of initializer.

    Returns:
        Value of the initializer.

    Raises:
        ValueError: The value of the argument 'init' is not correct.
    """
    if isinstance(init, str):
        if init == 'zeros':
            return 0.0
        if init == 'ones':
            return 1.0
        raise ValueError("The argument 'init' should be one of values in ['zeros', 'ones'].")
    if isinstance(init, numbers.Number):
        return float(init)
    raise ValueError("The argument 'init' should be number or string, but got {}.".format(type(init)))


def _get_unique_parameter_key():
    """
    Get parameter unique key.
    Used to identify the same Parameter for Worker and Server in the embedding cache scenario.

    Returns:
        Integer. The unique parameter key.
    """
    global _GLOBAL_PARAMETER_KEY
    _GLOBAL_PARAMETER_KEY += 1
    return _GLOBAL_PARAMETER_KEY


class Parameter(Tensor_):
    """
    `Parameter` is a `Tensor` subclass, when they are assigned as Cell attributes they are automatically added to
    the list of its parameters, and will appear, e.g. in `cell.get_parameters()` iterator.

    Note:
        In auto_parallel mode of  "semi_auto_parallel" and "auto_parallel", if init `Parameter` by
        a `Tensor`, the type of Parameter will be `Tensor`. `Tensor`
        will save the shape and type info of a tensor with no memory usage. The shape can be changed while
        compiling for auto-parallel. Call `init_data` will return a Tensor Parameter with initialized data.
        If there is an operator in the network that requires part of the inputs to be Parameter,
        then the Parameters as this part of the inputs are not allowed to be cast.
        Give each `Parameter` a unique name to facilitate subsequent operations and updates.
        If there are two or more `Parameter` objects with the same name in a network,
        will be prompted to set a unique name when defining.

    Args:
        default_input (Union[Tensor, int, float, numpy.ndarray, list]): Parameter data,
            to initialize the parameter data.
        name (str): Name of the parameter. Default: None.

            1) If the parameter is not given a name, the default name is its variable name. For example, the name of
            param_a below is name_a, and the name of param_b is the variable name param_b.

            .. code-block::

                self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
                self.param_b = Parameter(Tensor([2], ms.float32))

            2) If parameter in list or tuple is not given a name, will give it a unique name. For example, the names of
            parameters below are **Parameter$1** and **Parameter$2**.

            .. code-block::

                self.param_list = [Parameter(Tensor([3], ms.float32)),
                                   Parameter(Tensor([4], ms.float32))]

            3) If the parameter is given a name, and the same name exists between different parameters, an exception
            will be thrown. For example, "its name 'name_a' already exists." will be thrown.

            .. code-block::

                self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
                self.param_tuple = (Parameter(Tensor([5], ms.float32), name="name_a"),
                                    Parameter(Tensor([6], ms.float32)))

            4) If a parameter appear multiple times in list or tuple, check the name of the object only once. For
            example, the following example will not throw an exception.

            .. code-block::

                self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")
                self.param_tuple = (self.param_a, self.param_a)

        requires_grad (bool): True if the parameter requires gradient. Default: True.
        layerwise_parallel (bool): When layerwise_parallel is true in data/hybrid parallel mode,
            broadcast and gradients communication would not be applied to parameters. Default: False.
        parallel_optimizer (bool): It is used to filter the weight shard operation in semi auto or auto parallel
            mode. It works only when enable parallel optimizer in `mindspore.set_auto_parallel_context()`.
            Default: True.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Parameter, Tensor
        >>> import mindspore.ops as ops
        >>> import mindspore.nn as nn
        >>> import mindspore
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.matmul = ops.MatMul()
        ...         self.weight = Parameter(Tensor(np.ones((1, 2)), mindspore.float32), name="w", requires_grad=True)
        ...
        ...     def construct(self, x):
        ...         out = self.matmul(self.weight, x)
        ...         return out
        >>> net = Net()
        >>> x = Tensor(np.ones((2, 1)), mindspore.float32)
        >>> print(net(x))
        [[2.]]
        >>> net.weight.set_data(Tensor(np.zeros((1, 2)), mindspore.float32))
        >>> print(net(x))
        [[0.]]
    """
    _base_type = {}

    def __new__(cls, default_input, *args, **kwargs):
        init_data_flag = bool(isinstance(default_input, Tensor) and default_input.has_init)
        rc = sys.getrefcount(default_input)
        input_class, *class_init_args = Parameter._get_parameter_new_args(default_input, rc)
        new_type = Parameter._get_base_class(input_class)
        obj = input_class.__new__(new_type)
        input_class.__init__(obj, *class_init_args)
        # it's better to make the Initializer a kind of tensor.
        obj.init_mode = None
        obj.is_default_input_init = init_data_flag
        if obj.has_init:
            obj.init_mode = default_input
        return obj

    def __reduce_ex__(self, _):
        data = self
        if self.init_mode is not None:
            data = self.init_mode
        else:
            # cast to break deep infinite loop while deepcopy
            data = Tensor(self)
        return (
            Parameter, (data, self.name, self.requires_grad, self.layerwise_parallel))

    def __init__(self, default_input, name=None, requires_grad=True, layerwise_parallel=False, parallel_optimizer=True):
        self.param_info = ParamInfo()
        self.init_in_server = False
        self.name = name
        self.requires_grad = requires_grad
        self.layerwise_parallel = layerwise_parallel
        self.parallel_optimizer = parallel_optimizer
        # this flag for tensor copy data.
        self.init_flag = False
        # this flag is for ge variable copy data.
        self.is_init = False
        self._inited_param = None
        self._sliced = False
        self.is_param_ps = False
        self.push_weight_to_server = False
        self.pull_weight_from_server = False
        self.requires_aggr = True
        self._cast_type = None
        self._unique = False
        self.is_in_parallel = _is_in_parallel_mode()
        self.is_in_shard = False
        self._pipeline_stage_list = []
        self.slice_num = 1
        if -1 in self.shape:
            raise ValueError(f"All shape elements of the Parameter must be positive. But got None.")
        if isinstance(default_input, (Tensor_, Tensor)):
            # At embedding cache scenes, we need limit the size of memory for parameter.
            # And save out range data to persistent storage to support TB-Level size parameter.
            slice_num_of_persistent_data = get_slice_num(default_input.dtype, default_input.shape)
            if slice_num_of_persistent_data > 1:
                data_shape = list(default_input.shape)
                slice_first_dim = math.ceil(data_shape[0] / slice_num_of_persistent_data)
                data_shape[0] = slice_first_dim
                self.param_info.use_persistent_storage = True
                self.param_info.origin_shape = default_input.shape
                self.slice_num = slice_num_of_persistent_data
                Tensor_.__init__(self, default_input.dtype, tuple(data_shape))
            else:
                Tensor_.__init__(self, default_input.dtype, default_input.shape)

        elif isinstance(default_input, int):
            Tensor_.__init__(self, mstype.int64, ())
        elif isinstance(default_input, float):
            Tensor_.__init__(self, mstype.float32, ())
        elif isinstance(default_input, (np.ndarray, list)):
            Tensor_.__init__(self, default_input)
        else:
            raise TypeError(f"The type of the argument 'default_input' must be in ['Tensor', 'int', 'float',"
                            f" 'numpy.ndarray', 'list']. But got type {type(default_input)}.")
        self.param_info.parameter_shape = self.shape

        import mindspore.ops.operations.other_ops as other_ops
        self.load = other_ops.Load()

    def __deepcopy__(self, memodict):
        new_obj = Parameter(self)
        new_obj.name = self.name
        new_obj._inited_param = self._inited_param
        return new_obj

    def __str__(self):
        return f'Parameter (name={self.name}, shape={self.shape}, dtype={self.dtype}, ' \
               f'requires_grad={self.requires_grad})'

    def __repr__(self):
        return self.__str__()

    def __parameter__(self):
        """For parse check."""

    @staticmethod
    def _get_base_class(input_class):
        input_class_name = Parameter.__name__
        if input_class_name in Parameter._base_type:
            new_type = Parameter._base_type.get(input_class_name)
        else:
            new_type = type(input_class_name, (Parameter, input_class), {})
            Parameter._base_type[input_class_name] = new_type
        return new_type

    @staticmethod
    def _get_parameter_new_args(data, rc):
        """Set `set_data` of current `Parameter`."""
        if isinstance(data, bool):
            raise ValueError('Parameter data can not be `bool`')
        if isinstance(data, Tensor):
            if not data.has_init:
                if rc == 4:
                    # when ref count is 4, means the input data is not referenced
                    # in other place, so we can make a Tensor without copy data.
                    return (Tensor, data)
                # make a copy of Tensor to init the parameter.
                return (Tensor, data.asnumpy())

            not_init_data = _is_role_sched() or (_is_role_pserver() and _cache_enable()) or _is_in_parallel_mode()
            if not_init_data:
                # do not init data while in auto parallel.
                return (Tensor, None, data.dtype, get_slice_shape(data.dtype, data.shape), data.init)
            return (Tensor, data.init_data())
        if isinstance(data, int):
            return (Tensor, data, mstype.int32)
        if isinstance(data, float):
            return (Tensor, data, mstype.float32)
        return (Tensor, data)

    def set_param_ps(self, init_in_server=False):
        """
        Set whether the trainable parameter is updated by parameter server and whether the
        trainable parameter is initialized on server.

        Note:
            It only works when a running task is in the parameter server mode.
            It is supported only in graph mode.

        Args:
            init_in_server (bool): Whether trainable parameter updated by parameter server is
                initialized on server. Default: False.
        """
        if not _is_ps_mode() or not (_is_role_worker() or _is_role_pserver() or _is_role_sched()):
            raise RuntimeError("Must complete following two steps before calling set_param_ps: \n"
                               "1. context.set_ps_context(enable_ps=True) \n"
                               "2. export MS_ROLE environment variable \n"
                               "Please refer to the official website for detailed usage.")

        if context.get_context("mode") == context.PYNATIVE_MODE:
            raise RuntimeError("Parameter server training is not supported in pynative mode currently."
                               "Please switch to graph mode and retry.")
        self.is_param_ps = True
        self.init_in_server = init_in_server
        self.param_info.init_in_server = init_in_server

    def copy(self):
        """
        Copy the parameter.

        Returns:
            Parameter, a new parameter.
        """
        return self.clone(init='same')

    def set_param_fl(self, push_to_server=False, pull_from_server=False, requires_aggr=True):
        """
        Set the way of parameter and server interaction.

        Args:
            push_to_server (bool): Whether the parameter should be pushed to server. Default: False.
            pull_from_server (bool): Whether the parameter should be pulled from server. Default: False.
            requires_aggr (bool): Whether the parameter should be aggregated in the server. Default: True.
        """
        if push_to_server:
            self.push_weight_to_server = True
        if pull_from_server:
            self.pull_weight_from_server = True
        if not requires_aggr:
            self.requires_aggr = False
            self.param_info.requires_aggr = False

    @property
    def inited_param(self):
        """
        Get the new parameter after call the init_data.

        Default is a None, If `self` is a Parameter without data, after call the
        `init_data` the initialized Parameter with data will be recorded here.
        """
        return self._inited_param

    @property
    def param_info(self):
        return self._param_info

    @param_info.setter
    def param_info(self, param_info_):
        param_info_.obj = self
        self._param_info = param_info_
        Tensor_.param_info.fset(self, param_info_)

    @property
    def name(self):
        """Get the name of the parameter."""
        return self.param_info.name

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
            raise ValueError("The type of the Parameter's name should be 'string' or 'None', "
                             "but got {}.".format(type(name_)))

        if _is_role_worker() and self.cache_enable:
            if len(self.shape) != 2:
                raise RuntimeError("The dims of parameter '{}' must be 2, but got {}."
                                   .format(self.name, len(self.shape)))
            _reinsert_hash_table_size(name_, self.param_info.name, self.shape[0], self.shape[1])
        self.param_info.name = name_

    @property
    def sliced(self):
        """Get slice status of the parameter."""
        return self._sliced

    @sliced.setter
    def sliced(self, sliced_):
        self._sliced = sliced_

    @property
    def comm_fusion(self):
        """
        Get the fusion type (int) for communication operators corresponding to this parameter.

        In `AUTO_PARALLEL` and `SEMI_AUTO_PARALLEL` mode, some communication operators used for parameters or
        gradients aggregation are inserted automatically. The value of fusion must be greater than or equal to 0.
        When the value of fusion is 0, operators will not be fused together.

        """
        return self.param_info.comm_fusion

    @comm_fusion.setter
    def comm_fusion(self, comm_fusion_):
        if context.get_context("mode") == context.PYNATIVE_MODE and "auto_parallel" in _get_parallel_mode():
            raise RuntimeError(
                "`comm_fusion` does not support PYNATIVE_MODE in AUTO_PARALLEL and SEMI_AUTO_PARALLEL mode.")
        Validator.check_non_negative_int(comm_fusion_)
        self.param_info.comm_fusion = comm_fusion_

    @property
    def parallel_optimizer_comm_recompute(self):
        """
        Get the communication recompute status(bool) of optimizer parallel for the parameter.

        In `AUTO_PARALLEL` and `SEMI_AUTO_PARALLEL` mode, when applying parallel optimizer,
        some :class:`mindspore.ops.AllGather` operators
        used for parameters gathering are inserted automatically. It is used to control the recompute attr for those
        :class:`mindspore.ops.AllGather` operators.

        Note:
            - Only `Graph` mode is supported.
            - It is recommended to use cell.recompute(parallel_optimizer_comm_recompute=True/False) to configure
              the AllGather operators introducing by parallel optimizer rather than using this interface directly.
        """
        return self.param_info.parallel_optimizer_comm_recompute

    @parallel_optimizer_comm_recompute.setter
    def parallel_optimizer_comm_recompute(self, parallel_optimizer_comm_recompute_):
        Validator.check_bool(parallel_optimizer_comm_recompute_)
        self.param_info.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute_

    @property
    def unique(self):
        """Whether the parameter is already unique or not."""
        return self._unique

    @unique.setter
    def unique(self, unique_):
        self._unique = unique_

    def clone(self, init='same'):
        """
        Clone the parameter.

        Args:
            init (Union[Tensor, str, numbers.Number]): Initialize the shape and dtype of the parameter.
                If `init` is a `Tensor` or `numbers.Number`, clone a new parameter with the same shape
                and dtype, and the data of the new parameter will be set according to `init`. If `init`
                is a `str`, the `init` should be the alias of the class inheriting from `Initializer`.
                For example, if `init` is 'same', clone a new parameter with the same data, shape, and
                dtype. Default: 'same'.

        Returns:
            Parameter, a new parameter.
        """
        x = copy(self)
        param_info_clone = self.param_info.clone()
        info = self.param_info
        if hasattr(info, "cloned_obj"):
            info.cloned_obj.append(x)
        else:
            info.cloned_obj = [x]
        self.param_info = info
        param_info_clone.obj = x
        x.param_info = param_info_clone
        x.is_init = False
        x.init = self.init
        x.is_param_ps = self.is_param_ps
        x.init_in_server = self.init_in_server
        x.cache_enable = self.cache_enable
        if x.cache_enable:
            x.key = _get_unique_parameter_key()
        x.requires_aggr = self.requires_aggr
        if self.cache_shape:
            x.cache_shape = self.cache_shape
        if init != 'same':
            shape = self.shape if self.slice_num == 1 else self.param_info.origin_shape
            dtype = self.dtype
            x.set_data(initializer(init, shape=shape, dtype=dtype))
        return x

    @property
    def layerwise_parallel(self):
        """
        Get the layerwise parallel status(bool) of the parameter.

        When layerwise_parallel is true in `DATA_PARALLEL` and `HYBRID_PARALLEL` parallel mode, broadcast and gradients
        communication would not be applied to parameters.
        """
        return self.param_info.layerwise_parallel

    @layerwise_parallel.setter
    def layerwise_parallel(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("The argument `layerwise_parallel` must be bool type.")
        self.param_info.layerwise_parallel = value

    @property
    def parallel_optimizer(self):
        """
        Get the optimizer parallel status(bool) of the parameter.

        It is used to filter the weight shard operation in `AUTO_PARALLEL` and `SEMI_AUTO_PARALLEL` mode. It works only
        when enable parallel optimizer in `mindspore.set_auto_parallel_context()`.
        """
        return self.param_info.parallel_optimizer

    @parallel_optimizer.setter
    def parallel_optimizer(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("The argument `parallel_optimizer` must be bool type.")
        self.param_info.parallel_optimizer = value

    @property
    def cache_enable(self):
        """Return whether the parameter is cache enable."""
        return self.param_info.cache_enable

    @cache_enable.setter
    def cache_enable(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("The argument `cache_enable` must be bool type.")
        self.param_info.cache_enable = value

    @property
    def cache_shape(self):
        """Return the cache shape corresponding to the parameter if use cache."""
        return self.param_info.cache_shape

    @cache_shape.setter
    def cache_shape(self, value):
        if not isinstance(value, (tuple, list)):
            raise TypeError("The argument `cache_shape` must be tuple or list type.")
        self.param_info.cache_shape = value

    @property
    def key(self):
        """Return the parameter unique key."""
        return self.param_info.key

    @key.setter
    def key(self, value=-1):
        """Set the parameter unique key."""
        if not isinstance(value, int):
            raise TypeError("The argument `key` must be int type.")
        self.param_info.key = value

    @property
    def requires_grad(self):
        """
        Return whether the parameter requires gradient.
        """
        return self.param_info.requires_grad

    @requires_grad.setter
    def requires_grad(self, value=True):
        if not isinstance(value, bool):
            raise TypeError("The argument `requires_grad` must be bool type")
        self.param_info.requires_grad = value

    @property
    def data(self):
        """Return the parameter object."""
        return self

    def value(self):
        """
        Return the value of parameter object.

        Examples:
            >>> from mindspore import Tensor, Parameter
            >>> import numpy as np
            >>> x = Parameter(Tensor(np.array([1, 2], dtype=np.float32)), name="param")
            >>> x_value = x.value()
            >>> print(x_value)
            [1.  2.]
        """
        return self.load(self, monad.U)

    def _update_tensor_data(self, data):
        """Update the parameter by a Tensor."""
        if isinstance(self, Tensor):
            self.init_flag = False
            self.init = None
            return self.assign_value(data)
        new_param = Parameter(data, self.name, self.requires_grad)
        new_param.param_info = self.param_info
        return new_param

    @_LogActionOnce(logger=logger, key='add_pipeline_stage')
    def add_pipeline_stage(self, stage):
        logger.warning(f"This interface may be deleted in the future.")
        if not isinstance(stage, int) or stage < 0:
            raise TypeError("`stage` must be a positive number of int type")
        self._pipeline_stage_list.append(stage)

    def _raise_type_error(self, incoming):
        raise TypeError(f"Incoming Parameter dtype can not be converted to current dtype implicitly. "
                        f"Current dtype is {self.dtype}, and incoming is {incoming}. "
                        f"Use .set_dtype(xxx) to change the dtype.")

    @staticmethod
    def _set_data_check_input_valid(current_shape, data_shape, current_tensor_is_init,
                                    incoming_tensor_is_init, slice_shape=False, slice_num=1):
        if incoming_tensor_is_init and not current_tensor_is_init:
            raise TypeError("The original tensor data is initialized, but the argument 'data' is not initialized."
                            "Please initialize 'data' before call this method.")
        if tuple(current_shape) != tuple(data_shape):
            # If Slice create Parameter shape can be change.
            if not slice_shape and slice_num == 1:
                raise ValueError(f"Can not change the shape of Parameter which has been initialized."
                                 f" Current shape is {current_shape}, and incoming is {data_shape}.")

    @staticmethod
    def _from_tensor(tensor, *args, **kwargs):
        """Create a `Parameter` that data is shared from a `Tensor`."""
        if not isinstance(tensor, Tensor_):
            raise TypeError(f"The type of input must be Tensor, but got {type(tensor)}.")
        param = Tensor_.__new__(Parameter)
        Tensor_.__init__(param, tensor)
        param.init = None
        param.init_mode = None
        param.is_default_input_init = False
        Parameter.__init__(param, tensor, *args, **kwargs)
        return param

    @jit_forbidden_register
    def set_data(self, data, slice_shape=False):
        """
        Set Parameter's data.

        Args:
            data (Union[Tensor, int, float]): New data.
            slice_shape (bool): If slice the parameter is set to true, the shape is not checked for consistency.
                                Default: False.

        Returns:
            Parameter, the parameter after set data.
        """
        if not isinstance(data, (Tensor, int, float)):
            raise TypeError(f"Parameter data must be [`Tensor`, `int`, `float`] or a kind of `Tensor` "
                            f"(like `Tensor`). But with type {type(data)}.")
        if isinstance(data, (int, float)):
            if self.dtype in mstype.int_type and isinstance(data, float):
                self._raise_type_error(mstype.float_)
            data = Tensor(data, self.dtype)
        # both not init.
        incoming_tensor_is_init = isinstance(data, Tensor) and not data.has_init
        current_tensor_is_init = isinstance(self, Tensor) and not self.has_init
        Parameter._set_data_check_input_valid(self.shape, data.shape, current_tensor_is_init, incoming_tensor_is_init,
                                              slice_shape, self.slice_num)
        if self.dtype != data.dtype:
            if mstype.implicit_conversion_seq[self.dtype] < mstype.implicit_conversion_seq[data.dtype]:
                self._raise_type_error(data.dtype)
            else:
                from mindspore.ops import functional as F
                if isinstance(data, Tensor) and data.init is not None:
                    data.init_data()
                data = F.cast(data, self.dtype)
        if isinstance(data, Tensor) and data.has_init:
            # The parameter has been initialized, directly update by the data
            if current_tensor_is_init:
                self._update_tensor_data(data.init_data())
            else:
                # also update the related inited parameter data
                if self.inited_param is not None:
                    self.inited_param.set_data(data)
                self.init_mode = data
        elif incoming_tensor_is_init or current_tensor_is_init:
            self._update_tensor_data(data)
        self.sliced = slice_shape
        return self

    @staticmethod
    def _get_init_data_args(layout=None):
        """Get the data layout args."""
        init_data_args = ()
        if layout:
            if not isinstance(layout, tuple):
                raise TypeError("The argument 'layout' should be tuple, but got {}.".format(type(layout)))
            if len(layout) < 6:
                raise ValueError("The length of 'layout' must be larger than 5, but got {}.".format(len(layout)))
            slice_index = int(_get_slice_index(layout[0], layout[1]))
            init_data_args += (slice_index, layout[2], layout[5])
        return init_data_args

    def init_data(self, layout=None, set_sliced=False):
        """
        Initialize the parameter's data.

        Args:
            layout (Union[None, tuple]): The parameter's layout info.
                layout [dev_mat, tensor_map, slice_shape, filed_size, uniform_split, opt_shard_group]. Default: None.
                It's not None only in 'SEMI_AUTO_PARALLEL' or 'AUTO_PARALLEL' mode.

                - dev_mat (list(int)): The parameter's device matrix.
                - tensor_map (list(int)): The parameter's tensor map.
                - slice_shape (list(int)): The parameter's slice shape.
                - filed_size (int): The parameter's filed size.
                - uniform_split (bool): Whether the parameter is split evenly.
                - opt_shard_group (str): The group of the parameter while running optimizer parallel.

            set_sliced (bool): True if the parameter is set sliced after initializing the data.
                Default: False.

        Returns:
            Parameter, the `Parameter` after initializing data. If current `Parameter` was already initialized before,
            returns the same initialized `Parameter`.

        Raises:
            RuntimeError: If it is from Initializer, and parallel mode has changed after the Initializer created.
            ValueError: If the length of the layout is less than 6.
            TypeError: If `layout` is not tuple.
        """
        if self.is_default_input_init and self.is_in_parallel != _is_in_parallel_mode():
            raise RuntimeError("Must set or change parallel mode before any initializer Tensor created.")
        if self.init_mode is None:
            return self
        if self.inited_param is not None:
            return self.inited_param

        init_data_args = self._get_init_data_args(layout)

        if _is_role_sched():
            return self
        if self.init_in_server and self.is_param_ps and isinstance(self.init_mode, Tensor) and \
                self.init_mode.init is not None and _is_role_worker():
            if self.cache_enable:
                data = self.init_mode.init_data(*init_data_args)
            else:
                data = self.init_mode.init_data(0, [1])
        else:
            data = self.init_mode.init_data(*init_data_args)

        obj = self._update_tensor_data(data)
        if id(obj) != id(self):
            self._inited_param = obj
        obj.init_mode = None
        obj.sliced = set_sliced
        return obj


class ParameterTuple(tuple):
    """
    Inherited from tuple, ParameterTuple  is used to save multiple parameter.

    Note:
        It is used to store the parameters of the network into the parameter tuple collection.
    """

    def __new__(cls, iterable):
        """Create instance object of ParameterTuple."""
        data = tuple(iterable)
        ids = set()
        names = set()
        for x in data:
            if not isinstance(x, Parameter):
                raise TypeError(f"For ParameterTuple initialization, "
                                f"ParameterTuple input should be 'Parameter' collection, "
                                f"but got a {type(iterable)}. ")
            if id(x) not in ids:
                if x.name in names:
                    raise ValueError("The value {} , its name '{}' already exists. "
                                     "Please set a unique name for the parameter.".format(x, x.name))
                names.add(x.name)
                ids.add(id(x))
        return tuple.__new__(ParameterTuple, tuple(data))

    def clone(self, prefix, init='same'):
        """
        Clone the parameters in ParameterTuple element-wisely to generate a new ParameterTuple.

        Args:
            prefix (str): Namespace of parameter, the prefix string will be added to the names of parameters
                in parametertuple.

            init (Union[Tensor, str, numbers.Number]): Clone the shape and dtype of Parameters in ParameterTuple and
                set  data according to `init`. Default: 'same'.

                - If `init` is a `Tensor` , set the new Parameter data to the input Tensor.
                - If `init` is `numbers.Number` , set the new Parameter data to the input number.
                - If `init` is a `str`, data will be set according to the initialization method of the same name in
                  the `Initializer`.
                - If `init` is 'same', the new Parameter has the same value with the original Parameter.


        Returns:
            Tuple, the new Parameter tuple.
        """
        Validator.check_str_by_regular(prefix)
        new = []
        for x in self:
            x1 = x.clone(init)
            x1.name = prefix + "." + x1.name
            new.append(x1)

            if not x1.cache_enable:
                continue

            if _is_role_worker():
                _clone_hash_table(x.name, x.key, x1.name, x1.key)
                _insert_accumu_init_info(x1.name, init_to_value(init))
        return ParameterTuple(new)

    def __parameter_tuple__(self):
        """For parse check."""
