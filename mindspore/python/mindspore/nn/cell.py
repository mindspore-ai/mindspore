# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
"""cell"""
from __future__ import absolute_import

__all__ = [
    "register_cell_buffer_registration_hook",
]

import inspect
import os
import time
import warnings
import itertools
from collections import OrderedDict, namedtuple
from typing import (
    Dict,
    Optional,
    Callable,
    List,
    Tuple,
    Iterator,
    Any,
    TypeVar,
    Mapping
)

import weakref
import mindspore as ms
from mindspore._checkparam import args_type_check, check_hook_fn
from mindspore.graph.dynamic_shape._auto_dynamic import is_auto_dynamic, convert_inputs_to_dynamic
from mindspore import log as logger
from mindspore.common.hook_handle import HookHandle, _update_hook_version
from mindspore import context, ops
from mindspore._c_expression import init_pipeline, update_func_graph_hyper_params, Cell_, FuncGraph, MixedPrecisionType
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.graph.api import _cell_graph_executor, _pynative_executor, _get_args_for_run, cells_compile_cache, \
    _no_grad, _get_mutable_flags
from mindspore.graph.api import _convert_python_data
from mindspore.graph.api import _process_dyn_args, _generate_dyn_compile_args
from mindspore.common.parameter import _Buffer, Parameter, ParameterTuple, _is_parameter_generated
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations import _inner_ops as inner
from mindspore.parallel.shard import Shard
from mindspore.parallel._utils import _init_auto_parallel_context, _clear_auto_parallel_context
from mindspore.graph._utils.check_jit_forbidden_api import jit_forbidden_register
from mindspore.common._register_for_recompute import recompute_registry
from mindspore.graph.jit_config import JitConfig

_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_EXTRA_STATE_KEY_SUFFIX = "_extra_state"


class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]), ):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def register_cell_buffer_registration_hook(hook: Callable[..., None], ):
    r"""Register a buffer registration hook common to all cells.

    .. warning ::

        This adds global state to the `nn.Cell` cell

    The hook will be called every time :func:`register_buffer` is invoked.
    It should have the following signature::

        hook(cell, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        A handle that can be used to remove the added hook by calling
        `handle.remove()`.
    """
    handle = HookHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.handle_id] = hook
    return handle


class Cell(Cell_):
    """
    The basic building block of neural networks in MindSpore. The model or neural network layer should inherit this
    base class.

    Layers in `mindspore.nn` are also the subclass of Cell, such as :class:`mindspore.nn.Conv2d`,
    and :class:`mindspore.nn.ReLU`, etc. Cell will be compiled into a calculation
    graph in GRAPH_MODE (static graph mode) and used as the basic module of neural networks in
    PYNATIVE_MODE (dynamic graph mode).

    .. note::
        Cell is the inference mode by default. For a class that inherits a Cell,
        if the training and inference have different structures, the subclass performs the inference branch by default.
        To set the training mode, refer to :func:`mindspore.nn.Cell.set_train` .

    .. warning::
        In the subclass of Cell, it's not allowed to define a method named 'cast' and not allowed to define an attribute
        named 'phase' or 'cells', otherwise, an error will be raised.

    Args:
        auto_prefix (bool, optional): Whether to automatically generate NameSpace for Cell and its child cells. It also
                      affects the names of parameters in the `Cell`. If set to ``True`` , the parameter name will be
                      automatically prefixed, otherwise not. In general, the backbone network should be set to
                      ``True`` , otherwise the duplicate name problem will appear. The cell to train the backbone
                      network, such as optimizer and :class:`mindspore.nn.TrainOneStepCell`, should be set to
                      ``False`` , otherwise the parameter name in backbone will be changed by mistake.
                      Default: ``True`` .
        flags (dict, optional): Network configuration information, currently it is used for the binding of network
                      and dataset. Users can also customize network attributes by this parameter. Default: ``None`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
        >>> class MyCell(nn.Cell):
        ...     def __init__(self, forward_net):
        ...         super(MyCell, self).__init__(auto_prefix=False)
        ...         self.net = forward_net
        ...         self.relu = ops.ReLU()
        ...
        ...     def construct(self, x):
        ...         y = self.net(x)
        ...         return self.relu(y)
        >>>
        >>> inner_net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> my_net = MyCell(inner_net)
        >>> print(my_net.trainable_params())
        ... # If the 'auto_prefix' set to True or not set when call the '__init__' method of the parent class,
        ... # the parameter's name will be 'net.weight'.
        [Parameter (name=weight, shape=(240, 120, 4, 4), dtype=Float32, requires_grad=True)]
    """

    IGNORE_LIST = ['_scope', '_cell_init_args', '_auto_prefix', '_cells', '_params', '_create_time',
                   '_func_graph_flags', '_parameter_layout_dict', '_params_list', '_phase', '_bprop_debug',
                   '_forward_pre_hook', '_forward_hook', '_backward_pre_hook', '_backward_hook',
                   '_cell_backward_pre_hook', '_cell_backward_hook', '_param_prefix',
                   'requires_grad', 'cell_type', '_in_strategy', '_out_strategy']
    total_instance_count = 0
    _buffers: Dict[str, Optional[Tensor]]
    global_cells = weakref.WeakKeyDictionary()
    _no_auto_lazy_inline = True

    def __new__(class_, *args, **kwargs):
        # Use class_ to avoid name conflicts with input args and kwargs.
        this = Cell_.__new__(class_, *args, **kwargs)
        if Cell._no_auto_lazy_inline:
            return this

        Cell.global_cells[this] = (class_, args, kwargs)
        return this

    def __init__(self, auto_prefix=True, flags=None):
        Cell_.__init__(self, self._cell_tag)
        Cell.total_instance_count += 1
        super().__setattr__("_params", OrderedDict())
        super().__setattr__("_cells", OrderedDict())
        super().__setattr__("_buffers", {})
        super().__setattr__("_params_list", OrderedDict())
        super().__setattr__("_primitives", OrderedDict())

        super().__setattr__("_lazy_non_persistent_buffers_set", None)
        super().__setattr__("_lazy_state_dict_hooks", None)
        super().__setattr__("_lazy_state_dict_pre_hooks", None)
        super().__setattr__("_lazy_load_state_dict_pre_hooks", None)
        super().__setattr__("_lazy_load_state_dict_post_hooks", None)
        super().__setattr__("training", False)
        super().__setattr__("requires_grad", False)
        super().__setattr__("is_top_cell", False)
        super().__setattr__("_param_prefix", '')
        super().__setattr__("_auto_prefix", auto_prefix)
        super().__setattr__("_scope", None)
        super().__setattr__("_phase", 'train')
        super().__setattr__("_compile_phase", None)
        super().__setattr__("_parameter_layout_dict", None)
        super().__setattr__("_parallel_parameter_name_list", None)
        super().__setattr__("_parallel_parameter_merge_net_dict", None)
        super().__setattr__("_create_time", int(time.time() * 1e9))
        super().__setattr__("arguments_key", "")
        super().__setattr__("_compile_cache", None)
        super().__setattr__("_phase_cache", None)
        cells_compile_cache[id(self)] = self.compile_cache
        super().__setattr__("_id", 1)
        super().__setattr__("_exist_objs", None)
        super().__setattr__("_exist_names", None)
        super().__setattr__("_recompute_cell", None)
        super().__setattr__("mixed_precision_type", None)
        super().__setattr__("_lazy_construct_sig", None)
        super().__setattr__("_jit_graph_name", '')
        super().__setattr__("_compiled", False)
        init_pipeline()

        # call gc to release GE session resources used by non-used cell objects
        if os.getenv('GC_COLLECT_IN_CELL') == '1':
            logger.warning("The convenient environment 'GC_COLLECT_IN_CELL' is deprecated from version 2.5 "
                           "and will be removed in a future version.")

        if flags:
            self.add_flags(**flags)
        super().__setattr__("_bprop_debug", False)

        # hook
        super().__setattr__("_lazy_forward_pre_hook", None)
        super().__setattr__("_lazy_forward_hook", None)
        super().__setattr__("_lazy_backward_pre_hook", None)
        super().__setattr__("_lazy_backward_hook", None)
        super().__setattr__("_lazy_forward_pre_hook_with_kwargs", None)
        super().__setattr__("_lazy_forward_hook_with_kwargs", None)
        super().__setattr__("_cell_backward_pre_hook", None)
        super().__setattr__("_cell_backward_hook", None)
        super().__setattr__("_is_recursion_hook", False)

        super().__setattr__("cell_type", None)
        super().__setattr__("_has_config_recompute", False)
        super().__setattr__("_lazy_user_parameters", None)
        super().__setattr__("_dynamic_shape_inputs", None)
        super().__setattr__("_has_mutable_args_list", None)
        super().__setattr__("_jit_config_dict", {})
        super().__setattr__("grad_ops_label", False)
        super().__setattr__("_is_check_and_refresh", False)
        super().__setattr__("_amp_level", "")
        super().__setattr__("_init_flag", False)
        super().__setattr__("_shard_fn", None)
        super().__setattr__("_in_strategy", None)
        super().__setattr__("_out_strategy", None)
        super().__setattr__("has_bprop", False)
        super().__setattr__("_saved_tensors_pack_hook", None)
        super().__setattr__("_saved_tensors_unpack_hook", None)

        if hasattr(self, "bprop"):
            super().__setattr__("has_bprop", True)

    def __getstate__(self):
        base = Cell_.__getstate__(self)
        return base, self.__dict__

    def __setstate__(self, state):
        base, dict_ = state
        Cell_.__setstate__(self, base)
        self.__dict__ = dict_

    def __bool__(self):
        return True

    @property
    def _cell_tag(self):
        # `<class 'xxxxxxx'>` to `xxxxxxx`
        return str(self.__class__)[8:-2]

    @property
    def create_time(self):
        return self._create_time

    @property
    def _non_persistent_buffers_set(self):
        """_non_persistent_buffers_set"""
        if self._lazy_non_persistent_buffers_set is None:
            super().__setattr__("_lazy_non_persistent_buffers_set", set())
        return self._lazy_non_persistent_buffers_set

    @property
    def _state_dict_hooks(self):
        """_state_dict_hooks"""
        if self._lazy_state_dict_hooks is None:
            super().__setattr__("_lazy_state_dict_hooks", OrderedDict())
        return self._lazy_state_dict_hooks

    @property
    def _state_dict_pre_hooks(self):
        """_state_dict_pre_hooks"""
        if self._lazy_state_dict_pre_hooks is None:
            super().__setattr__("_lazy_state_dict_pre_hooks", OrderedDict())
        return self._lazy_state_dict_pre_hooks

    @property
    def _load_state_dict_pre_hooks(self):
        """_load_state_dict_pre_hooks"""
        if self._lazy_load_state_dict_pre_hooks is None:
            super().__setattr__("_lazy_load_state_dict_pre_hooks", OrderedDict())
        return self._lazy_load_state_dict_pre_hooks

    @property
    def _load_state_dict_post_hooks(self):
        """_load_state_dict_post_hooks"""
        if self._lazy_load_state_dict_post_hooks is None:
            super().__setattr__("_lazy_load_state_dict_post_hooks", OrderedDict())
        return self._lazy_load_state_dict_post_hooks

    @property
    def compile_cache(self):
        """compile_cache"""
        if self._compile_cache is None:
            super().__setattr__("_compile_cache", set())
        return self._compile_cache

    @property
    def phase_cache(self):
        """phase_cache"""
        if self._phase_cache is None:
            super().__setattr__("_phase_cache", {})
        return self._phase_cache

    @property
    def _forward_pre_hook(self):
        """_forward_pre_hook"""
        if self._lazy_forward_pre_hook is None:
            super().__setattr__("_lazy_forward_pre_hook", OrderedDict())
        return self._lazy_forward_pre_hook

    @property
    def _forward_hook(self):
        """_forward_hook"""
        if self._lazy_forward_hook is None:
            super().__setattr__("_lazy_forward_hook", OrderedDict())
        return self._lazy_forward_hook

    @property
    def _backward_pre_hook(self):
        """_backward_pre_hook"""
        if self._lazy_backward_pre_hook is None:
            super().__setattr__("_lazy_backward_pre_hook", OrderedDict())
        return self._lazy_backward_pre_hook

    @property
    def _backward_hook(self):
        """_backward_hook"""
        if self._lazy_backward_hook is None:
            super().__setattr__("_lazy_backward_hook", OrderedDict())
        return self._lazy_backward_hook

    @property
    def _forward_pre_hook_with_kwargs(self):
        """_backward_hook"""
        if self._lazy_forward_pre_hook_with_kwargs is None:
            super().__setattr__("_lazy_forward_pre_hook_with_kwargs", OrderedDict())
        return self._lazy_forward_pre_hook_with_kwargs

    @property
    def _forward_hook_with_kwargs(self):
        """_backward_hook"""
        if self._lazy_forward_hook_with_kwargs is None:
            super().__setattr__("_lazy_forward_hook_with_kwargs", OrderedDict())
        return self._lazy_forward_hook_with_kwargs

    @property
    def _user_parameters(self):
        """_user_parameters"""
        if self._lazy_user_parameters is None:
            super().__setattr__("_lazy_user_parameters", [])
        return self._lazy_user_parameters

    @_user_parameters.setter
    def _user_parameters(self, value):
        """_user_parameters"""
        if not isinstance(value, list):
            raise TypeError(f"For 'Cell', the property '_user_parameters' must be list type, "
                            f"but got type {type(value)}.")
        self._lazy_user_parameters = value

    @property
    def cell_init_args(self):
        return self._cell_init_args

    @property
    def exist_names(self):
        """
        Get exist parameter names adding by tuple or list of parameter.
        """
        if self._exist_names is None:
            super().__setattr__("_exist_names", set(""))
        return self._exist_names

    @property
    def exist_objs(self):
        if self._exist_objs is None:
            super().__setattr__("_exist_objs", set())
        return self._exist_objs

    @property
    def _construct_sig(self):
        if self._lazy_construct_sig is None:
            super().__setattr__("_lazy_construct_sig", inspect.signature(self.construct))
        return self._lazy_construct_sig

    @property
    def param_prefix(self):
        """
        Param prefix is the prefix of current cell's direct child parameter.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.dense = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         x = self.dense(x)
            ...         return x
            >>> net = Net()
            >>> net.update_cell_prefix()
            >>> print(net.dense.param_prefix)
            dense
        """
        return self._param_prefix

    @property
    def bprop_debug(self):
        """
        Get whether cell custom bprop debug is enabled.
        """
        return self._bprop_debug

    @property
    def compiled(self):
        """
        Get whether `Cell` is compiled in graph mode.
        """
        return self._compiled

    @bprop_debug.setter
    def bprop_debug(self, value):
        """
        Set whether to enable cell custom bprop debug.

        Note:
            When bprop is defined in cell, the bprop function will be executed
            in python interpreter when bprop debug is true, and will be parsed
            and add to graph when bprop debug is false.

        Args:
            value (bool): Specifies whether to enable bprop debug. Default: ``False``.
        """
        if not isinstance(value, bool):
            raise TypeError(f"For 'Cell', the property 'bprop_debug' must be bool type, but got type {type(value)}.")
        self._bprop_debug = value

    def update_cell_prefix(self):
        """
        Update the `param_prefix` of all child cells.

        After being invoked, it can get all the cell's children's name prefix by '_param_prefix'.
        """
        cells_name = self.cells_and_names()

        for cell_name, cell in cells_name:
            cell._param_prefix = cell_name

    def update_cell_type(self, cell_type):
        """
        The current cell type is updated when a quantization aware training network is encountered.

        After being invoked, it can set the cell type to 'cell_type'.

        Args:
            cell_type(str): The type of cell to be updated, cell_type can be "quant" or "second-order".
        """
        self.cell_type = cell_type

    @cell_init_args.setter
    def cell_init_args(self, value):
        if not isinstance(value, str):
            raise TypeError(f"For 'Cell', the property 'cell_init_args' must be string type, "
                            f"but got type {type(value)}.")
        self._cell_init_args = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, str):
            raise TypeError(f"For 'Cell', the property 'phase' must be string type, but got type {type(value)}.")
        self._phase = value

    @property
    def compile_phase(self):
        return self._compile_phase

    @compile_phase.setter
    def compile_phase(self, value):
        if not isinstance(value, str):
            raise TypeError(f"For 'Cell', 'compile_phase' must be string type, but got type {type(value)}.")
        self._compile_phase = value
        for cell in self._cells.values():
            if cell is not None:
                cell.compile_phase = value

    @property
    def parameter_layout_dict(self):
        """
        `parameter_layout_dict` represents the tensor layout of a parameter, which is inferred by shard strategy and
        distributed operator information.
        """
        if self._parameter_layout_dict is None:
            super().__setattr__("_parameter_layout_dict", {})
        return self._parameter_layout_dict

    @property
    def cls_name(self):
        return self.__class__.__name__

    @parameter_layout_dict.setter
    def parameter_layout_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"For 'Cell', the property 'parameter_layout_dict' must be dict type, "
                            f"but got type {type(value)}.")
        self._parameter_layout_dict = value

    @property
    def parallel_parameter_name_list(self):
        if self._parallel_parameter_name_list is None:
            super().__setattr__("_parallel_parameter_name_list", ())
        return self._parallel_parameter_name_list

    @parallel_parameter_name_list.setter
    def parallel_parameter_name_list(self, value):
        if not isinstance(value, list):
            raise TypeError(f"For 'Cell', the property 'parallel_parameter_name_list' must be list type, "
                            f"but got type {type(value)}.")
        self._parallel_parameter_name_list = value

    @property
    def pipeline_stage(self):
        """
        `pipeline_stage` represents the pipeline stage of current Cell.
        """
        return self._pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        """
        Set the `pipeline_stage` of a Cell.

        Args:
            value (int): The pipeline stage of a parameter.

        Raises:
            TypeError: If `value` is not int type or is a bool type.
            ValueError: If `value` is not a positive integer.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("For 'Cell', the property 'pipeline_stage' "
                            "must be int type, but got type : {}".format(type(value)))

        if value < 0:
            raise ValueError("For 'Cell', the property 'pipeline_stage' "
                             "can not be less than 0, but got {}".format(value))
        self._pipeline_stage = value

    @property
    def pipeline_segment(self):
        """
        `pipeline_segment` represents the pipeline segment of current Cell.
        """
        return self._pipeline_segment

    @pipeline_segment.setter
    def pipeline_segment(self, value):
        """
        Set the `pipeline_segment` of a Cell. Only effective in zero_bubble_v scheduler.

        Args:
            value (int): The pipeline segment of a parameter.

        Raises:
            TypeError: If `value` is not int type or is a bool type.
            ValueError: If `value` is not a positive integer.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("For 'context.set_auto_parallel_context', the argument 'pipeline_stages' "
                            "must be int type, but got type : {}".format(type(value)))

        if value < 0:
            raise ValueError("For 'context.set_auto_parallel_context', the argument 'pipeline_stages' "
                             "can not be less than 0, but got {}".format(value))
        self._pipeline_segment = value

    @property
    def parallel_parameter_merge_net_dict(self):
        if self._parallel_parameter_merge_net_dict is None:
            super().__setattr__("_parallel_parameter_merge_net_dict", {})
        return self._parallel_parameter_merge_net_dict

    @parallel_parameter_merge_net_dict.setter
    def parallel_parameter_merge_net_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError(f"For 'Cell', the property 'parallel_parameter_merge_net_dict' must be dict type, "
                            f"but got type {type(value)}.")
        self._parallel_parameter_merge_net_dict = value

    @property
    def jit_config_dict(self):
        return self._jit_config_dict

    @property
    def enable_backward_hook(self):
        return self._enable_backward_hook

    @jit_forbidden_register
    def register_buffer(
            self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        r"""Add a buffer to the cell.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's `running_mean`
        is not a parameter, but is part of the cell's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting `persistent` to ``False`` . The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this cell's :attr:`state_dict` .

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this cell using the given name.
            tensor (Tensor): Buffer to be registered. If ``None`` ,
                the buffer is not included in the cell's :attr:`state_dict` .
            persistent (bool, optional): Whether the buffer is part of this cell's :attr:`state_dict`. Default ``True``.

        Examples:
            >>> import mindspore
            ...
            >>> class Net(mindspore.nn.Cell):
            ...    def __init__(self):
            ...        super().__init__()
            ...        self.register_buffer("buffer0", mindspore.tensor([1, 2, 3]))
            ...
            ...    def construct(self, x):
            ...        return x + self.net_buffer
            ...
            >>> net = Net()
            >>> net.register_buffer("buffer0", mindspore.tensor([4, 5, 6]))
            >>> print(net.buffer0)
            [4 5 6]
        """

        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Cell.__init__() call")
        if not isinstance(name, str):
            raise TypeError(
                f"buffer name should be a string.But got this type: {type(name)}"
            )
        if "." in name:
            raise KeyError('buffer name can\'t contain "."')
        if name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                f"cannot assign '{type(tensor)}' object to buffer '{name}' "
                "(mindspore Tensor or None required)"
            )
        for hook in _global_buffer_registration_hooks.values():
            output = hook(self, name, tensor)
            if output is not None:
                tensor = output
        if tensor is not None:
            tensor._is_buffer = True
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    @jit_forbidden_register
    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by `target` if it exists, otherwise throw an error.

        See the docstring for `get_sub_cell` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify `target` .

        Args:
            target (str): The fully-qualified string name of the buffer
                to look for. (See `get_sub_cell` for how to specify a
                fully-qualified string.)

        Returns:
            Tensor

        Examples:
            >>> import mindspore
            ...
            ...
            >>> class NetC(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_c", mindspore.tensor([0, 0, 0]))
            ...
            ...     def construct(self, x):
            ...         return x + self.buffer_c
            ...
            ...
            >>> class NetB(mindspore.nn.Cell):
            ...     def __init__(self, net_c):
            ...         super().__init__()
            ...         self.net_c = net_c
            ...         self.register_buffer("buffer_b", mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return self.net_c(x) + self.buffer_b
            ...
            ...
            >>> class NetA(mindspore.nn.Cell):
            ...     def __init__(self, net_b):
            ...         super().__init__()
            ...         self.net_b = net_b
            ...         self.register_buffer("buffer_a", mindspore.tensor([4, 5, 6]))
            ...
            ...     def construct(self, x):
            ...         return self.net_b(x) + self.buffer_a
            ...
            ...
            >>> net_c = NetC()
            >>> net_b = NetB(net_c)
            >>> net_a = NetA(net_b)
            >>> buffer_c = net_a.get_buffer("net_b.net_c.buffer_c")
            >>> print(f'buffer_c is {buffer_c}')
            buffer_c is [0 0 0]

        """
        cell_path, _, buffer_name = target.rpartition(".")

        cell = self.get_sub_cell(cell_path)

        if not hasattr(cell, buffer_name):
            raise AttributeError(
                cell._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer = getattr(cell, buffer_name)

        if buffer_name not in cell._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    @jit_forbidden_register
    def named_buffers(
            self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        r"""Return an iterator over cell buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str, optional): prefix to prepend to all buffer names. Default ``""``.
            recurse (bool, optional): if ``True`` , then yields buffers of this cell
                and all sub cells. Otherwise, yields only buffers that
                are direct members of this cell. Default ``True``.
            remove_duplicate (bool, optional): Whether to remove the duplicated buffers in the result. Default ``True``.

        Returns:
            Iterator[Tuple[str, Tensor]], an iterator of tuple containing the name and buffer.

        Examples:
            >>> import mindspore
            ...
            ...
            >>> class NetB(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_b", mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return x + self.buffer_b
            ...
            ...
            >>> class NetA(mindspore.nn.Cell):
            ...     def __init__(self, net_b):
            ...         super().__init__()
            ...         self.net_b = net_b
            ...         self.register_buffer("buffer_a", mindspore.tensor([4, 5, 6]))
            ...
            ...     def construct(self, x):
            ...         return self.net_b(x) + self.buffer_a
            ...
            ...
            >>> net_b = NetB()
            >>> net_a = NetA(net_b)
            >>>
            >>> for name, buffer in net_a.named_buffers():
            ...     print(f'buffer name is {name}, buffer is {buffer}')
            buffer name is buffer_a, buffer is [4 5 6]
            buffer name is net_b.buffer_b, buffer is [1 2 3]

        """
        gen = self._named_members(
            lambda cell: cell._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    @jit_forbidden_register
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over cell buffers.

        Args:
            recurse (bool, optional): If ``True`` , then yields buffers of this cell
                and all sub cells. Otherwise, yields only buffers that
                are direct members of this cell. Default ``True``.

        Returns:
            Iterator[Tensor], an iterator of buffer.

        Examples:
            >>> import mindspore
            ...
            ...
            >>> class NetB(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_b", mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return x + self.buffer_b
            ...
            ...
            >>> class NetA(mindspore.nn.Cell):
            ...     def __init__(self, net_b):
            ...         super().__init__()
            ...         self.net_b = net_b
            ...         self.register_buffer("buffer_a", mindspore.tensor([4, 5, 6]))
            ...
            ...     def construct(self, x):
            ...         return self.net_b(x) + self.buffer_a
            ...
            ...
            >>> net_b = NetB()
            >>> net_a = NetA(net_b)
            >>>
            >>> for buffer in net_a.buffers():
            ...     print(f'buffer is {buffer}')
            buffer is [4 5 6]
            buffer is [1 2 3]

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def _named_members(self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True):
        r"""Help yield various names + members of cells."""
        memo = set()
        cells = (
            self.cells_and_names(name_prefix=prefix)
            if recurse
            else [(prefix, self)]
        )
        for cell_prefix, cell in cells:
            members = get_members_fn(cell)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = cell_prefix + ("." if cell_prefix else "") + k
                yield name, v

    @jit_forbidden_register
    def get_sub_cell(self, target: str) -> "Cell":
        """Return the sub cell given by `target` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Cell`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): NetB(
                    (net_c): NetC(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (dense): Dense(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Cell`` ``A``. ``A`` has a nested
        sub cell ``net_b``, which itself has two sub cells ``net_c``
        and ``dense``. ``net_c`` then has a sub cell ``conv``.)

        To check whether we have the ``dense`` sub cell, we
        would call `get_sub_cell("net_b.dense")`. To check whether
        we have the ``conv`` sub cell, we would call
        `get_sub_cell("net_b.net_c.conv")`.

        The runtime of ``get_sub_cell`` is bounded by the degree
        of cell nesting in `target`. A query against
        `name_cells` achieves the same result, but it is O(N) in
        the number of transitive cells. So, for a simple check to see
        if some sub cells exist, ``get_sub_cell`` should always be
        used.

        Args:
            target (str): The fully-qualified string name of the sub cell
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            Cell

        Examples:
            >>> import mindspore
            ...
            ...
            >>> class NetC(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_c", mindspore.tensor([0, 0, 0]))
            ...         self.dense_c = mindspore.nn.Dense(5, 3)
            ...
            ...     def construct(self, x):
            ...         return self.dense_c(x) + self.buffer_c
            ...
            ...
            >>> class NetB(mindspore.nn.Cell):
            ...     def __init__(self, net_c):
            ...         super().__init__()
            ...         self.net_c = net_c
            ...         self.register_buffer("buffer_b", mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return self.net_c(x) + self.buffer_b
            ...
            ...
            >>> class NetA(mindspore.nn.Cell):
            ...     def __init__(self, net_b):
            ...         super().__init__()
            ...         self.net_b = net_b
            ...         self.register_buffer("buffer_a", mindspore.tensor([4, 5, 6]))
            ...
            ...     def construct(self, x):
            ...         return self.net_b(x) + self.buffer_a
            ...
            ...
            >>> net_c = NetC()
            >>> net_b = NetB(net_c)
            >>> net_a = NetA(net_b)
            >>> net_c = net_a.get_sub_cell("net_b.net_c")
            >>> print(f'net_c is {net_c}')
            net_c is NetC(
                (dense_c): Dense(input_channels=5, output_channels=3, has_bias=True)
            )

        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        cell = self

        for item in atoms:
            if not hasattr(cell, item):
                raise AttributeError(
                    cell._get_name() + " has no " "attribute `" + item + "`"
                )

            cell = getattr(cell, item)

            if not isinstance(cell, Cell):
                raise AttributeError("`" + item + "` is not " "an nn.Cell")

        return cell

    def get_func_graph_proto(self):
        """Return graph binary proto."""
        exec_id = ".".join([self.phase, str(self.create_time), str(id(self))])
        return _cell_graph_executor._get_func_graph_proto(self, exec_id, "anf_ir", True)

    def __getattr__(self, name):
        if '_params' in self.__dict__:
            params = self.__dict__['_params']
            if name in params:
                return params[name]
        if '_buffers' in self.__dict__:
            buffers = self.__dict__['_buffers']
            if name in buffers:
                return buffers[name]
        if '_cells' in self.__dict__:
            cells = self.__dict__['_cells']
            if name in cells:
                return cells[name]
        if '_params_list' in self.__dict__:
            params_list = self.__dict__['_params_list']
            if name in params_list:
                return params_list[name]
        raise AttributeError("The '{}' object has no attribute '{}'.".format(type(self).__name__, name))

    def __del__(self):
        if isinstance(cells_compile_cache, dict):
            # while deepcopy a cell instance, the copied cell instance can't be added to cells_compile_cache
            # here using pop(id(self), None) to avoid KeyError exception
            cells_compile_cache.pop(id(self), None)
        if hasattr(self, "compile_cache") and self.compile_cache:
            _cell_graph_executor.del_net_res(self, self.compile_cache)
        Cell.total_instance_count -= 1
        Cell.global_cells.pop(self, None)

    def __delattr__(self, name):
        if name in self._params:
            del self._params[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._cells:
            del self._cells[name]
        elif '_params_list' in self.__dict__ and name in self._params_list:
            del self._params_list[name]
        else:
            object.__delattr__(self, name)

    def cast_inputs(self, inputs, dst_type):
        """
        Cast inputs to specified type.

        .. warning::
            This interface will be deprecated in future versions.
        """
        logger.warning("'cast_inputs' will be deprecated in future versions.")

    def run_construct(self, cast_inputs, kwargs):
        """
        Run the construct function.

        Note:
            This function will be removed in a future version. It is not recommended to call this function.

        Args:
            cast_inputs (tuple): The input objects of Cell.
            kwargs (dict): Provide keyword arguments.

        Returns:
            output, the output object of Cell.
        """
        logger.warning(f"The 'run_construct' function of '{self.cls_name}' will be removed in a future version. "
                       f"Calling this function is not recommended.")
        output = self._run_construct(cast_inputs, kwargs)
        return output

    def _run_construct(self, *args, **kwargs):
        """Run the construct function"""
        if self._forward_pre_hook:
            args, kwargs = self._run_forward_pre_hook(args, kwargs)

        if self._backward_hook:
            args = self._cell_backward_hook(args)

        if self._shard_fn is not None:
            output = self._shard_fn(*args, **kwargs)
        elif _pynative_executor.requires_grad():
            if self._saved_tensors_pack_hook is not None:
                with ms.saved_tensors_hooks(self._saved_tensors_pack_hook, self._saved_tensors_unpack_hook):
                    output = self._run_grad_construct(*args, **kwargs)
            else:
                output = self._run_grad_construct(*args, **kwargs)
        else:
            output = self.construct(*args, **kwargs)

        if self._forward_hook:
            output = self._run_forward_hook(args, kwargs, output)

        if self._backward_hook:
            output = self._cell_backward_hook(output)

        if self._backward_pre_hook:
            output = self._cell_backward_pre_hook(output)

        return output

    def _run_grad_construct(self, *args, **kwargs):
        if self._recompute_cell is not None:
            output = self._recompute_cell(*args, **kwargs)
        elif self.has_bprop:
            output = self._call_custom_bprop(*args, **kwargs)
        else:
            output = self.construct(*args, **kwargs)
        return output

    def _check_construct_args(self, *args):
        """Check the args needed by the function construct"""
        positional_args = 0
        default_args = 0
        has_var = False
        for value in inspect.signature(self.construct).parameters.values():
            if value.kind is inspect.Parameter.VAR_POSITIONAL or value.kind is inspect.Parameter.VAR_KEYWORD:
                has_var = True
            if value.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if value.default is inspect.Parameter.empty:
                    positional_args += 1
                else:
                    default_args += 1

        if has_var:
            return

        if len(args) < positional_args:
            raise TypeError(f"For 'Cell', the function construct requires {positional_args} positional argument, "
                            f"but got {len(args)}. When using set_inputs, please make sure that all networks "
                            f"and loss functions are configured with set_inputs.")

        if len(args) > positional_args + default_args:
            construct_inputs_names = self.construct.__code__.co_varnames
            if 'self' not in construct_inputs_names:
                raise TypeError("For 'Cell', the method 'construct' must have parameter 'self'. ")

            raise TypeError(f"For 'Cell', the function construct requires {positional_args} positional argument and "
                            f"{default_args} default argument, total {positional_args + default_args}, "
                            f"but got {len(args)}.")

    def _get_prims_recursively(self):
        all_prims = []
        for _, value in self._primitives.items():
            if value:
                all_prims.append(value)

        for cell in self.cells():
            all_prims.extend(cell._get_prims_recursively())

        return all_prims

    def set_data_parallel(self):
        """
        For all primitive ops in this cell(including ops of cells that wrapped by this cell),
        if parallel strategy is not specified, then instead of auto-searching, data parallel
        strategy will be generated for those primitive ops.

        Note:
            Only effective while using auto_parallel_context = ParallelMode.AUTO_PARALLEL under graph mode.

        Examples:
            >>> import mindspore.nn as nn
            >>> net = nn.Dense(3, 4)
            >>> net.set_data_parallel()
        """
        all_prims = self._get_prims_recursively()
        for prim in all_prims:
            prim.add_prim_attr("strategy_gen_mode", "data_parallel")

    def offload(self, backward_prefetch="Auto"):
        """
        Set the cell offload. All primitive ops in the cell will be set offload. For the intermediate
        activations calculated by these primitive ops, we will not save them in the forward pass, but
        offload them and onload them in the backward pass.

        Note:
            - If Cell.offload is called, the mode should be set to "GRAPH_MODE".
            - If Cell.offload is called, lazyinline should be enabled.

        Args:
            backward_prefetch(Union[str, int], optional): The timing for prefetching activations in advance in backward
                                                          pass. Default: ``"Auto"``. If set it to ``"Auto"``, framework
                                                          will start to prefetch activations one operator in advance.
                                                          If set it to a positive int value, framework will start to
                                                          prefetch activations ``backward_prefetch`` operators in
                                                          advance, such as 1, 20, 100.
        Examples:
            >>> import mindspore.nn as nn
            >>> from mindspore import ops
            >>> from mindspore.common import Tensor, Parameter
            >>> from mindspore.graph.lazy_inline import lazy_inline
            >>>
            >>> class Block(nn.Cell):
            ...     def __init__(self):
            ...         super(Block, self).__init__()
            ...         self.transpose1 = ops.Transpose()
            ...         self.transpose2 = ops.Transpose()
            ...         self.transpose3 = ops.Transpose()
            ...         self.transpose4 = ops.Transpose()
            ...         self.real_div1 = ops.RealDiv()
            ...         self.real_div2 = ops.RealDiv()
            ...         self.batch_matmul1 = ops.BatchMatMul()
            ...         self.batch_matmul2 = ops.BatchMatMul()
            ...         self.softmax = ops.Softmax(-1)
            ...         self.expand_dims = ops.ExpandDims()
            ...         self.sub = ops.Sub()
            ...         self.y = Parameter(Tensor(np.ones((1024, 128, 128)).astype(np.float32)))
            ...     def construct(self, x):
            ...         transpose1 = self.transpose1(x, (0, 2, 1, 3))
            ...         real_div1 = self.real_div1(transpose1, Tensor(2.37891))
            ...         transpose2 = self.transpose2(x, (0, 2, 3, 1))
            ...         real_div2 = self.real_div2(transpose2, Tensor(2.37891))
            ...         batch_matmul1 = self.batch_matmul1(real_div1, real_div2)
            ...         expand_dims = self.expand_dims(self.y, 1)
            ...         sub = self.sub(Tensor([1.0]), expand_dims)
            ...         soft_max = self.softmax(sub)
            ...         transpose3 = self.transpose3(x, (0, 2, 1, 3))
            ...         batch_matmul2 = self.batch_matmul2(soft_max[0], transpose3)
            ...         transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
            ...         return transpose4
            >>>
            >>> class OuterBlock(nn.Cell):
            ...     @lazy_inline
            ...     def __init__(self):
            ...         super(OuterBlock, self).__init__()
            ...         self.block = Block()
            ...     def construct(self, x):
            ...         return self.block(x)
            >>>
            >>> class Nets(nn.Cell):
            ...     def __init__(self):
            ...         super(Nets, self).__init__()
            ...         self.blocks = nn.CellList()
            ...         for _ in range(3):
            ...             b = OuterBlock()
            ...             b.offload()
            ...             self.blocks.append(b)
            ...     def construct(self, x):
            ...         out = x
            ...         for i in range(3):
            ...             out = self.blocks[i](out)
            ...         return out
        """
        if isinstance(backward_prefetch, str):
            Validator.check_string(backward_prefetch, ['Auto'], 'backward_prefetch', self.cls_name)
        else:
            Validator.check_non_negative_int(backward_prefetch)
        for prim in self._get_prims_recursively():
            prim._offload(backward_prefetch=backward_prefetch)

    def shard(self, in_strategy, out_strategy=None, parameter_plan=None):
        """
        Defining the input and output layouts of this cell and the parallel strategies of remaining ops will be
        generated by sharding propagation. In Graph mode, use this method to specify distribution strategy for a Cell,
        strategy for others will be set by sharding propagation.
        in_strategy and out_strategy define the input and output layout respectively.
        in_strategy/out_strategy should be a tuple, each element of which corresponds to the desired layout of
        this input/output, which can refer to the description of :func:`mindspore.ops.Primitive.shard`.
        The parallel strategies of remaining operators are derived from the strategy specified by the input and output.

        Note:
            - It is valid only in semi auto parallel or auto parallel mode.
              In other parallel modes, strategies set here will be ignored.
            - If the input contain Parameter, its strategy should be set in `in_strategy`.

        .. warning::
            The method is currently not supported in PyNative mode.

        Args:
            in_strategy (tuple): Define the layout of inputs, each element of the tuple should be a tuple. Tuple
                                 defines the layout of the corresponding input.
            out_strategy (Union[None, tuple]): Define the layout of outputs similar with in_strategy.
                                               Default: ``None`` .
            parameter_plan (Union[dict, None]): Define the layout for the specified parameters. Each element in dict
                                                defines the layout of the parameter like "param_name: layout".
                                                The key is a parameter name of type 'str'.
                                                The value is a 1-D integer tuple, indicating the corresponding layout.
                                                If the parameter name is incorrect or the corresponding parameter
                                                has been set, the parameter setting will be ignored.
                                                Default: ``None`` .

        Examples:
            >>> import mindspore.nn as nn
            >>>
            >>> class Block(nn.Cell):
            ...   def __init__(self):
            ...     self.dense1 = nn.Dense(10, 10)
            ...     self.relu = nn.ReLU()
            ...     self.dense2 = nn.Dense2(10, 10)
            ...   def construct(self, x):
            ...     x = self.relu(self.dense2(self.relu(self.dense1(x))))
            ...     return x
            >>>
            >>> class example(nn.Cell):
            ...   def __init__(self):
            ...     self.block1 = Block()
            ...     self.block2 = Block()
            ...     self.block2.shard(in_strategy=((2, 1),), parameter_plan={'self.block2.dense1.weight': (4, 1)})
            ...   def construct(self, x):
            ...     x = self.block1(x)
            ...     x = self.block2(x)
            ...     return x
        """
        if ms.communication.management.get_group_size() == 1:
            return

        shard_fn = Shard()
        self._shard_fn = shard_fn(self, in_strategy, out_strategy, parameter_plan)

        if self._in_strategy is not None:  # pylint: disable=E0203
            msg = (
                      "For '%s', 'Shard' has been configured more than once. "
                      "The existing in_strategy is %s and the existing out_strategy is %s. "
                      "The new in_strategy %s and out_strategy %s may not take effect. "
                      "It is recommended to configure 'Shard' only once."
                  ) % (
                      self._cell_tag,
                      self._in_strategy,  # pylint: disable=E0203
                      self._out_strategy,  # pylint: disable=E0203
                      shard_fn.in_strategy,
                      shard_fn.out_strategy,
                  )
            logger.warning(msg)
        self._in_strategy = shard_fn.in_strategy
        self._out_strategy = shard_fn.out_strategy

    def _init_check(self):
        for param in self.get_parameters(expand=False):
            if param.has_init:
                param.init_data()
        self._init_flag = True

    def _self_check(self):
        try:
            if not self._is_check_and_refresh:  # pylint: disable=E0203
                self.check_names_and_refresh_name()
                self._is_check_and_refresh = True
        except AttributeError as e:
            raise AttributeError(f"The '{type(self).__name__}' object does not inherit attribute from 'cell'. "
                                 f"Please use 'super().__init__()'.") from e

    def _predict(self, *args, **kwargs):
        '''Graph executor for predict'''
        if not hasattr(self, "phase"):
            return False, None
        if (self.phase == "prefill" or self.phase == 'increment') and self.phase in self.phase_cache:
            new_args = _get_args_for_run(self, args, kwargs, self._has_mutable_args_list, self.sequence_modified,
                                         True, True)
            if self.jit_config_dict:
                jit_config_dict = self.jit_config_dict
            else:
                jit_config_dict = JitConfig().jit_config_dict
            _cell_graph_executor._graph_executor.set_jit_config(jit_config_dict)
            res = _cell_graph_executor._graph_executor(tuple(new_args), self.phase_cache[self.phase])
            res = _convert_python_data(res)
            return True, res
        return False, None

    def __call__(self, *args, **kwargs):
        # Run in Graph mode.
        if context._get_mode() == context.GRAPH_MODE and os.getenv("MS_JIT") != '0':
            self._compiled = True
            if kwargs:
                bound_arguments = self._construct_sig.bind(*args, **kwargs)
                bound_arguments.apply_defaults()
                args = bound_arguments.args
                kwargs = bound_arguments.kwargs

            predict_compiled, res = self._predict(*args, **kwargs)
            if predict_compiled:
                return res
            self._check_construct_args(*args)
            self._self_check()
            self.__compile_cell_hook__ = True
            out = self.compile_and_run(*args, **kwargs)
            return out

        # Run in PyNative mode.
        if not (self._init_flag or self._is_check_and_refresh):
            self._init_check()
            self._self_check()

        if not (self.requires_grad or self._dynamic_shape_inputs or self.mixed_precision_type):
            if not (self._forward_pre_hook or self._forward_hook or self._backward_pre_hook or self._backward_hook or
                    self._shard_fn or self._recompute_cell or (self.has_bprop and _pynative_executor.requires_grad())):
                return self.construct(*args, **kwargs)

            return self._run_construct(*args, **kwargs)

        return self._complex_call(*args, **kwargs)

    def _complex_call(self, *args, **kwargs):
        """
        PyNative call with requires_grad or hooks
        """
        self._call_pre_process(*args, **kwargs)

        if not (self._forward_pre_hook or self._forward_hook or self._backward_pre_hook or self._backward_hook or
                self._shard_fn or self._recompute_cell or self.has_bprop or self._saved_tensors_pack_hook):
            output = self.construct(*args, **kwargs)
        else:
            output = self._run_construct(*args, **kwargs)

        self._call_post_process(output, *args, **kwargs)

        return output

    def _call_pre_process(self, *args, **kwargs):
        """
        Process cell info before call construct
        """
        if self.requires_grad and (not _pynative_executor.grad_flag() or _pynative_executor.high_order()):
            self.is_top_cell = True
            _pynative_executor.set_grad_flag(True)
            _pynative_executor.new_graph(self, *args, **kwargs)
        elif self._dynamic_shape_inputs is not None:
            _pynative_executor.set_cell_use_dynamic_shape_process(True)

        # Set mixed precision
        if self.mixed_precision_type is not None:
            _pynative_executor.set_mixed_precision_type(self.mixed_precision_type)

    def _call_post_process(self, output, *args, **kwargs):
        """
        Process cell info after call construct
        """
        if self.requires_grad and self.is_top_cell:
            _pynative_executor.end_graph(self, output, *args, **kwargs)
            self.is_top_cell = False
        elif self._dynamic_shape_inputs is not None:
            _pynative_executor.set_cell_use_dynamic_shape_process(False)

        # mixed precision reset
        if self.mixed_precision_type is not None:
            _pynative_executor.set_mixed_precision_type(MixedPrecisionType.NOTSET, False)

    def _call_custom_bprop(self, *args, **kwargs):
        """
        Call custom bprop for cell bprop.
        """
        with _no_grad():
            output = self.construct(*args, **kwargs)
        if self.construct.__defaults__ or self.construct.__kwdefaults__:
            bound_arguments = self._construct_sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            kwargs = bound_arguments.kwargs
        return _pynative_executor.call_custom_bprop(self, output, *args, **kwargs)

    def _add_attr(self, name, value):
        if name and name[:2] != '__' and name not in Cell.IGNORE_LIST:
            super()._add_attr(name, value)

    def _set_attr_for_param_or_param_tuple(self, name, value):
        """Set attr for param and tensor."""
        if isinstance(value, Parameter):
            if name in self.__dict__:
                del self.__dict__[name]
            self.insert_param_to_cell(name, value)
        elif isinstance(value, ParameterTuple):
            exist_names = set("")
            exist_objs = set()
            for item in value:
                if item in exist_objs:
                    # If there are multiple identical objects, their names only check once.
                    continue
                exist_objs.add(item)
                if _is_parameter_generated(item.name):
                    item.name = "Parameter$" + str(self._id)
                    self._id += 1
                if item.name in exist_names:
                    raise ValueError("The value {} , its name '{}' already exists. "
                                     "Please set a unique name for the parameter.".format(value, item.name))
                exist_names.add(item.name)
                self.insert_param_to_cell(item.name, item, check_name_contain_dot=False)

            object.__setattr__(self, name, value)

    def _set_attr_for_parameter_in_list_or_tuple(self, name, value):
        """Set attr for parameter in list or tuple."""
        for item in value:
            if item in self.exist_objs:
                # If there are multiple identical objects, their names only check once.
                continue
            self.exist_objs.add(item)
            if item.name in self.exist_names:
                raise ValueError(f"The value {value} , its name '{item.name}' already exists. "
                                 "Please set a unique name for the parameter.")
            self.exist_names.add(item.name)
        object.__setattr__(self, name, value)

    def _set_attr_for_cell(self, name, value):
        """Set attr for cell."""
        if name in self.__dict__:
            del self.__dict__[name]
        if self._auto_prefix:
            value.update_parameters_name(name + '.')
        self.insert_child_to_cell(name, value)
        if hasattr(self, '_cell_init_args'):
            self.cell_init_args += str({name: value})

    def _set_attr_for_params(self, name, value):
        if isinstance(value, Tensor) and self._params[name] is not None:
            self._params[name].set_data(value)
        elif value is not None:
            raise TypeError(f"For 'Cell', the type of {name} must be Parameter or ParameterTuple, "
                            f"but got {type(value).__name__}.")
        else:
            self.insert_param_to_cell(name, None)

    def _set_attr_for_object(self, name, value):
        """Set attr for py object."""
        params = self.__dict__.get('_params')
        if params is not None and name in params:
            if value is not None:
                if isinstance(value, Tensor):
                    params[name].set_data(value)
                    return
                raise TypeError(
                    f"Parameter '{name}' already exists in network, "
                    f"can not assign this type: '{type(value)}' as a parameter.")
            params[name] = None
            return
        cells = self.__dict__.get('_cells')
        if cells is not None and name in cells:
            if value is not None:
                raise TypeError(
                    f"Sub cell '{name}' already exists in network, "
                    f"can not assign this type: '{type(value)}' as a cell.")
            cells[name] = None
            return
        buffers = self.__dict__.get('_buffers')
        if buffers is not None and name in buffers:
            if value is not None:
                raise TypeError(
                    f"Buffer '{name}' already exists in network, "
                    f"can not assign this type: '{type(value)}' as a buffer.")
            buffers[name] = None
            return
        object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, ParameterTuple)):
            self._set_attr_for_param_or_param_tuple(name, value)
        elif _is_parameter_list_or_tuple(value):
            self._set_attr_for_parameter_in_list_or_tuple(name, value)
        elif isinstance(value, Cell):
            self._set_attr_for_cell(name, value)
        elif isinstance(value, _Buffer):
            if name in self.__dict__:
                del self.__dict__[name]
            self.register_buffer(name, value)
        elif isinstance(value, Primitive):
            value.set_prim_instance_name(name)
            self._primitives[name] = value
            object.__setattr__(self, name, value)
        else:
            self._set_attr_for_object(name, value)

    def _get_name(self):
        return self.__class__.__name__

    def extend_repr(self):
        """
        Expand the description of Cell.

        To print customized extended information, re-implement this method in your own cells.
        """
        return ''

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        extra_lines = []
        extend_repr = self.extend_repr()
        # empty string will be split into list ['']
        if extend_repr:
            extra_lines = extend_repr.split("\n")
        child_lines = []
        for key, cell in self._cells.items():
            cell_str = repr(cell)
            cell_str = _addindent(cell_str, 2)
            child_lines.append("(" + key + "): " + cell_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def set_inputs(self, *inputs, **kwargs):
        """
        Save set inputs for computation graph. The number of inputs should be the same with that of the datasets. When
        using Model for dynamic shape, please make sure that all networks and loss functions passed to the Model are
        configured with set_inputs. The shape of input Tensor can be either dynamic or static.

        .. note::
            There are two mode:

            - Full mode: arguments will be used as all compile inputs for graph-compiling.
            - Incremental mode: arguments will set to some of the Cell inputs, which will be substituted into the input
              at the corresponding position for graph-compiling.

            Only one of inputs or kwargs can be set. Inputs for full mode and kwargs for incremental mode.

        Args:
            inputs (tuple): Full mode arguments.
            kwargs (dict): Incremental mode arguments. The acceptable key is the name of parameter defined
                in `self.construct`.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn, Tensor
            >>>
            >>> class ReluNet(nn.Cell):
            ...     def __init__(self):
            ...         super(ReluNet, self).__init__()
            ...         self.relu = nn.ReLU()
            ...     def construct(self, x):
            ...         return self.relu(x)
            >>>
            >>> net = ReluNet()
            >>> input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
            >>> net.set_inputs(input_dyn)
            >>> input = Tensor(np.random.random([3, 10]), dtype=ms.float32)
            >>> output = net(input)
            >>>
            >>> net2 = ReluNet()
            >>> net2.set_inputs(x=input_dyn)
            >>> output = net2(input)
        """
        if self.grad_ops_label:
            logger.warning('For Cell, set_inputs must be set before the gradient function of the network is '
                           'generated.')
        if kwargs and inputs:
            raise ValueError('For Cell, set_inputs should only set inputs or kwargs(inputs: %s, kwargs: %s)!'
                             % (inputs, kwargs))

        if not kwargs:
            self._dynamic_shape_inputs = inputs
            if context._get_mode() == context.PYNATIVE_MODE:
                _pynative_executor.set_dynamic_input(self, *self._dynamic_shape_inputs)
            else:
                self._check_construct_args(*inputs)
        else:
            self._dynamic_shape_inputs = _process_dyn_args(self.construct, kwargs)

    def get_inputs(self):
        """
        Returns the dynamic_inputs of a cell object in one network.

        Returns:
            inputs (tuple), Inputs of the Cell object.

        .. warning::
            This is an experimental API that is subject to change or deletion.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn, Tensor
            >>>
            >>> class ReluNet(nn.Cell):
            ...     def __init__(self):
            ...         super(ReluNet, self).__init__()
            ...         self.relu = nn.ReLU()
            ...     def construct(self, x):
            ...         return self.relu(x)
            >>>
            >>> net = ReluNet()
            >>> input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
            >>> net.set_inputs(input_dyn)
            >>> get_inputs = net.get_inputs()
            >>> print(get_inputs)
            (Tensor(shape=[3, -1], dtype=Float32, value= ),)

        """

        return self._dynamic_shape_inputs

    def _check_parameter_consistency(self, set_inputs, net_inputs):
        """Check consistency for parameter."""
        for index, (set_input, net_input) in enumerate(zip(set_inputs, net_inputs)):
            if isinstance(set_input, Tensor):
                if not isinstance(net_input, Tensor):
                    raise TypeError(
                        f"For 'set_inputs' and tuple(list) in 'set_inputs',the type of {index + 1}th input must "
                        f"be Tensor, but got {type(net_input)}.")
                if isinstance(set_input, Parameter) != isinstance(net_input, Parameter):
                    raise TypeError(
                        f"For 'set_inputs' and tuple(list) in 'set_inputs', the {index + 1}th input must be the same "
                        f"as expected, but got expected: {type(set_input)} and input: {type(net_input)}.")
            elif isinstance(set_input, (tuple, list)):
                if not isinstance(net_input, (tuple, list)):
                    raise TypeError(
                        f"The {index + 1}th input type of 'set_inputs' or tuple(list) in "
                        f"'set_inputs' must be tuple or list, but got {type(net_input)}.")
                self._check_parameter_consistency(set_input, net_input)

    def _get_compile_args(self, args):
        """Get compile arguments."""
        # this is used only for test
        set_by_auto_dynamic = False
        if is_auto_dynamic():
            if self._dynamic_shape_inputs is None:
                set_by_auto_dynamic = True
            else:
                if isinstance(self._dynamic_shape_inputs, (list, tuple)) and self._dynamic_shape_inputs[0] is None:
                    set_by_auto_dynamic = True
        if set_by_auto_dynamic:
            self._dynamic_shape_inputs = convert_inputs_to_dynamic(*args)

        if self._dynamic_shape_inputs is not None:
            logger.debug("Compiled Graph with dynamic shape")
            compile_args = _generate_dyn_compile_args(args, self._dynamic_shape_inputs)
            _cell_graph_executor._graph_executor.check_argument_consistency(compile_args, args, "set_inputs")
            self._check_parameter_consistency(compile_args, args)
            Validator.check_symbolic_shape(compile_args, args)
            return compile_args
        return args

    def compile(self, *args, **kwargs):
        """
        Compile Cell as a computation graph, the input must be consistent with the input defined in construct.

        Args:
            args (tuple): Args of the Cell object.
            kwargs (dict): Kwargs of the Cell object.
        """
        _init_auto_parallel_context(self)
        compile_args = self._get_compile_args(args)
        self._has_mutable_args_list = _get_mutable_flags(compile_args)
        self.sequence_modified = []
        _cell_graph_executor.set_real_args(args, kwargs)
        _cell_graph_executor.compile(self, *compile_args, phase=self.phase,
                                     jit_config_dict=self._jit_config_dict, **kwargs)
        _clear_auto_parallel_context(self)

    def compile_and_run(self, *args, **kwargs):
        """
        Compile and run Cell, the input must be consistent with the input defined in construct.

        Note:
            It is not recommended to call directly.

        Args:
            args (tuple): Args of the Cell object.
            kwargs (dict): Kwargs of the Cell object.

        Returns:
            Object, the result of executing.
        """
        self.compile(*args, **kwargs)
        new_args = _get_args_for_run(self, args, kwargs, self._has_mutable_args_list, self.sequence_modified, False)
        if self.jit_config_dict:
            jit_config_dict = self.jit_config_dict
        else:
            jit_config_dict = JitConfig().jit_config_dict
        _cell_graph_executor._graph_executor.set_jit_config(jit_config_dict)
        return _cell_graph_executor(self, *new_args, phase=self.phase)

    def insert_param_to_cell(self, param_name, param, check_name_contain_dot=True):
        """
        Adds a parameter to the current cell.

        Inserts a parameter with given name to the cell. The method is currently used in
        `mindspore.nn.Cell.__setattr__`.

        Args:
            param_name (str): Name of the parameter.
            param (Parameter): Parameter to be inserted to the cell.
            check_name_contain_dot (bool): Determines whether the name input is compatible. Default: ``True`` .

        Raises:
            KeyError: If the name of parameter is null or contains dot.
            TypeError: If the type of parameter is not Parameter.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn, Parameter
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...
            ...     def construct(self, x):
            ...         x = self.relu(x)
            ...         return x
            >>> net = Net()
            >>> net.insert_param_to_cell("bias", Parameter(Tensor([1, 2, 3])))
            >>> print(net.bias)
            Parameter(name=bias, shape=(3,), dtype=Int64, requires_grad=True)
        """
        if not param_name:
            raise KeyError("For 'insert_param_to_cell', the argument 'param_name' should not be None.")
        if check_name_contain_dot and '.' in param_name:
            raise KeyError("For 'insert_param_to_cell', the argument 'param_name' should not contain'.' ")
        if '_params' not in self.__dict__:
            raise AttributeError("For 'insert_param_to_cell', please call Cell.__init__() firstly.")
        if hasattr(self, param_name) and param_name not in self._params:
            raise KeyError(f"For 'insert_param_to_cell', the {param_name} parameter already exists in the network."
                           f"Cannot insert another parameter with the same name.")
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"For 'insert_param_to_cell', the argument 'param' must be 'Parameter' if not None, "
                            f"but got {type(param)}.")
        if isinstance(param, Parameter) and _is_parameter_generated(param.name):
            param.name = param_name
        self._params[param_name] = param

    def insert_child_to_cell(self, child_name, child_cell):
        """
        Adds a child cell to the current cell with a given name.

        Args:
            child_name (str): Name of the child cell.
            child_cell (Cell): The child cell to be inserted.

        Raises:
            KeyError: Child Cell's name is incorrect or duplicated with the other child name.
            TypeError: If type of `child_name` is not str.
            TypeError: Child Cell's type is incorrect.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> net1 = nn.ReLU()
            >>> net2 = nn.Dense(2, 2)
            >>> net1.insert_child_to_cell("child", net2)
            >>> print(net1)
            ReLU(
              (child): Dense(input_channels=2, output_channels=2, has_bias=True)
            )
        """
        if not isinstance(child_name, str):
            raise TypeError(f"For 'insert_child_to_cell', the type of parameter 'child_name' must be str, "
                            f"but got {type(child_name)}.")
        if not child_name or '.' in child_name:
            raise KeyError("For 'insert_child_to_cell', the parameter 'child_name' can not be None and "
                           "can not contain '.' ")
        if hasattr(self, child_name) and child_name not in self._cells:
            raise KeyError(f"For 'insert_child_to_cell', the {child_name} child cell already exists in the network."
                           f"Cannot insert another child cell with the same name.")
        if not isinstance(child_cell, Cell) and child_cell is not None:
            raise TypeError(f"For 'insert_child_to_cell', the argument 'child_cell' must be 'Cell' if not None, "
                            f"but got type {type(child_cell)}.")
        self._cells[child_name] = child_cell

    def construct(self, *args, **kwargs):
        """
        Defines the computation to be performed. This method must be overridden by all subclasses.

        Note:
            It is not supported currently that inputs contain both tuple and non-tuple types at same time.

        Args:
            args (tuple): Tuple of variable parameters.
            kwargs (dict): Dictionary of variable keyword parameters.

        Returns:
            Tensor, returns the computed result.
        """
        raise AttributeError("For 'Cell', the method 'construct' is not defined.")

    def remove_redundant_parameters(self):
        """
        Remove the redundant parameters.

        .. warning::
            This interface will be deprecated in future versions.
        """
        logger.warning("'remove_redundant_parameters' will be deprecated in future versions.")

    def _get_cell_parallel_mode(self):
        """Determine whether the current cell is in parallel mode."""
        is_parallel_mode = False
        for _, param in self.parameters_and_names():
            if param.param_info.is_param_init:
                is_parallel_mode = True
                break
        return is_parallel_mode

    def init_parameters_data(self, auto_parallel_mode=False):
        """
        Initialize all parameters and replace the original saved parameters in cell.

        Note:
            trainable_params() and other similar interfaces may return different parameter instance after
            `init_parameters_data`. It is not recommended to save these results.

        Args:
            auto_parallel_mode (bool): If running in auto_parallel_mode. Default: ``False`` .

        Returns:
            Dict[Parameter, Parameter], returns a dict of original parameter and replaced parameter.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.dense = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         x = self.dense(x)
            ...         return x
            >>> net = Net()
            >>> print(net.init_parameters_data())
            {Parameter (name=dense.weight, shape=(2,2), dtype=Float32, requires_grad=True):
             Parameter (name=dense.weight, shape=(2,2), dtype=Float32, requires_grad=True),
             Parameter (name=dense.bias, shape=(2,), dtype=Float32, requires_grad=True):
             Parameter (name=dense.bias, shape=(2,), dtype=Float32, requires_grad=True)}
        """
        replace = {}

        def _updata(param):
            if param in replace:
                return replace.get(param)
            new_p = param.init_data(None, set_sliced=param.sliced)
            replace[param] = new_p
            return new_p

        # replace all original usage.
        cells = self.cells_and_names()
        is_parallel_mode = self._get_cell_parallel_mode()

        for _, cell in cells:
            params = cell._params.items()
            for param_name, param in params:
                if param.param_info.is_pipeline_shared_param:
                    continue
                if is_parallel_mode and not param.sliced:
                    continue
                if not auto_parallel_mode:
                    cell._params[param_name] = _updata(param)
                    continue
                if param.name in self.parallel_parameter_name_list:
                    cell._params[param_name] = _updata(param)
            cell_dict = cell.__dict__
            for key in cell_dict:
                if isinstance(cell_dict[key], ParameterTuple):
                    param_tuple = cell_dict[key]
                    new_param_tuple = []
                    for param in param_tuple:
                        if param.param_info.is_pipeline_shared_param:
                            continue
                        if is_parallel_mode and not param.sliced:
                            continue
                        if not auto_parallel_mode:
                            new_param_tuple.append(_updata(param))
                            continue
                        if param.name in self.parallel_parameter_name_list:
                            new_param_tuple.append(_updata(param))
                        else:
                            new_param_tuple.append(param)
                    cell.__dict__[key] = ParameterTuple(new_param_tuple)
        return replace

    def parameters_dict(self, recurse=True):
        """
        Gets the parameters dictionary of this cell.

        Args:
            recurse (bool): Whether contains the parameters of subcells. Default: ``True`` .

        Returns:
            OrderedDict, return parameters dictionary.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn, Parameter
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.dense = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         x = self.dense(x)
            ...         return x
            >>> net = Net()
            >>> print(net.parameters_dict())
            OrderedDict([('dense.weight', Parameter(name=dense.weight, shape=(2, 2), dtype=Float32,
            requires_grad=True)), ('dense.bias', Parameter(name=dense.bias, shape=(2,), dtype=Float32,
            requires_grad=True))])
        """
        param_dict = OrderedDict()
        for param in self.get_parameters(expand=recurse):
            param_dict[param.name] = param
        return param_dict

    def parameters_broadcast_dict(self, recurse=True):
        """
        Gets the parameters broadcast dictionary of this cell.

        Args:
            recurse (bool): Whether contains the parameters of subcells. Default: ``True`` .

        Returns:
            OrderedDict, return parameters broadcast dictionary.
        """
        param_dict = OrderedDict()
        for param in self.get_parameters(expand=recurse):
            if param.layerwise_parallel is False:
                param_dict[param.name] = param
        if not param_dict:
            return None
        return param_dict

    def update_parameters_name(self, prefix='', recurse=True):
        """
        Adds the `prefix` string to the names of parameters.

        Args:
            prefix (str): The prefix string. Default: ``''`` .
            recurse (bool): Whether contains the parameters of subcells. Default: ``True`` .
        """

        Validator.check_str_and_none_by_regular(prefix)
        for name, param in self.parameters_and_names(expand=recurse):
            if prefix != '':
                param.is_init = False
            param.name = prefix + name

    def _update_local_parameters_name(self, prefix='', recurse=True):
        """
        Updates the names of local parameters with given prefix string.

        Adds the given prefix to the names of local parameters.

        Local parameters means the parameters without user input.

        Args:
            prefix (str): The prefix string. Default: ''.
            recurse (bool): Whether contains the parameters of subcells. Default: ``True``.
        """

        Validator.check_str_by_regular(prefix)
        for name, param in self.parameters_and_names(expand=recurse):
            if name in self._user_parameters:
                continue
            if prefix != '':
                param.is_init = False
            param.name = prefix + name

    @jit_forbidden_register
    def trainable_params(self, recurse=True):
        """
        Returns all trainable parameters.

        Returns a list of all trainable parameters.

        Args:
            recurse (bool): Whether contains the trainable parameters of subcells. Default: ``True`` .

        Returns:
            List, the list of trainable parameters.

        Tutorial Examples:
            - `Model Training - Optimizer
              <https://mindspore.cn/tutorials/en/master/beginner/train.html#optimizer>`_
        """
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=recurse)))

    @jit_forbidden_register
    def untrainable_params(self, recurse=True):
        """
        Returns all untrainable parameters.

        Returns a list of all untrainable parameters.

        Args:
            recurse (bool): Whether contains the untrainable parameters of subcells. Default: ``True`` .

        Returns:
            List, the list of untrainable parameters.
        """
        return list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=recurse)))

    @jit_forbidden_register
    def get_parameters(self, expand=True):
        """
        Returns an iterator over cell parameters.

        Yields parameters of this cell. If `expand` is ``true`` , yield parameters of this cell and all subcells.
        For more details about subcells, please see the example below.

        Args:
            expand (bool): If ``true`` , yields parameters of this cell and all subcells. Otherwise, only yield
                           parameters that are direct members of this cell. Default: ``True`` .

        Returns:
            Iteration, all parameters at the cell.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn, ops, Tensor
            >>> import numpy as np
            >>> class TestNet(nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.my_w1 = ms.Parameter(Tensor(np.ones([4, 4]), ms.float32))
            ...         self.my_w2 = ms.Parameter(Tensor(np.ones([16]), ms.float32))
            ...     def construct(self, x):
            ...         x += self.my_w1
            ...         x = ops.reshape(x, (16,)) - self.my_w2
            ...         return x
            >>> class TestNet2(nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.my_t1 = ms.Parameter(Tensor(np.ones([4, 4]), ms.float32))
            ...         # self.subcell is a subcell of TestNet2, when using expand=True, the parameters of TestNet will
            ...         # also be gathered.
            ...         self.subcell = TestNet()
            ...     def construct(self, x):
            ...         x += self.my_w1
            ...         x = ops.reshape(x, (16,)) - self.my_w2
            ...         return x
            >>> net = TestNet2()
            >>> print([p for p in net.get_parameters(expand=True)])
            [Parameter (name=my_t1, shape=(4, 4), dtype=Float32, requires_grad=True), Parameter (name=subcell.my_w1,
            shape=(4, 4), dtype=Float32, requires_grad=True), Parameter (name=subcell.my_w2, shape=(16,), dtype=Float32,
            requires_grad=True)]
        """
        for _, param in self.parameters_and_names(expand=expand):
            yield param

    # pylint: disable=missing-docstring
    def check_names_and_refresh_name(self):
        if not hasattr(self, "_params"):
            return
        all_name = [i.name for i in dict(self.parameters_and_names()).values()]
        if len(set(all_name)) < len(all_name):
            self.update_parameters_name()
            self.check_names()

    def check_names(self):
        """
        Check the names of cell parameters.
        """
        names = set("")
        for value, param in self.parameters_and_names():
            if param.name in names:
                raise ValueError("The value of {} is {}, its name '{}' already exists. "
                                 "Please set a unique name for the parameter.".format(value, param, param.name))
            names.add(param.name)

    def parameters_and_names(self, name_prefix='', expand=True):
        """
        Returns an iterator over cell parameters.

        Includes the parameter's name and itself.

        Args:
            name_prefix (str): Namespace. Default: ``''`` .
            expand (bool): If true, yields parameters of this cell and all subcells. Otherwise, only yield parameters
                           that are direct members of this cell. Default: ``True`` .

        Returns:
            Iteration, all the names and corresponding parameters in the cell.

        Examples:
            >>> from mindspore import nn
            >>> n = nn.Dense(3, 4)
            >>> names = []
            >>> for m in n.parameters_and_names():
            ...     if m[0]:
            ...         names.append(m[0])

        Tutorial Examples:
            - `Building a Network - Model Parameters
              <https://mindspore.cn/tutorials/en/master/beginner/model.html#model-parameters>`_
        """
        cells = []
        if expand:
            cells = self.cells_and_names(name_prefix=name_prefix)
        else:
            cells.append((name_prefix, self))

        params_set = set()
        for cell_name, cell in cells:
            params = cell._params.items()
            for par_name, par in params:
                if par is not None and par.inited_param is not None:
                    par = par.inited_param
                if par is not None and id(par) not in params_set:
                    params_set.add(id(par))
                    par_new_name = par_name
                    if cell_name:
                        par_new_name = cell_name + '.' + par_new_name

                    yield par_new_name, par

    def cells_and_names(self, cells=None, name_prefix=''):
        """
        Returns an iterator over all cells in the network, including the cell's name and itself.

        Args:
            cells (str): Cells to iterate over. Default: ``None`` .
            name_prefix (str): Namespace. Default: ``''`` .

        Returns:
            Iteration, all the child cells and corresponding names in the cell.

        Examples:
            >>> from mindspore import nn
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.conv = nn.Conv2d(3, 64, 3)
            ...     def construct(self, x):
            ...         out = self.conv(x)
            ...         return out
            >>> names = []
            >>> n = Net()
            >>> for m in n.cells_and_names():
            ...     if m[0]:
            ...         names.append(m[0])
        """
        t_cells = cells if cells else set()
        if self in t_cells:
            return

        t_cells.add(self)
        yield name_prefix, self

        for name, cell in self._cells.items():
            if cell:
                cells_name_prefix = name
                if name_prefix:
                    cells_name_prefix = name_prefix + '.' + cells_name_prefix
                yield from cell.cells_and_names(t_cells, cells_name_prefix)

    def cells(self):
        """
        Returns an iterator over immediate cells.

        Returns:
            Iteration, the immediate cells in the cell.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.dense = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         x = self.dense(x)
            ...         return x
            >>> net = Net()
            >>> print(net.cells())
            odict_values([Dense(input_channels=2, output_channels=2, has_bias=True)])
        """
        return self.name_cells().values()

    def _set_scope(self, name):
        """Sets the name on the first time."""
        if self._scope is None:
            self._scope = name
        elif self._scope == 'recompute_':
            self._scope = self._scope + name

    def _children_scope_recursive(self, parent_prefix='Default'):
        """Generates the scope of each layer of the network recursively."""
        reserve_class_name_in_scope = context.get_context("reserve_class_name_in_scope")

        for name, cell in self.name_cells().items():
            class_name = ("-" + cell.__class__.__name__) if reserve_class_name_in_scope else ""
            yield parent_prefix + "/" + name + class_name, cell

        for name, cell in self.name_cells().items():
            class_name = ("-" + cell.__class__.__name__) if reserve_class_name_in_scope else ""
            for key, value in cell._children_scope_recursive(parent_prefix + "/" + name + class_name):
                yield key, value

    def get_scope(self):
        """
        Returns the scope of a cell object in one network.

        Returns:
            String, scope of the cell.
        """
        return self._scope

    def generate_scope(self):
        """Generate the scope for each cell object in the network."""
        for name, cell in self._children_scope_recursive():
            cell._set_scope(name)

    def name_cells(self):
        """
        Returns an iterator over all immediate cells in the network.

        Include name of the cell and cell itself.

        Returns:
            Dict, all the child cells and corresponding names in the cell.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.dense = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         x = self.dense(x)
            ...         return x
            >>> net = Net()
            >>> print(net.name_cells())
            OrderedDict([('dense', Dense(input_channels=2, output_channels=2, has_bias=True))])
        """
        value_set = set()
        cells = OrderedDict()
        for name, cell in self._cells.items():
            if cell is not None and cell not in value_set:
                value_set.add(cell)
                cells[name] = cell
        return cells

    def _add_mixed_precision_flag(self, **flags):
        """Add mixed precision flag to current cell"""
        if "fp16" in flags and flags.get("fp16", False):
            self.mixed_precision_type = MixedPrecisionType.FP16
            Cell_.set_mixed_precision_type(self, MixedPrecisionType.FP16)
        if "fp32" in flags and flags.get("fp32", False):
            self.mixed_precision_type = MixedPrecisionType.FP32
            Cell_.set_mixed_precision_type(self, MixedPrecisionType.FP32)
        if "bf16" in flags and flags.get("bf16", False):
            self.mixed_precision_type = MixedPrecisionType.BF16
            Cell_.set_mixed_precision_type(self, MixedPrecisionType.BF16)

    def apply(self, fn):
        """
        Applies fn recursively to every subcell (as returned by .cells()) as well as self.
        Typical use includes initializing the parameters of a model.

        Args:
            fn (function): function to be applied to each subcell.

        Returns:
            Cell, self.

        Examples:
            >>> import mindspore.nn as nn
            >>> from mindspore.common.initializer import initializer, One
            >>> net = nn.SequentialCell(nn.Dense(2, 2), nn.Dense(2, 2))
            >>> def func(cell):
            ...     if isinstance(cell, nn.Dense):
            ...         cell.weight.set_data(initializer(One(), cell.weight.shape, cell.weight.dtype))
            >>> net.apply(func)
            SequentialCell(
              (0): Dense(input_channels=2, output_channels=2, has_bias=True)
              (1): Dense(input_channels=2, output_channels=2, has_bias=True)
            )
            >>> print(net[0].weight.asnumpy())
            [[1. 1.]
             [1. 1.]]
        """
        for cell in self.cells():
            cell.apply(fn)
        fn(self)
        return self

    def add_flags(self, **flags):
        """
        Add customized attributes for cell.

        This method is also called when the cell class is instantiated and the class parameter 'flags' is set to True.

        Args:
            flags (dict): Network configuration information, currently it is used for the binding of network and
                dataset. Users can also customize network attributes by this parameter.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...
            ...     def construct(self, x):
            ...         x = self.relu(x)
            ...         return x
            >>> net = Net()
            >>> net.add_flags(sink_mode=True)
            >>> print(net.sink_mode)
            True
        """
        if not hasattr(self, "_func_graph_flags"):
            self._func_graph_flags = {}
        self._func_graph_flags.update({**flags})
        self.__dict__.update({**flags})
        self._add_mixed_precision_flag(**flags)
        return self

    def add_flags_recursive(self, **flags):
        """
        If a cell contains child cells, this method can recursively customize attributes of all cells.

        Args:
            flags (dict): Network configuration information, currently it is used for the binding of network and
                dataset. Users can also customize network attributes by this parameter.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...
            ...     def construct(self, x):
            ...         x = self.relu(x)
            ...         return x
            >>> net = Net()
            >>> net.add_flags_recursive(sink_mode=True)
            >>> print(net.sink_mode)
            True
        """
        self.add_flags(**flags)
        for cell in self.cells():
            cell.add_flags_recursive(**flags)
        return self

    def _add_init_args(self, **args):
        if hasattr(self, '_cell_init_args'):
            self._cell_init_args += str({**args})

    def get_flags(self):
        """
        Get the self_defined attributes of the cell, which can be added by `add_flags` method.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...
            ...     def construct(self, x):
            ...         x = self.relu(x)
            ...         return x
            >>> net = Net()
            >>> net.add_flags(sink_mode=True)
            >>> print(net.get_flags())
            {'sink_mode':True}
        """
        if not hasattr(self, "_func_graph_flags"):
            self._func_graph_flags = {}
        return self._func_graph_flags

    def to_float(self, dst_type):
        """
        Add cast on all inputs of cell and child cells to run with certain float type.

        If `dst_type` is `mindspore.dtype.float16`, all the inputs of Cell, including input, Parameter and Tensor, will
        be cast to float16. Please refer to the usage in source code of :func:`mindspore.amp.build_train_network`.

        Note:
            Multiple calls will overwrite.

        Args:
            dst_type (:class:`mindspore.dtype`): Transfer cell to run with dst_type.
                dst_type can be `mstype.float16` , `mstype.float32` or `mstype.bfloat16`.

        Returns:
            Cell, the cell itself.

        Raises:
            ValueError: If dst_type is not `mstype.float32` , `mstype.float16` or `mstype.bfloat16`.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore.nn as nn
            >>> from mindspore import dtype as mstype
            >>>
            >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
            >>> net.to_float(mstype.float16)
            Conv2d(input_channels=120, output_channels=240, kernel_size=(4, 4), stride=(1, 1), pad_mode=same,
            padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=None, format=NCHW)
        """
        if dst_type not in (mstype.float16, mstype.float32, mstype.bfloat16):
            raise ValueError("For 'to_float', the argument 'dst_type' must be mstype.float32, mstype.float16 or "
                             "mstype.bfloat16, but got type: {} and value: {}.".format(type(dst_type), dst_type))
        flags = {'fp16': dst_type == mstype.float16, 'fp32': dst_type == mstype.float32,
                 'bf16': dst_type == mstype.bfloat16}
        self._add_init_args(**flags)
        self.add_flags_recursive(**flags)
        return self

    def set_boost(self, boost_type):
        """
        In order to improve the network performance, configure the network auto enable to
        accelerate the algorithm in the algorithm library.

        If `boost_type` is not in the algorithm library, please view the algorithm in the algorithm library through
        `algorithm library <https://gitee.com/mindspore/mindspore/tree/master/mindspore/python/mindspore/boost>`_.

        Note:
            Some acceleration algorithms may affect the accuracy of the network, please choose carefully.

        Args:
            boost_type (str): accelerate algorithm.

        Returns:
            Cell, the cell itself.

        Raises:
            ValueError: If boost_type is not in the algorithm library.
        """
        if boost_type not in ("less_bn",):
            raise ValueError("For 'set_boost', the argument 'boost_type' must be 'less_bn', "
                             "but got {}.".format(boost_type))
        flags = {"less_bn": boost_type == "less_bn"}
        self.add_flags_recursive(**flags)
        return self

    def set_grad(self, requires_grad=True):
        """
        Sets the cell flag for gradient.


        Args:
            requires_grad (bool): Specifies if the net need to grad, if it is
                ``true`` , the cell will construct backward network in pynative mode. Default: ``True`` .

        Returns:
            Cell, the cell itself.
        """
        self.requires_grad = requires_grad
        return self

    def set_train(self, mode=True):
        """
        Sets the cell to training mode.

        The cell itself and all children cells will be set to training mode. Layers that have different constructions
        for training and predicting, such as `BatchNorm`, will distinguish between the branches by this attribute. If
        set to true, the training branch will be executed, otherwise another branch.

        Note:
            When execute function Model.train(), framework will call Cell.set_train(True).
            When execute function Model.eval(), framework will call Cell.set_train(False).

        Args:
            mode (bool): Specifies whether the model is training. Default: ``True`` .

        Returns:
            Cell, the cell itself.

        Tutorial Examples:
            - `Model Training - Implementing Training and Evaluation
              <https://mindspore.cn/tutorials/en/master/beginner/train.html#training-and-evaluation>`_
        """
        if mode:
            self._phase = 'train'
        else:
            self._phase = 'predict'
        self.add_flags_recursive(training=mode)
        return self

    def set_broadcast_flag(self, mode=True):
        """
        Set parameter broadcast mode for this cell.

        Args:
            mode (bool): Specifies whether the mode is parameter broadcast. Default: ``True`` .
        """
        self.add_flags_recursive(broadcast_flag=mode)
        return self

    def set_jit_config(self, jit_config):
        """
        Set jit config for cell.

        Args:
            jit_config (JitConfig): Jit config for compile. For details, please refer to :class:`mindspore.JitConfig`.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...
            ...     def construct(self, x):
            ...         x = self.relu(x)
            ...         return x
            >>> net = Net()
            >>> jitconfig = ms.JitConfig()
            >>> net.set_jit_config(jitconfig)
        """
        if self._jit_config_dict:
            logger.warning("For Cell, jit config can only be set once, ignore this setting.")
        else:
            self._jit_config_dict = jit_config.jit_config_dict

    @jit_forbidden_register
    def register_forward_pre_hook(self, hook_fn, with_kwargs=False):
        """
        Register forward pre hook function for Cell object.

        The hook will be called before :func:`mindspore.nn.Cell.construct` is invoked.

        The hook function should be one of the following signatures:

        - `hook_fn(cell, args) -> None or new_args` , when `with_kwargs` is ``Flase`` .
        - `hook_fn(cell, args, kwargs) -> None or (new_args, new_kwargs)` , when `with_kwargs` is ``True`` .

        where:

        - `cell` (Cell): Cell object on which the hook is registered.
        - `args` (tuple): Positional arguments passed to the `construct` function.
        - `kwargs` (dict): Keyword arguments passed to the `construct` function. Only passed to `hook_fn` when
          `with_kwargs` is ``True`` .

        Note:
            - The `hook_fn` can modify the forward inputs by returning new inputs. If `with_kwargs` is ``Flase`` , a
              single value (whick will be wrapped into a tuple unless already a tuple) or a tuple of args should be
              returned. If `with_kwargs` is ``True`` , both `args` and `kwargs` should be returned.
            - In order to prevent running failed when switching to graph mode, it is not recommended to call it in the
              `construct` function of Cell object.
            - In the pynative mode, if this method is called inside the `construct` function of the Cell object, a
              `hook_fn` will be added at each run time of Cell object.

        Args:
            hook_fn (function): Python function. Forward pre hook function.
            with_kwargs (bool, optional): Specifies whether hook_fn will be passed the kwargs given to the `construct`
                function. Default: ``False`` .

        Returns:
            A handle corresponding to the `hook_fn` . The handle can be used to remove the added `hook_fn` by calling
            `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn, ops
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def forward_pre_hook_fn(cell, inputs):
            ...     print("forward inputs: ", inputs)
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.mul = nn.MatMul()
            ...         self.handle = self.mul.register_forward_pre_hook(forward_pre_hook_fn)
            ...
            ...     def construct(self, x, y):
            ...         x = x + x
            ...         x = self.mul(x, y)
            ...         return x
            >>> grad = ops.GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
            forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1],
                            dtype=Float32, value= [ 1.00000000e+00]))
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32,
            value= [ 2.00000000e+00]))
        """
        check_hook_fn(hook_fn)
        handle = HookHandle(self._forward_pre_hook, extra_dict=self._forward_pre_hook_with_kwargs)
        self._forward_pre_hook[handle.handle_id] = hook_fn
        if with_kwargs:
            self._forward_pre_hook_with_kwargs[handle.handle_id] = True
        _update_hook_version()
        return handle

    @jit_forbidden_register
    def _run_forward_pre_hook(self, args, kwargs):
        """
        Running forward pre hook function registered on Cell object.
        """
        for hook_id, hook_fn in self._forward_pre_hook.items():
            if hook_id in self._forward_pre_hook_with_kwargs:
                ret = hook_fn(self, args, kwargs)
                if ret is not None:
                    if isinstance(ret, tuple) and len(ret) == 2:
                        args, kwargs = ret
                    else:
                        raise RuntimeError(
                            "forward pre hook with kwargs must return None or a tuple of (new_args, new_kwargs), "
                            f"but got {ret}"
                        )
            else:
                ret = hook_fn(self, args)
                if ret is not None:
                    if not isinstance(ret, tuple):
                        ret = (ret,)
                    args = ret
        return args, kwargs

    def _jit_forward_pre_hook(self, inputs):
        """
        Compile forward pre hook function registered on Cell object.

        Args:
            inputs: The input objects of cell object.

        Returns:
            - **outputs** - New input objects or none.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
        """
        forward_pre_hook_inputs = inputs
        for fn in self._forward_pre_hook.values():
            ret = fn(self, forward_pre_hook_inputs)
            if ret is not None:
                if not isinstance(ret, tuple):
                    forward_pre_hook_inputs = (ret,)
                else:
                    forward_pre_hook_inputs = ret

        if len(forward_pre_hook_inputs) != len(inputs):
            raise TypeError(
                "The forward pre hook return value size is {} not equal to input size {}".format(
                    len(forward_pre_hook_inputs), len(inputs)))
        return forward_pre_hook_inputs

    @jit_forbidden_register
    def register_forward_hook(self, hook_fn, with_kwargs=False):
        """
        Register forward hook function for Cell object.

        This hook will be called after :func:`mindspore.nn.Cell.construct` has computed an output.

        The hook function should be one of the following signatures:

        - `hook_fn(cell, args, output) -> None or new_output` , when `with_kwargs` is ``False`` .
        - `hook_fn(cell, args, kwargs, output) -> None or new_output` , when `with_kwargs` is ``True`` .

        where:

        - `cell` (Cell): Cell object on which the hook is registered.
        - `args` (tuple): Positional arguments passed to the `construct` function.
        - `kwargs` (dict): Keyword arguments passed to the `construct` function. Only passed to `hook_fn` when
          `with_kwargs` is ``True`` .
        - `output`: Output generated by the `construct` function.

        Note:
            - The `hook_fn` can modify the forward outputs by returning new outputs.
            - In order to prevent running failed when switching to graph mode, it is not recommended to call it in the
              `construct` function of Cell object.
            - In the pynative mode, if this method is called inside the `construct` function of the Cell object, a
              `hook_fn` will be added at each run time of Cell object.

        Args:
            hook_fn (function): Python function. Forward hook function.
            with_kwargs (bool, optional): Specifies whether hook_fn will be passed the kwargs given to the `construct`
                function. Default: ``False`` .

        Returns:
            A handle corresponding to the `hook_fn` . The handle can be used to remove the added `hook_fn` by calling
            `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn, ops
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def forward_hook_fn(cell, inputs, output):
            ...     print("forward inputs: ", inputs)
            ...     print("forward output: ", output)
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.mul = nn.MatMul()
            ...         self.handle = self.mul.register_forward_hook(forward_hook_fn)
            ...
            ...     def construct(self, x, y):
            ...         x = x + x
            ...         x = self.mul(x, y)
            ...         return x
            >>> grad = ops.GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
            forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1],
                            dtype=Float32, value= [ 1.00000000e+00]))
            forward output: 2.0
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32,
            value= [ 2.00000000e+00]))
        """
        if self.has_bprop:
            return HookHandle()
        check_hook_fn(hook_fn)
        handle = HookHandle(self._forward_hook, extra_dict=self._forward_hook_with_kwargs)
        self._forward_hook[handle.handle_id] = hook_fn
        if with_kwargs:
            self._forward_hook_with_kwargs[handle.handle_id] = True
        _update_hook_version()
        return handle

    def _jit_forward_hook(self, inputs, output):
        """
        Compile forward hook function registered on Cell object.

        Args:
            inputs: The input objects of Cell object.
            output: The output object of Cell object.

        Returns:
            - **output** - New output object or none.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
        """
        forward_hook_output = output
        for fn in self._forward_hook.values():
            ret = fn(self, inputs, forward_hook_output)
            if ret is not None:
                forward_hook_output = ret

        if isinstance(output, tuple):
            if not isinstance(forward_hook_output, tuple):
                forward_hook_output = (forward_hook_output,)
            if len(forward_hook_output) != len(output):
                raise TypeError(
                    "The forward hook return value size is {} not equal to output size {}".format(
                        len(forward_hook_output), len(output)))
        return forward_hook_output

    @jit_forbidden_register
    def _run_forward_hook(self, args, kwargs, output):
        """
        Running forward hook function registered on Cell object.
        """
        for hook_id, hook_fn in self._forward_hook.items():
            if hook_id in self._forward_hook_with_kwargs:
                ret = hook_fn(self, args, kwargs, output)
            else:
                ret = hook_fn(self, args, output)
            if ret is not None:
                output = ret
        return output

    @jit_forbidden_register
    def register_backward_pre_hook(self, hook_fn):
        """
        Register the backward pre hook function.

        Note:
            - The 'hook_fn' must be defined as the following code.
              `cell` is the Cell object. `grad_output` is the gradient passed to the Cell.
            - The 'hook_fn' should have the following signature:
              hook_fn(cell, grad_output) -> New grad_output gradient or None.
            - The 'hook_fn' is executed in the python environment. In order to prevent running failed when switching to
              graph mode, it is not recommended to write it in the `construct` function of Cell object.
            - In the pynative
              mode, if the `register_backward_pre_hook` function is called in the `construct` function of the Cell
              object, a hook function will be added at each run time of Cell object.

        Args:
            hook_fn (function): Python function. Backward pre hook function.

        Returns:
            A handle corresponding to the `hook_fn` . The handle can be used to remove the added `hook_fn` by calling
            `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn, ops
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def backward_pre_hook_fn(cell, grad_output):
            ...     print("backward input: ", grad_output)
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...         self.handle = self.relu.register_backward_pre_hook(backward_pre_hook_fn)
            ...
            ...     def construct(self, x):
            ...         x = x + x
            ...         x = self.relu(x)
            ...         return x
            >>> grad = ops.GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)))
            backward input: (Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]),)
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
        """
        check_hook_fn(hook_fn)
        handle = HookHandle(self._backward_pre_hook, extra_dict=None)
        self._backward_pre_hook[handle.handle_id] = hook_fn
        if self._cell_backward_pre_hook is None:  # pylint: disable=E0203
            # Generate a CellBackwardHook prim, and add function for it
            self._cell_backward_pre_hook = inner.CellBackwardHook(self.cls_name + "(" + str(id(self)) + ")",
                                                                  self, self._backward_pre_hook)
            self._cell_backward_pre_hook.register_backward_pre_hook()
        _update_hook_version()
        return handle

    def get_extra_state(self) -> Any:
        """Return any extra state to include in the cell's state_dict.

        This function is called from ``state_dict``.
        Implement this and a corresponding ``set_extra_state`` for your cell
        if you need to store extra state.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. Only provide backwards compatibility guarantees
        for serializing tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object, any extra state to store in the cell's state_dict.
        """
        raise RuntimeError(
            "Reached a code path in Cell.get_extra_state() that should never be called."

        )

    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from `load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        `get_extra_state` for your cell if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`.
        """
        raise RuntimeError(
            "Reached a code path in Cell.set_extra_state() that should never be called."
        )

    @jit_forbidden_register
    def register_state_dict_post_hook(self, hook):
        r"""Register a post-hook for the :func:`mindspore.nn.Cell.state_dict` method.

        It should have the following signature:

        hook(cell, state_dict, prefix, local_metadata) -> None

        The registered hooks can modify the ``state_dict`` inplace.

        Args:
            hook (Callable): The hook function after `state_dict` is called.

        Returns:
            A handle that can be used to remove the added hook by calling
            `handle.remove()`.
        """
        handle = HookHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.handle_id] = hook
        return handle

    @jit_forbidden_register
    def register_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook for the :func:`mindspore.nn.Cell.state_dict` method.

        It should have the following signature:

        hook(cell, prefix, keep_vars) -> None

        The registered hooks can be used to perform pre-processing before the `state_dict`
        call is made.

        Args:
            hook (Callable): The hook function before `state_dict` is called.

        Returns:
            A handle that can be used to remove the added hook by calling
            `handle.remove()`.

        Examples:
            >>> import mindspore
            ...
            ...
            >>> class NetA(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_a", mindspore.tensor([1, 2, 3]))
            ...         self.param_a = mindspore.Parameter(mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return x + self.buffer_a + self.param_a
            ...
            ...
            >>> def _add_extra_param(cell, prefix, keep_vars):
            ...     cell._params["extra_param"] = mindspore.Parameter(mindspore.tensor([4, 5, 6]))
            ...
            ...
            >>> net = NetA()
            >>> handle = net.register_state_dict_pre_hook(_add_extra_param)
            >>> net_state_dict = net.state_dict()
            >>> handle.remove()
            >>> print("extra_param" in net_state_dict)
            True
        """
        handle = HookHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.handle_id] = hook
        return handle

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save cell state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the cell, but not its descendants. This is called on every
        sub cell in :func:`mindspore.nn.Cell.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                cell
        """
        for name, param in self._params.items():
            if param is not None:
                destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if (
                getattr(self.__class__, "get_extra_state", Cell.get_extra_state)
                is not Cell.get_extra_state
        ):
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar("T_destination", bound=Dict[str, Any])

    @jit_forbidden_register
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the cell.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the cell's parameters and buffers.

        .. warning::
            - Currently ``state_dict()`` also accepts positional arguments for
              ``destination``, ``prefix`` and ``keep_vars`` in order. However,
              this is being deprecated and keyword arguments will be enforced in
              future releases.

            - Please avoid the use of argument ``destination`` as it is not
              designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of cell will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): A prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): Whether the state_dict returns a copy. Default: ``False`` , returns a reference.

        Returns:
            Dict, a dictionary containing a whole state of the cell.

        Examples:
            >>> import mindspore
            >>> class Model(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_a", mindspore.tensor([4, 5, 6]))
            ...         self.param_a = mindspore.Parameter(mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return x + self.buffer_a + self.param_a
            ...
            ...
            >>> model = Model()
            >>> print(model.state_dict())
            OrderedDict([('param_a', Parameter (name=param_a, shape=(3,), dtype=Int64, requires_grad=True)), \
            ('buffer_a', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]))])
        """
        if args:
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html"
                " for details.",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
        if destination is not None and not isinstance(destination, dict):
            raise TypeError(f"The type of destination must be OrderedDict, but got {type(destination)}")
        if not isinstance(prefix, str):
            raise TypeError(f"The type of prefix must be string, but got {type(prefix)}")
        if not isinstance(keep_vars, bool):
            raise TypeError(f"The type of keep_vars must be bool, but got {type(keep_vars)}")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = {}
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, cell in self._cells.items():
            if cell is not None:
                cell.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                raise RuntimeError("state_dict post-hook must return None")
        return destination

    @jit_forbidden_register
    def register_load_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook to be run before cell's :func:`mindspore.nn.Cell.load_state_dict` is called.

        It should have the following signature:

        hook(cell, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None

        Args:
            hook (Callable): The hook function before `load_state_dict` is called.

        Returns:
            A handle that can be used to remove the added hook by calling
            `handle.remove()`.
        """
        handle = HookHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.handle_id] = hook
        return handle

    @jit_forbidden_register
    def register_load_state_dict_post_hook(self, hook):
        r"""Register a post-hook to be run after cell's :func:`mindspore.nn.Cell.load_state_dict` is called.

        It should have the following signature:

        hook(cell, incompatible_keys) -> None

        The ``cell`` argument is the current cell that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Args:
            hook (Callable): The hook function after `load_state_dict` is called.

        Returns:
            A handle that can be used to remove the added hook by calling
            `handle.remove()`.
        """
        handle = HookHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.handle_id] = hook
        return handle

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into only this cell, but not its descendants.

        This is called on every sub cell
        in :func:`mindspore.nn.Cell.load_state_dict`. Metadata saved for this
        cell in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :func:`mindspore.nn.Cell.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                cell
            local_metadata (dict): a dict containing the metadata for this cell.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this cell
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :func:`mindspore.nn.Cell.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._params.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not isinstance(input_param, Tensor):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        "expected Tensor or Tensor-like object from checkpoint but "
                        f"received {type(input_param)}"
                    )
                    continue

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(
                        f"size mismatch for {key}: copying a param with shape {input_param.shape} from checkpoint, "
                        f"the shape in current model is {param.shape}."
                    )
                    continue
                try:
                    param.assign_value(Tensor(input_param.asnumpy(), dtype=param.dtype))
                except Exception as ex:  # pylint: disable=W0703
                    error_msgs.append(
                        f'While copy the parameter named "{key}", '
                        f"whose shape in the model are {param.shape} and "
                        f"whose shape in the checkpoint are {input_param.shape}, "
                        f"an exception occurred : {ex.args}."
                    )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", Cell.set_extra_state) is not Cell.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):].split(".", 1)
                    # Must be cell if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._cells:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    @jit_forbidden_register
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Copy parameters and buffers from :attr:`state_dict` into this cell and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this cell's :func:`mindspore.nn.Cell.state_dict` function.

        Args:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
            strict (bool, optional): Whether to strictly enforce that the keys
                in input `state_dict` match the keys returned by this cell's
                :func:`mindspore.nn.Cell.state_dict` function. Default ``True`` .

        Returns:
            A namedtuple with ``missing_keys`` and ``unexpected_keys`` fields,

            - `missing_keys` is a list of str containing any keys that are expected
              by this cell but missing from the provided ``state_dict``.

            - `unexpected_keys` is a list of str containing the keys that are not
              expected by this cell but present in the provided ``state_dict``.

        Note:
            If `strict` is ``True`` and a parameter or buffer is registered as ``None``, but its corresponding key
            exists in :attr:`state_dict`, and :func:`mindspore.nn.Cell.load_state_dict` will raise a ``RuntimeError``.

        Examples:
            >>> import mindspore
            >>> import os
            >>> class Model(mindspore.nn.Cell):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.register_buffer("buffer_a", mindspore.tensor([4, 5, 6]))
            ...         self.param_a = mindspore.Parameter(mindspore.tensor([1, 2, 3]))
            ...
            ...     def construct(self, x):
            ...         return x + self.buffer_a + self.param_a
            ...
            ...
            >>> model = Model()
            >>> print(model.state_dict())
            >>> mindspore.save_checkpoint(model.state_dict(), './model_state_dict_ckpt')
            >>> new_model = Model()
            >>> new_model.load_state_dict(mindspore.load_checkpoint('./model_state_dict_ckpt'))
            >>> print(new_model.state_dict())
            >>> os.remove('./model_state_dict_ckpt')
            OrderedDict([('param_a', Parameter (name=param_a, shape=(3,), dtype=Int64, requires_grad=True)), \
            ('buffer_a', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]))])
            OrderedDict([('param_a', Parameter (name=param_a, shape=(3,), dtype=Int64, requires_grad=True)), \
            ('buffer_a', Tensor(shape=[3], dtype=Int64, value= [4, 5, 6]))])
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(cell, local_state_dict, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            cell._load_from_state_dict(
                local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
            )
            for name, child in cell._cells.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)  # noqa: F821

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in cell._load_state_dict_post_hooks.values():
                out = hook(cell, incompatible_keys)
                if out is not None:
                    raise RuntimeError(
                        "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                        "expected to return new values, if incompatible_keys need to be modified,"
                        "it should be done inplace."
                    )

        load(self, state_dict)
        del load

        if strict:
            if unexpected_keys:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in unexpected_keys)
                    ),
                )
            if missing_keys:
                error_msgs.insert(
                    0,
                    "Missing key(s) in state_dict: {}. ".format(
                        ", ".join(f'"{k}"' for k in missing_keys)
                    ),
                )

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @jit_forbidden_register
    def register_backward_hook(self, hook_fn):
        """
        Register the backward hook function.

        Note:
            - The 'hook_fn' must be defined as the following code.
              `cell` is the registered Cell object. `grad_input` is the gradient computed and passed to
              the next Cell or primitive, which can be return a new gradient or None. `grad_output` is the gradient
              passed to the Cell.
            - The 'hook_fn' should have the following signature:
              hook_fn(cell, grad_input, grad_output) -> New grad_input gradient or none.
            - The 'hook_fn' is executed in the python environment. In order to prevent running failed when switching to
              graph mode, it is not recommended to write it in the `construct` function of Cell object. In the pynative
              mode, if the `register_backward_hook` function is called in the `construct` function of the Cell object,
              a hook function will be added at each run time of Cell object.

        Args:
            hook_fn (function): Python function. Backward hook function.

        Returns:
            A handle corresponding to the `hook_fn` . The handle can be used to remove the added `hook_fn` by calling
            `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor, nn, ops
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def backward_hook_fn(cell, grad_input, grad_output):
            ...     print("backward input: ", grad_output)
            ...     print("backward output: ", grad_input)
            ...
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.relu = nn.ReLU()
            ...         self.handle = self.relu.register_backward_hook(backward_hook_fn)
            ...
            ...     def construct(self, x):
            ...         x = x + x
            ...         x = self.relu(x)
            ...         return x
            >>> grad = ops.GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)))
            backward input: (Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]),)
            backward output: (Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]),)
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
        """
        check_hook_fn(hook_fn)
        handle = HookHandle(self._backward_hook, extra_dict=None)
        self._backward_hook[handle.handle_id] = hook_fn
        if self._cell_backward_hook is None:  # pylint: disable=E0203
            # Generate a CellBackwardHook prim, and add function for it
            self._cell_backward_hook = inner.CellBackwardHook(self.cls_name + "(" + str(id(self)) + ")",
                                                              self, self._backward_hook)
            self._cell_backward_hook.register_backward_hook()
        _update_hook_version()
        return handle

    @jit_forbidden_register
    def register_saved_tensors_hooks(self, pack_hook, unpack_hook):
        """
        Register hook functions for packing and unpacking saved tensors.

        The effective scope of this method is limited to the :func:`mindspore.nn.Cell.construct` function.
        For more details, please refer to :class:`mindspore.saved_tensors_hooks` .

        Args:
            pack_hook (Callable): A function that defines how to process a tensor
                                  before it is saved during the forward pass.
            unpack_hook (Callable): A function that defines how to recover the tensor
                                    when it is needed during the backward computation.
        """
        self._saved_tensors_pack_hook = pack_hook
        self._saved_tensors_unpack_hook = unpack_hook

    def set_comm_fusion(self, fusion_type, recurse=True):
        """
        Set `comm_fusion` for all the parameters in this cell. Please refer to the description of
        :class:`mindspore.Parameter.comm_fusion`.

        Note:
            The value of attribute will be overwritten when the function is called multiply.

        Args:
            fusion_type (int): The value of `comm_fusion`.
            recurse (bool): Whether sets the trainable parameters of subcells. Default: ``True`` .
        """
        Validator.check_non_negative_int(fusion_type)
        for param in self.trainable_params(recurse):
            param.comm_fusion = fusion_type
        return self

    def _set_recompute_scope(self, mode):
        prefix = 'recompute_'
        if mode:
            if self._scope is None:
                self._scope = prefix
            elif not self._scope.startswith(prefix):
                self._scope = prefix + self._scope
        elif self._scope is not None and self._scope.startswith(prefix):
            self._scope = self._scope[len(prefix):]

    def _mp_comm_recompute(self, mp_comm_recompute=True):
        """
        Set the model parallel communication in cell recomputed.
        """
        for _, value in self._primitives.items():
            if value:
                value.add_prim_attr("recompute_comm_op", mp_comm_recompute)
        for cell in self.cells():
            cell._mp_comm_recompute(mp_comm_recompute)

    def _parallel_optimizer_comm_recompute(self, parallel_optimizer_comm_recompute=False):
        """
        Set the parallel optimizer communication in cell recomputed.
        """
        for param in self.trainable_params():
            param.parallel_optimizer_comm_recompute = parallel_optimizer_comm_recompute

    def _recompute_slice_activation(self, slice_activation=False):
        """
        Slice the cell output which would remains in memory.
        """
        for _, value in self._primitives.items():
            if value:
                value.add_prim_attr("slice_activation", slice_activation)
        for cell in self.cells():
            cell._recompute_slice_activation(slice_activation)

    def _recompute(self, mode=True, output_recompute=False):
        """
        Set the cell recomputed.
        """
        Validator.check_bool(mode)
        Validator.check_bool(output_recompute)
        if not self._has_config_recompute:  # pylint: disable=E0203
            self._has_config_recompute = True
        else:
            logger.info("The recompute interface can be configured only once."
                        " When the parent cell is configured, the child cell should not be configured")
            return
        self._set_recompute_scope(mode)
        if mode and not output_recompute:
            self.add_flags(output_no_recompute=True)
        for cell in self.cells():
            cell._recompute(mode, True)

    @args_type_check(mp_comm_recompute=bool, parallel_optimizer_comm_recompute=bool)
    def recompute(self, *, use_reentrant=True, output_recompute=False, **kwargs):
        r"""
        Set the cell recomputed. All the primitive in the cell except the outputs will be set recomputed.
        If a primitive set recomputed feeds into some backward nodes for computing gradient, rather than
        storing the intermediate activation computed in forward pass, we will recompute it in backward pass.

        Note:

            - If the computation involves something like randomization or global variable, the equivalence
              is not guaranteed currently.
            - If the recompute api of a primitive in this cell is also called, the recompute mode of this
              primitive is subject to the recompute api of the primitive.
            - The interface can be configured only once.
              Therefore, when the parent cell is configured, the child cell should not be configured.
            - The outputs of cell are excluded from recomputation by default, which is based on our configuration
              experience to reduce memory footprint. If a cell has only one primitive and the primitive is wanted
              to be set recomputed, use the recompute api of the primtive.
            - When the memory remains after applying the recomputation, configuring 'mp_comm_recompute=False'
              to improve performance if necessary.
            - When the memory still not enough after applying the recompute, configuring
              'parallel_optimizer_comm_recompute=True' to save more memory if necessary.
              Cells in the same fusion group should have the same parallel_optimizer_comm_recompute configures.

        Keyword Arguments:
            use_reentrant(bool, optional): This keyword is only valid in PyNative mode.
                If use_reentrant=True is set, we will implement recomputation through a custom bprop function,
                which does not support differentiation of complex types such as List/Tuple; if use_reentrant=False is
                set, we will use the saved_tensors_hook functionality to implement recomputation, which supports
                differentiation of tensors inside complex types. Default: ``True`` .
            output_recompute(bool, optional): This keyword is only valid in PyNative mode. If output_recompute=True is
                set, we will implement recomputation by saved_tensors_hook functionality default. The output of this
                cell will not be stored by subsequent operations for backward. when there are two adjacent cells both
                requiring recomputation (where the output of one cell serves as the input to the other), the
                recomputation of these two operators will be merged. In this case, the output activation values of the
                first cell will not be saved. If output_recompute=False, we will not merge adjacent cells.
                Default: ``False`` .
            \*\*kwargs: Other arguments

                - mp_comm_recompute (bool, optional): Specifies whether the model parallel communication operators
                  in the cell are recomputed in auto parallel or semi auto parallel mode. Default: ``True`` .
                - parallel_optimizer_comm_recompute (bool, optional): Specifies whether the communication operator
                  allgathers introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                  Default: ``False`` .
        """
        if output_recompute:
            use_reentrant = False
        self._recompute_cell = recompute_registry.get()(self.construct, use_reentrant, output_recompute)
        self._recompute()
        if 'mp_comm_recompute' in kwargs:
            self._mp_comm_recompute(kwargs.get('mp_comm_recompute', False))
        if 'parallel_optimizer_comm_recompute' in kwargs:
            if kwargs.get('parallel_optimizer_comm_recompute', False):
                logger.warning("Currently, the communication operator allgathers introduced by optimizer shard "
                               "is replaced with zero3.")
        if 'recompute_slice_activation' in kwargs:
            self._recompute_slice_activation(kwargs.get('recompute_slice_activation', False))

        for key in kwargs:
            if key not in ('mp_comm_recompute', 'parallel_optimizer_comm_recompute', 'recompute_slice_activation'):
                raise ValueError("For 'recompute', keyword '%s' is not recognized! "
                                 "the key kwargs must be 'mp_comm_recompute', "
                                 "'parallel_optimizer_comm_recompute', 'recompute_slice_activation'" % key)

    def _get_attr_from_cell(self, network):
        if not isinstance(network, Cell):
            return
        if hasattr(network, "jit_config_dict"):
            self._jit_config_dict = network.jit_config_dict
        if hasattr(network, "_amp_level"):
            self._amp_level = getattr(network, "_amp_level")

    def _set_jit_graph_name(self, key):
        """
        Set jit graph name.
        """
        self._jit_graph_name = key

    def _jit_backward_pre_hook(self, grad_output):
        new_grad_output = grad_output
        if not isinstance(grad_output, tuple):
            new_grad_output = (grad_output,)

        for fn in self._backward_pre_hook.values():
            ret = fn(self, new_grad_output)
            if ret is not None:
                if not isinstance(ret, tuple):
                    output = (ret,)
                else:
                    output = ret
            else:
                output = ops.Depend()(new_grad_output, ret)
            new_grad_output = output

        if not isinstance(grad_output, tuple):
            if len(new_grad_output) == 1:
                return new_grad_output[0]
            raise TypeError(
                "The backward pre hook return value size is {} not equal to input size 1".format(
                    len(new_grad_output)))

        if len(new_grad_output) != len(grad_output):
            raise TypeError(
                "The backward pre hook return value size is {} not equal to input size {}".format(
                    len(new_grad_output), len(grad_output)))

        return new_grad_output

    def _jit_backward_hook(self, grad_input, grad_output):
        backward_hook_input = grad_input
        backward_hook_output = grad_output
        if not isinstance(grad_input, tuple):
            backward_hook_input = (grad_input,)
        if not isinstance(grad_output, tuple):
            backward_hook_output = (grad_output,)

        for fn in self._backward_hook.values():
            ret = fn(self, backward_hook_input, backward_hook_output)
            if ret is not None:
                if not isinstance(ret, tuple):
                    output = (ret,)
                else:
                    output = ret
            else:
                output = ops.Depend()(backward_hook_input, ret)

            backward_hook_input = output

        if not isinstance(grad_input, tuple):
            return backward_hook_input[0]

        if len(backward_hook_input) != len(grad_input):
            raise TypeError(
                "The backward hook return value size is {} not equal to input size {}".format(
                    len(backward_hook_input), len(grad_input)))
        return backward_hook_input


class GraphCell(Cell):
    """
    Base class for running the graph loaded from a MindIR.

    This feature is still under development. Currently `GraphCell` do not support modifying the structure of the
    diagram, and can only use data that shape and type are the same as the input when exporting the MindIR.

    Args:
        graph (FuncGraph): A compiled graph loaded from MindIR.
        params_init (dict): Parameters need to be inited in the graph.
            The key is the parameter name whose type is str, and the value is a Tensor or Parameter.
            If the parameter exists in the graph according to the name, update it's value.
            If the parameter does not exist, ignore it. Default: ``None`` .
        obf_random_seed (Union[int, None]): The random seed used for dynamic obfuscation, which is not supported now.

    Raises:
        NotImplementedError: Dynamic structure obfuscation is not supported now.
        TypeError: If the `graph` is not a FuncGraph.
        TypeError: If the `params_init` is not a dict.
        TypeError: If the key of the `params_init` is not a str.
        TypeError: If the value of the `params_init` is neither a Tensor nor a Parameter.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor
        >>> from mindspore import context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> net = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")
        >>> input = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> ms.export(net, input, file_name="net", file_format="MINDIR")
        >>> graph = ms.load("net.mindir")
        >>> net = nn.GraphCell(graph)
        >>> output = net(input)
        >>> print(output)
        [[[[4. 6. 4.]
           [6. 9. 6.]
           [4. 6. 4.]]]]
    """

    def __init__(self, graph, params_init=None, obf_random_seed=None):
        super().__init__(auto_prefix=True)
        if obf_random_seed is not None:
            raise NotImplementedError("Dynamic structure obfuscation is not supported now.")
        if not isinstance(graph, FuncGraph):
            raise TypeError(f"For 'GraphCell', the argument 'graph' must be a FuncGraph loaded from MindIR, "
                            f"but got type {type(graph)}.")
        self.graph = graph
        params_init = {} if params_init is None else params_init
        if not isinstance(params_init, dict):
            raise TypeError(f"For 'GraphCell', the argument 'params_init' must be a dict, but got {type(params_init)}.")
        for name, value in params_init.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise TypeError("For 'GraphCell', the key of the 'params_init' must be str, "
                                "and the value must be Tensor or Parameter, "
                                f"but got the key type: {type(name)}, and the value type: {type(value)}")

        params_dict = update_func_graph_hyper_params(self.graph, params_init)
        for name, param in params_dict.items():
            self._params[name] = param

    def construct(self, *inputs):
        return self.graph(*inputs)

    def __call__(self, *args, **kwargs):
        self.phase = "graph_load_from_mindir"
        self._add_attr("graph_load_from_mindir", self.graph)
        return self.compile_and_run(*args, **kwargs)


def _is_parameter_list_or_tuple(value):
    """
    Check the type of input in list or tuple is Parameter.
    :param value: list or tuple.
    :return: The types of all inputs are parameter.
    """
    if isinstance(value, (list, tuple)) and value:
        for item in value:
            if not isinstance(item, Parameter):
                return False
        return True
    return False


def _addindent(s_, num_spaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
