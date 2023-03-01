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
"""cell"""
from __future__ import absolute_import

import gc
import inspect
import os
import time
from collections import OrderedDict
from types import FunctionType, MethodType
import numpy

import mindspore.dataset as ds
from mindspore._checkparam import args_type_check
from mindspore import log as logger
from mindspore.common.parameter import PARAMETER_NAME_DEFAULT
from mindspore.common.hook_handle import HookHandle
from mindspore.context import ParallelMode
from mindspore.ops.composite import Shard
from mindspore import context
from mindspore._c_expression import init_pipeline, update_func_graph_hyper_params, Cell_, FuncGraph, MixedPrecisionType
from mindspore._checkparam import Validator
from mindspore.common import dtype as mstype
from mindspore.common.api import _cell_graph_executor, _pynative_executor, _get_args_for_run, cells_compile_cache
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore.ops.operations import Cast
from mindspore.ops.primitive import Primitive
from mindspore.ops.operations import _inner_ops as inner
from mindspore.parallel._tensor import _load_tensor_by_layout


def _check_args(args):
    """Check the input args's type"""
    index = 1
    for item in args:
        if isinstance(item, Tensor) and item.has_init:
            item.init_data()
        elif isinstance(item, numpy.ndarray):
            suffix = "th"
            if index == 1:
                suffix = "st"
            elif index == 2:
                suffix = "nd"
            elif index == 3:
                suffix = "rd"

            input_index = str(index) + suffix
            raise TypeError(f"For 'Cell', inputs should not be numpy array. Only support bool, int, float, None, "
                            f"Tensor, Parameter, mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint"
                            f"), and tuple or list containing only these types, and dict whose values are these "
                            f"types, but the {input_index} arg type is {type(item)}.")
        index += 1


class Cell(Cell_):
    """
    The basic building block of neural networks in MindSpore. The model or neural network layer should inherit this
    base class.

    Layers in `mindspore.nn` are also the subclass of Cell, such as :class:`mindspore.nn.Conv2d`,
    and :class:`mindspore.nn.ReLU`, etc. Cell will be compiled into a calculation
    graph in GRAPH_MODE (static graph mode) and used as the basic module of neural networks in
    PYNATIVE_MODE (dynamic graph mode).

    Args:
        auto_prefix (bool): Whether to automatically generate NameSpace for Cell and its subcells. It will affect the
                      name of the parameter in the network. If set to True, the network parameter
                      name will be prefixed, otherwise it will not. Default: True.
        flags (dict): Network configuration information, currently it is used for the binding of network and dataset.
                      Users can also customize network attributes by this parameter. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
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

    IGNORE_LIST = ['_scope', '_cell_init_args', '_auto_prefix', '_cells', '_params', '_construct_inputs_names',
                   '_construct_inputs_num', '_create_time', '_func_graph_flags', '_parallel_inputs_run',
                   '_parameter_layout_dict', '_params_list', '_tensor_list', '_phase', '_auto_parallel_mode',
                   '_forward_pre_hook', '_forward_hook', '_enable_forward_pre_hook', '_enable_forward_hook',
                   '_bprop_debug', '_enable_backward_hook', '_cell_backward_hook', '_is_run', '_param_prefix',
                   '_attr_synced', 'pynative', 'requires_grad', '_auto_parallel_compile_and_run', 'cell_type']

    def __init__(self, auto_prefix=True, flags=None):
        Cell_.__init__(self, self._cell_tag)
        self._params = OrderedDict()
        self._cells = OrderedDict()
        self._params_list = OrderedDict()
        self._tensor_list = OrderedDict()
        self._primitives = OrderedDict()
        self.training = False
        self.requires_grad = False
        self.pynative = False
        self._attr_synced = False
        self._param_prefix = ''
        self._auto_prefix = auto_prefix
        self._scope = None
        self._phase = 'train'
        self._parameter_layout_dict = {}
        self._parallel_parameter_name_list = ()
        self._parallel_parameter_merge_net_dict = {}
        self._create_time = int(time.time() * 1e9)
        self.arguments_key = ""
        self.compile_cache = set()
        cells_compile_cache[id(self)] = self.compile_cache
        self.parameter_broadcast_done = False
        self._id = 1
        self.exist_names = set("")
        self.exist_objs = set()
        init_pipeline()

        # call gc to release GE session resources used by non-used cell objects
        if os.getenv('GC_COLLECT_IN_CELL') == '1':
            gc.collect()

        self._construct_inputs_num = 0
        self._construct_inputs_names = []
        self._auto_parallel_mode = False
        self._parallel_inputs_run = None
        if flags:
            self.add_flags(**flags)
        self._bprop_debug = False
        self._forward_pre_hook = OrderedDict()
        self._forward_hook = OrderedDict()
        self._enable_forward_pre_hook = False
        self._enable_forward_hook = False
        self._enable_backward_hook = False
        self._cell_backward_hook = None
        self.cell_type = None
        self._auto_parallel_compile_and_run = False
        self.cast = Cast()
        self._has_config_recompute = False
        self._user_parameters = []
        self._dynamic_shape_inputs = None
        self.saved_dynamic_shape = None
        self._jit_config_dict = dict()
        self.grad_ops_label = False

    def __getstate__(self):
        base = Cell_.__getstate__(self)
        return base, self.__dict__

    def __setstate__(self, state):
        base, dict_ = state
        Cell_.__setstate__(self, base)
        self.__dict__ = dict_
        self._attr_synced = False

    @property
    def _cell_tag(self):
        # `<class 'xxxxxxx'>` to `xxxxxxx`
        return str(self.__class__)[8:-2]

    @property
    def create_time(self):
        return self._create_time

    @property
    def cell_init_args(self):
        return self._cell_init_args

    @property
    def param_prefix(self):
        """
        Param prefix is the prefix of current cell's direct child parameter.
        """
        return self._param_prefix

    @property
    def bprop_debug(self):
        """
        Get whether cell custom bprop debug is enabled.
        """
        return self._bprop_debug

    @bprop_debug.setter
    def bprop_debug(self, value):
        """
        Set whether to enable cell custom bprop debug.

        Note:
            When bprop is defined in cell, the bprop function will be executed
            in python interpreter when bprop debug is true, and will be parsed
            and add to graph when bprop debug is false.

        Args:
            value (bool): Specifies whether to enable bprop debug. Default: False.
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
    def parameter_layout_dict(self):
        """
        `parameter_layout_dict` represents the tensor layout of a parameter, which is inferred by shard strategy and
        distributed operator information.
        """
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
        return self._parallel_parameter_name_list

    @parallel_parameter_name_list.setter
    def parallel_parameter_name_list(self, value):
        if not isinstance(value, list):
            raise TypeError(f"For 'Cell', the property 'parallel_parameter_name_list' must be list type, "
                            f"but got type {type(value)}.")
        self._parallel_parameter_name_list = value

    @property
    def pipeline_stage(self):
        return self._pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("For 'Cell', the property 'pipeline_stage' "
                            "must be int type, but got type : {}".format(type(value)))

        if value < 0:
            raise ValueError("For 'Cell', the property 'pipeline_stage' "
                             "can not be less than 0, but got {}".format(value))
        self._pipeline_stage = value
        for item in self.trainable_params():
            item.add_pipeline_stage(value)

    @property
    def parallel_parameter_merge_net_dict(self):
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

    def get_func_graph_proto(self):
        """Return graph binary proto."""
        exec_id = ".".join([self.phase, str(self.create_time), str(id(self))])
        return _cell_graph_executor._get_func_graph_proto(self, exec_id, "anf_ir", True)

    def __getattr__(self, name):
        if '_params' in self.__dict__:
            params = self.__dict__['_params']
            if name in params:
                if context._get_mode() == context.PYNATIVE_MODE:
                    return self.cast_param(params[name])
                return params[name]
        if '_cells' in self.__dict__:
            cells = self.__dict__['_cells']
            if name in cells:
                return cells[name]
        if '_tensor_list' in self.__dict__:
            tensor_list = self.__dict__['_tensor_list']
            if name in tensor_list:
                return self.cast_param(tensor_list[name])
        if '_params_list' in self.__dict__:
            params_list = self.__dict__['_params_list']
            if name in params_list:
                para_list = params_list[name]
                cast_list = list()
                for para in para_list:
                    cast_list.append(self.cast_param(para))
                para_list = ParameterTuple(cast_list)
                return para_list
        raise AttributeError("The '{}' object has no attribute '{}'.".format(type(self).__name__, name))

    def __del__(self):
        if context.get_context is not None and context._get_mode() == context.PYNATIVE_MODE:
            _pynative_executor.del_cell(self)

        # while deepcopy a cell instance, the copied cell instance can't be added to cells_compile_cache
        # here using pop(id(self), None) to avoid KeyError exception
        cells_compile_cache.pop(id(self), None)
        if self.compile_cache:
            _cell_graph_executor.del_net_res(self.compile_cache)

    def __delattr__(self, name):
        if name in self._params:
            del self._params[name]
        elif name in self._cells:
            del self._cells[name]
        else:
            if '_params_list' in self.__dict__ and name in self._params_list:
                del self._params_list[name]
            elif '_tensor_list' in self.__dict__ and name in self._tensor_list:
                del self._tensor_list[name]
            object.__delattr__(self, name)
        self._attr_synced = False

    def _cast_mixed_precision_inputs(self, inputs, dst_type):
        """Cast input for mixed precision"""
        res = list()
        for item in inputs:
            if isinstance(item, tuple):
                res.append(self._cast_mixed_precision_inputs(item, dst_type))
            elif isinstance(item, float):
                res.append(self.cast(item, dst_type))
            elif hasattr(item, "dtype") and item.dtype in {mstype.float16, mstype.float32, mstype.float64}:
                res.append(self.cast(item, dst_type))
            else:
                res.append(item)
        return tuple(res)

    def cast_inputs(self, inputs, dst_type):
        """
        Cast inputs to specified type.

        Args:
            inputs (tuple[Tensor]): The cell inputs.
            dst_type (mindspore.dtype): The specified data type.

        returns:
            tuple[Tensor], the result with destination data type.
        """
        res = list()
        for item in inputs:
            if isinstance(item, tuple):
                res.append(self.cast_inputs(item, dst_type))
            else:
                res.append(self.cast(item, dst_type))
        return tuple(res)

    def _do_parameter_broadcast(self):
        if context.get_auto_parallel_context("parallel_mode") == ParallelMode.DATA_PARALLEL:
            if not self.parameter_broadcast_done:
                _pynative_executor.parameter_broadcast(self, self.phase, self._auto_parallel_mode)
                self.parameter_broadcast_done = True

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

    def _run_construct(self, cast_inputs, kwargs):
        """Run the construct function"""
        if self._enable_forward_pre_hook:
            cast_inputs = self._run_forward_pre_hook(cast_inputs)
        if self._enable_backward_hook:
            output = self._backward_hook_construct(*cast_inputs)
        elif hasattr(self, "_shard_fn"):
            output = self._shard_fn(*cast_inputs, **kwargs)
        else:
            output = self.construct(*cast_inputs, **kwargs)
        if self._enable_forward_hook:
            output = self._run_forward_hook(cast_inputs, output)
        return output

    def _check_construct_args(self, *inputs, **kwargs):
        """Check the args needed by the function construct"""
        if kwargs:
            raise ValueError(f"For 'Cell', expect no kwargs here, maybe you pass wrong arguments, "
                             f"or there is a key in kwargs that is not used as a function argument. "
                             f"args: {inputs}, kwargs: {kwargs}")
        positional_args = 0
        default_args = 0
        for value in inspect.signature(self.construct).parameters.values():
            if value.kind is inspect.Parameter.VAR_POSITIONAL or value.kind is inspect.Parameter.VAR_KEYWORD:
                return
            if value.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                if value.default is inspect.Parameter.empty:
                    positional_args += 1
                else:
                    default_args += 1

        if len(inputs) < positional_args:
            raise TypeError(f"For 'Cell', the function construct requires {positional_args} positional argument, "
                            f"but got {len(inputs)}. When using set_inputs, please make sure that all networks "
                            f"and loss functions are configured with set_inputs.")

        if len(inputs) > positional_args + default_args:
            raise TypeError(f"For 'Cell', the function construct requires {positional_args} positional argument and "
                            f"{default_args} default argument, total {positional_args + default_args}, "
                            f"but got {len(inputs)}.")

    def _hook_fn_registered(self):
        if self._enable_forward_pre_hook or self._enable_forward_hook or self._enable_backward_hook:
            return True
        for cell in self.cells():
            if cell._hook_fn_registered():
                return True
        return False

    def _get_prims_recursively(self):
        all_prims = list()
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
        if context._get_mode() == context.PYNATIVE_MODE:
            raise ValueError("set_data_parallel: does not support PyNative mode.")

        all_prims = self._get_prims_recursively()
        for prim in all_prims:
            prim.add_prim_attr("strategy_gen_mode", "data_parallel")

    def shard(self, in_strategy, out_strategy, parameter_plan=None, device="Ascend", level=0):
        """
        Defining the input and output layouts of this cell and the parallel strategies of remaining ops will be
        generated by sharding propagation. In PyNative mode, use this method
        to specify a Cell for distributed execution in graph mode.
        in_strategy and out_strategy define the input and output layout respectively.
        in_strategy/out_strategy should be a tuple, each element of which corresponds to the desired layout of
        this input/output, and None represents data_parallel,
        which can refer to the description of `mindspore.ops.Primitive.shard`.
        The parallel strategies of remaining operators are derived from the strategy specified by the input and output.

        Note:
            Only effective in PYNATIVE_MODE and in either ParallelMode.AUTO_PARALLEL with
            search_mode in auto_parallel_context set as sharding_propagation.

        Args:
            in_strategy (tuple): Define the layout of inputs, each element of the tuple should be a tuple or None. Tuple
                             defines the layout of the corresponding input and None represents a data parallel strategy.
            out_strategy (tuple): Define the layout of outputs similar with in_strategy.
            parameter_plan (Union[dict, None]): Define the layout for the specified parameters. Each element in dict
                                                defines the layout of the parameter like "param_name: layout".
                                                The key is a parameter name of type 'str'.
                                                The value is a 1-D integer tuple, indicating the corresponding layout.
                                                If the parameter name is incorrect or the corresponding parameter
                                                has been set, the parameter setting will be ignored.
                                                Default: None.
            device (string): Select a certain device target. It is not in use right now.
                             Support ["CPU", "GPU", "Ascend"]. Default: "Ascend".
            level (int): Option for parallel strategy infer algorithm, namely the object function, maximize computation
                         over communication ratio, maximize speed performance, minimize memory usage etc. It is not in
                         use right now. Support ["0", "1", "2"]. Default: "0".

        Returns:
            Cell, the cell itself.

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
            ...     self.block2.shard(in_strategy=((2, 1),), out_strategy=(None,),
            ...                       parameter_plan={'self.block2.shard.dense1.weight': (4, 1)})
            ...   def construct(self, x):
            ...     x = self.block1(x)
            ...     x = self.block2(x)
            ...     return x
        """
        # Transfer parameter_plan from dict to tuple
        shard_fn = Shard()
        fn = shard_fn(self, in_strategy, out_strategy, parameter_plan, device, level)
        object.__setattr__(self, "_shard_fn", fn)
        return self

    def auto_cast_inputs(self, inputs):
        """
        Auto cast inputs in mixed precision scenarios.

        Args:
            inputs (tuple): the inputs of construct.

        Returns:
            Tuple, the inputs after data type cast.
        """
        cast_inputs = inputs
        mixed_type = self.get_mixed_precision_type()
        if mixed_type == MixedPrecisionType.FP16:
            cast_inputs = self._cast_mixed_precision_inputs(inputs, mstype.float16)
        if mixed_type == MixedPrecisionType.FP32:
            cast_inputs = self._cast_mixed_precision_inputs(inputs, mstype.float32)

        return cast_inputs

    def __call__(self, *args, **kwargs):
        if self.__class__.construct is Cell.construct:
            logger.warning(f"The '{self.__class__}' does not override the method 'construct', "
                           f"it will call the super class(Cell) 'construct'.")
        if kwargs:
            bound_arguments = inspect.signature(self.construct).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            kwargs = bound_arguments.kwargs

        # Run in Graph mode.
        if context._get_mode() == context.GRAPH_MODE:
            self._check_construct_args(*args, **kwargs)
            if self._hook_fn_registered():
                logger.warning(f"For 'Cell', it's not support hook function in graph mode. If you want to use hook "
                               f"function, please use context.set_context to set pynative mode.")
            out = self.compile_and_run(*args)
            return out

        # Run in PyNative mode.
        if _pynative_executor.is_first_cell():
            _pynative_executor.set_lazy_build(True)
            _pynative_executor._optimizer = getattr(self, "optimizer", None)
            _pynative_executor._top_cell = self
            # There many Casts in parameter_broadcast. Enable lazy_build and build faster.
            self._do_parameter_broadcast()

        _check_args(args)

        if self.requires_grad:
            _pynative_executor.set_grad_flag(True)

        if self._dynamic_shape_inputs is not None:
            self._check_compile_dynamic_shape(*args)

        try:
            _pynative_executor.new_graph(self, *args, **kwargs)
            cast_inputs = self.auto_cast_inputs(args)
            output = self._run_construct(cast_inputs, kwargs)
            _pynative_executor.end_graph(self, output, *args, **kwargs)
        except Exception as err:
            _pynative_executor.clear_res()
            raise err

        if isinstance(output, Parameter):
            output = output.data
        return output

    def _add_attr(self, name, value):
        if name and name[:2] != '__' and name not in Cell.IGNORE_LIST:
            super(Cell, self)._add_attr(name, value)

    def _sync_attr_for_compile(self):
        """Sync the attr to c++ object."""
        if self._attr_synced:
            return
        cells = self.__dict__.get('_cells')
        for key in cells:
            cell = cells[key]
            cell._sync_attr_for_compile()
            self._add_attr(key, cell)
        params = self.__dict__.get('_params')
        for key in params:
            if '.' in key:
                continue
            param = params[key]
            self._add_attr(key, param)
        params_list = self.__dict__.get('_params_list')
        for key in params_list:
            params_list_item = params_list[key]
            self._add_attr(key, params_list_item)
        for key in self.__dict__:
            value = self.__dict__[key]
            self._add_attr(key, value)
        self._attr_synced = True

    def _set_attr_for_parameter(self, name, value):
        """Set attr for parameter."""
        cells = self.__dict__.get('_cells')
        params = self.__dict__.get('_params')
        if params is None:
            raise AttributeError("For 'Cell', can not assign params before Cell.__init__() is called.")
        if name in self.__dict__:
            if self.__dict__[name] is not None:
                raise TypeError(f"For 'Cell', the {name} should not be Parameter.")
            del self.__dict__[name]
        if cells and name in cells:
            raise TypeError(f"For 'Cell', the {name} must be Cell, but got Parameter.")
        self.insert_param_to_cell(name, value)

    def _set_attr_for_parameter_tuple(self, name, value):
        """Set attr for parameter in ParameterTuple."""
        params = self.__dict__.get('_params')
        params_list = self.__dict__.get('_params_list')
        if params is None:
            raise AttributeError("For 'Cell', can not assign params before Cell.__init__() is called.")
        exist_names = set("")
        exist_objs = set()
        for item in value:
            if item in exist_objs:
                # If there are multiple identical objects, their names only check once.
                continue
            exist_objs.add(item)
            if item.name == PARAMETER_NAME_DEFAULT:
                logger.warning("For 'Cell', the parameter definition is deprecated.\n"
                               "Please set a unique name for the parameter in ParameterTuple '{}'.".format(value))
                item.name = item.name + "$" + str(self._id)
                self._id += 1
            self.insert_param_to_cell(item.name, item, check_name_contain_dot=False)
            if item.name in exist_names:
                raise ValueError("The value {} , its name '{}' already exists. "
                                 "Please set a unique name for the parameter.".format(value, item.name))
            exist_names.add(item.name)

        if context._get_mode() == context.PYNATIVE_MODE:
            if name in self.__dict__:
                del self.__dict__[name]
            if name in params:
                del params[name]
            params_list[name] = value
        else:
            object.__setattr__(self, name, value)

    def _set_attr_for_parameter_in_list_or_tuple(self, name, value):
        """Set attr for parameter in list or tuple."""
        for item in value:
            if item in self.exist_objs:
                # If there are multiple identical objects, their names only check once.
                continue
            self.exist_objs.add(item)
            if item.name == PARAMETER_NAME_DEFAULT:
                item.name = item.name + "$" + str(self._id)
                self._id += 1
            if item.name in self.exist_names:
                raise ValueError("The value {} , its name '{}' already exists. "
                                 "Please set a unique name for the parameter.".format(value, item.name))
            self.exist_names.add(item.name)
        object.__setattr__(self, name, value)

    def _set_attr_for_cell(self, name, value):
        """Set attr for cell."""
        cells = self.__dict__.get('_cells')
        params = self.__dict__.get('_params')
        if cells is None:
            raise AttributeError("For 'Cell', can not assign cells before Cell.__init__() is called.")
        if name in self.__dict__:
            del self.__dict__[name]
        if params and name in params:
            raise TypeError(f"For 'Cell', the {name} must be Parameter, but got Cell.")
        if self._auto_prefix:
            value.update_parameters_name(name + '.')
        cells[name] = value
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

    def _set_attr_for_tensor(self, name, value):
        if context._get_mode() == context.PYNATIVE_MODE:
            tensor_list = self.__dict__.get('_tensor_list')
            if name in self.__dict__:
                del self.__dict__[name]
            tensor_list[name] = value
        else:
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        cells = self.__dict__.get('_cells')
        params = self.__dict__.get('_params')
        if isinstance(value, Parameter):
            self._set_attr_for_parameter(name, value)
        elif isinstance(value, ParameterTuple):
            self._set_attr_for_parameter_tuple(name, value)
        elif isinstance(value, (list, tuple)) and value and _check_param_list_tuple(value):
            self._set_attr_for_parameter_in_list_or_tuple(name, value)
        elif isinstance(value, Cell):
            self._set_attr_for_cell(name, value)
        elif params and name in params:
            self._set_attr_for_params(name, value)
        elif cells and name in cells:
            if value is not None:
                raise TypeError(f"For 'Cell', the type of {name} must be cell, but got {type(value).__name__}.")
            self._cells[name] = None
        elif isinstance(value, Tensor):
            self._set_attr_for_tensor(name, value)
        else:
            if isinstance(value, Primitive):
                value.set_prim_instance_name(name)
                self._primitives[name] = value
            object.__setattr__(self, name, value)
        if name not in Cell.IGNORE_LIST:
            self._attr_synced = False

    def extend_repr(self):
        """
        Expand the description of Cell.

        To print customized extended information, re-implement this method in your own cells.
        """
        return ''

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        extra_str = self.extend_repr()
        info_str = self.__class__.__name__ + '<'
        if self._cells:
            sub_str = '\n'
            if extra_str:
                sub_str += '{}\n'.format(self.extend_repr())
            for key, value in self._cells.items():
                sub_str += '({}): {}\n'.format(key, repr(value))
            sub_str = sub_str.replace('\n', '\n  ') + '>'
            info_str += sub_str
        else:
            info_str += extra_str + '>'
        return info_str

    def load_parameter_slice(self, params):
        """
        Replace parameters with sliced tensors by parallel strategies.

        Please refer to the usage in source code of `mindspore.common._CellGraphExecutor.compile`.

        Args:
            params (dict): The parameters dictionary used for initializing the data graph.
        """
        if params is None:
            params = self.parameters_dict()
        if isinstance(params, OrderedDict):
            for key in params:
                tensor = params[key].data
                if key not in self.parameter_layout_dict:
                    logger.info("The layout dict does not contain the key %s.", key)
                    continue
                if params[key].sliced:
                    logger.debug("The param %s is already sliced.", key)
                    continue
                layout = self.parameter_layout_dict[key]
                new_tensor = _load_tensor_by_layout(tensor, layout)
                params[key].set_data(new_tensor, True)
        else:
            raise TypeError("For 'load_parameter_slice', the argument 'params' must be OrderedDict type, "
                            "but got {}.".format(type(params)))

    def _load_inputs(self, *inputs):
        """
        Slice inputs tensors by parallel strategies.

        Args:
            inputs (Function or Cell): inputs of construct method.
        """
        parallel_inputs_run = []
        # judge if *args exists in input
        if self.argspec[1] is not None:
            prefix = self.argspec[1]
            for i in range(len(inputs)):
                key = prefix + str(i)
                self._construct_inputs_names = self._construct_inputs_names + (key,)
                self._construct_inputs_num = self._construct_inputs_num + 1
        for i, tensor in enumerate(inputs):
            key = self._construct_inputs_names[i]
            # if input is not used, self.parameter_layout_dict may not contain the key
            if key not in self.parameter_layout_dict:
                logger.warning("Layout dict does not contain the key %s.", key)
                parallel_inputs_run.append(tensor)
            else:
                layout = self.parameter_layout_dict[key]
                new_tensor = _load_tensor_by_layout(tensor, layout)
                parallel_inputs_run.append(new_tensor)
        return tuple(parallel_inputs_run)

    def set_parallel_input_with_inputs(self, *inputs):
        """
        Slice inputs tensors by parallel strategies.

        Args:
            inputs (tuple): inputs of construct method.
        """
        self._parallel_inputs_run = self._load_inputs(*inputs)

    def _get_construct_inputs_number_and_name(self):
        """Compute self._construct_inputs_names and self._construct_inputs_num"""
        from mindspore._extends.parse.parser import get_parse_method_of_class

        fn = get_parse_method_of_class(self)
        self.argspec = inspect.getfullargspec(fn)
        self._construct_inputs_num = fn.__code__.co_argcount
        self._construct_inputs_names = fn.__code__.co_varnames

        if self._construct_inputs_num <= 0:
            raise ValueError(f"For 'set_auto_parallel', the number of inputs must be greater than 0,"
                             f"but got {self._construct_inputs_num}.")
        if self._construct_inputs_names[0] != 'self':
            raise ValueError(f"First member of fn function must be self, but got {self._construct_inputs_names[0]}")
        if self._construct_inputs_num - 1 > len(self._construct_inputs_names):
            raise ValueError(f"Num of inputs must be greater than num of fn function members, num of inputs is \
                {self._construct_inputs_names - 1}, num of fn function members is {len(self._construct_inputs_names)}")
        self._construct_inputs_names = self._construct_inputs_names[1:self._construct_inputs_num]
        self._construct_inputs_num = self._construct_inputs_num - 1

    def set_inputs(self, *inputs):
        """
        Save set inputs for computation graph. The number of inputs should be the same with that of the datasets. When
        using Model for dynamic shape, please make sure that all networks and loss functions passed to the Model are
        configured with set_inputs. The inputs can be Tensor of either dynamic or static shape.

        Args:
            inputs (tuple): Inputs of the Cell object.

        Note:
            This is an experimental interface that is subject to change or deletion.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn, Tensor, context
            >>>
            >>> class reluNet(nn.Cell):
            ...     def __init__(self):
            ...         super(reluNet, self).__init__()
            ...         self.relu = nn.ReLU()
            ...     def construct(self, x):
            ...         return self.relu(x)
            >>>
            >>> net = reluNet()
            >>> input_dyn = Tensor(shape=[3, None], dtype=ms.float32)
            >>> net.set_inputs(input_dyn)
            >>> input1 = Tensor(np.random.random([3, 10]), dtype=ms.float32)
            >>> output = net(input1)
        """
        if self.grad_ops_label:
            logger.warning(f'For Cell, set_inputs must be set before the gradient function of the network is '
                           f'generated.')
        for ele in inputs:
            if isinstance(ele, str):
                raise TypeError(f"For element in 'set_inputs', the type must not be str.")
        self._dynamic_shape_inputs = inputs
        self._check_construct_args(*inputs)
        if self._dynamic_shape_inputs:
            ds.config.set_dynamic_shape(True)
        if context._get_mode() == context.PYNATIVE_MODE:
            _pynative_executor.set_dynamic_input(self, *self._dynamic_shape_inputs)

    def get_inputs(self):
        """
        Returns the dynamic_inputs of a cell object in one network.

        Returns:
            inputs (tuple), Inputs of the Cell object.

        Note:
            This is an experimental interface that is subject to change or deletion.
        """

        return self._dynamic_shape_inputs

    def compile(self, *inputs):
        """
        Compile Cell as a computation graph, the input must be consistent with the input defined in construct.

        Args:
            inputs (tuple): Inputs of the Cell object.
        """
        if self._dynamic_shape_inputs is None or self._dynamic_shape_inputs[0] is None:
            _cell_graph_executor.compile(self, *inputs, phase=self.phase, auto_parallel_mode=self._auto_parallel_mode,
                                         jit_config_dict=self._jit_config_dict)
        else:
            self._check_compile_dynamic_shape(*inputs)
            if self.saved_dynamic_shape:
                for i in range(len(self.saved_dynamic_shape)):
                    if self.saved_dynamic_shape[i].shape != self._dynamic_shape_inputs[i].shape:
                        return

            self.saved_dynamic_shape = self._dynamic_shape_inputs
            _cell_graph_executor.compile(self, *self._dynamic_shape_inputs, phase=self.phase,
                                         auto_parallel_mode=self._auto_parallel_mode,
                                         jit_config_dict=self._jit_config_dict)
            logger.debug("Compiled Graph with dynamic shape")

    def compile_and_run(self, *inputs):
        """
        Compile and run Cell, the input must be consistent with the input defined in construct.

        Note:
            It is not recommended to call directly.

        Args:
            inputs (tuple): Inputs of the Cell object.

        Returns:
            Object, the result of executing.
        """
        self._auto_parallel_compile_and_run = True
        self.compile(*inputs)

        new_inputs = _get_args_for_run(self, inputs)
        return _cell_graph_executor(self, *new_inputs, phase=self.phase)

    def auto_parallel_compile_and_run(self):
        """
        Whether or not to execute compile and run in 'AUTO_PARALLEL' or 'SEMI_AUTO_PARALLEL' mode.

        Returns:
            bool, `_auto_parallel_compile_and_run` value.
        """
        return self._auto_parallel_compile_and_run

    def exec_checkpoint_graph(self):
        """Executes saving checkpoint graph operation."""
        _cell_graph_executor(self, phase='save')

    def insert_param_to_cell(self, param_name, param, check_name_contain_dot=True):
        """
        Adds a parameter to the current cell.

        Inserts a parameter with given name to the cell. The method is currently used in
        `mindspore.nn.Cell.__setattr__`.

        Args:
            param_name (str): Name of the parameter.
            param (Parameter): Parameter to be inserted to the cell.
            check_name_contain_dot (bool): Determines whether the name input is compatible. Default: True.

        Raises:
            KeyError: If the name of parameter is null or contains dot.
            TypeError: If the type of parameter is not Parameter.
        """
        if not param_name:
            raise KeyError("For 'insert_param_to_cell', the argument 'param_name' should not be None.")
        if check_name_contain_dot and '.' in param_name:
            raise KeyError("For 'insert_param_to_cell', the argument 'param_name' should not contain \".\"")
        if '_params' not in self.__dict__:
            raise AttributeError("For 'insert_param_to_cell', please call Cell.__init__() firstly.")
        if hasattr(self, param_name) and param_name not in self._params:
            raise KeyError("For 'insert_param_to_cell', the {} parameter already exists in the network. Cannot "
                           "insert another parameter with the same name.".format(param_name))
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"For 'insert_param_to_cell', the argument 'param' must be 'Parameter' if not None, "
                            f"but got {type(param)}.")
        if isinstance(param, Parameter) and param.name == PARAMETER_NAME_DEFAULT:
            param.name = param_name
        self._params[param_name] = param

    def cast_param(self, param):
        """
        Cast parameter according to auto mix precision level in pynative mode.

        This interface is currently used in the case of auto mix precision and usually needs not to be used explicitly.

        Args:
            param (Parameter): Parameters, the type of which should be cast.

        Returns:
            Parameter, the input parameter with type automatically cast.
        """
        mixed_type = self.get_mixed_precision_type()
        if mixed_type != MixedPrecisionType.NOTSET:
            if mixed_type == MixedPrecisionType.FP32:
                param.set_cast_dtype(mstype.float32)
            elif mixed_type == MixedPrecisionType.FP16:
                param.set_cast_dtype(mstype.float16)
        elif hasattr(param, "set_cast_dtype"):
            # retest dtype
            param.set_cast_dtype()
        return param

    def insert_child_to_cell(self, child_name, child_cell):
        """
        Adds a child cell to the current cell with a given name.

        Args:
            child_name (str): Name of the child cell.
            child_cell (Cell): The child cell to be inserted.

        Raises:
            KeyError: Child Cell's name is incorrect or duplicated with the other child name.
            TypeError: Child Cell's type is incorrect.
        """
        if not child_name or '.' in child_name:
            raise KeyError("For 'insert_child_to_cell', the parameter 'child_name' can not be None and "
                           "can not contain '.'")
        if hasattr(self, child_name) and child_name not in self._cells:
            raise KeyError("For 'insert_child_to_cell', the {} child cell already exists in the network. Cannot "
                           "insert another child cell with the same name.".format(child_name))
        if not isinstance(child_cell, Cell) and child_cell is not None:
            raise TypeError(f"For 'insert_child_to_cell', the argument 'child_cell' must be 'Cell' if not None, "
                            f"but got type {type(child_cell)}.")
        self._cells[child_name] = child_cell

    def construct(self, *inputs, **kwargs):
        """
        Defines the computation to be performed. This method must be overridden by all subclasses.

        Note:
            It is not supported currently that inputs contain both tuple and non-tuple types at same time.

        Args:
            inputs (tuple): Tuple of variable parameters.
            kwargs (dict): Dictionary of variable keyword parameters.

        Returns:
            Tensor, returns the computed result.
        """
        return None

    def remove_redundant_parameters(self):
        """
        Remove the redundant parameters.

        This interface usually needs not to be used explicitly.
        """
        cells = self.cells_and_names()
        for _, cell in cells:
            params = cell._params.items()
            for param_name, param in list(params):
                if param.name not in self.parallel_parameter_name_list:
                    cell._params.pop(param_name)
                    logger.info("remove the redundant parameter: %s", param.name)
                    continue
            cell_dict = cell.__dict__
            for key in cell_dict:
                if isinstance(cell_dict[key], ParameterTuple):
                    param_tuple = cell_dict[key]
                    new_param_tuple = []
                    for param in param_tuple:
                        if param.name not in self.parallel_parameter_name_list:
                            logger.info("remove the redundant parameter: %s in ParameterTuple", param.name)
                            continue
                        new_param_tuple.append(param)
                    cell.__dict__[key] = ParameterTuple(new_param_tuple)

    def init_parameters_data(self, auto_parallel_mode=False):
        """
        Initialize all parameters and replace the original saved parameters in cell.

        Note:
            trainable_params() and other similar interfaces may return different parameter instance after
            `init_parameters_data`, do not save these results.

        Args:
            auto_parallel_mode (bool): If running in auto_parallel_mode. Default: False.

        Returns:
            Dict[Parameter, Parameter], returns a dict of original parameter and replaced parameter.
        """
        replace = dict()

        def _updata(param):
            if param in replace:
                return replace.get(param)
            layout = None
            set_sliced = False
            if auto_parallel_mode:
                set_sliced = True
                if param.name not in self.parameter_layout_dict:
                    logger.debug("Layout dict does not contain the key %s.", param.name)
                else:
                    layout = self.parameter_layout_dict[param.name]
            new_p = param.init_data(layout, set_sliced=set_sliced)
            replace[param] = new_p
            return new_p

        # replace all original usage.
        cells = self.cells_and_names()
        for _, cell in cells:
            params = cell._params.items()
            for param_name, param in params:
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
            recurse (bool): Whether contains the parameters of subcells. Default: True.

        Returns:
            OrderedDict, return parameters dictionary.
        """
        param_dict = OrderedDict()
        for param in self.get_parameters(expand=recurse):
            param_dict[param.name] = param
        return param_dict

    def parameters_broadcast_dict(self, recurse=True):
        """
        Gets the parameters broadcast dictionary of this cell.

        Args:
            recurse (bool): Whether contains the parameters of subcells. Default: True.

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
            prefix (str): The prefix string. Default: ''.
            recurse (bool): Whether contains the parameters of subcells. Default: True.
        """

        Validator.check_str_by_regular(prefix)
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
            recurse (bool): Whether contains the parameters of subcells. Default: True.
        """

        Validator.check_str_by_regular(prefix)
        for name, param in self.parameters_and_names(expand=recurse):
            if name in self._user_parameters:
                continue
            if prefix != '':
                param.is_init = False
            param.name = prefix + name

    def trainable_params(self, recurse=True):
        """
        Returns all trainable parameters.

        Returns a list of all trainable parameters.

        Args:
            recurse (bool): Whether contains the trainable parameters of subcells. Default: True.

        Returns:
            List, the list of trainable parameters.
        """
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=recurse)))

    def untrainable_params(self, recurse=True):
        """
        Returns all untrainable parameters.

        Returns a list of all untrainable parameters.

        Args:
            recurse (bool): Whether contains the untrainable parameters of subcells. Default: True.

        Returns:
            List, the list of untrainable parameters.
        """
        return list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=recurse)))

    def get_parameters(self, expand=True):
        """
        Returns an iterator over cell parameters.

        Yields parameters of this cell. If `expand` is true, yield parameters of this cell and all subcells.

        Args:
            expand (bool): If true, yields parameters of this cell and all subcells. Otherwise, only yield parameters
                           that are direct members of this cell. Default: True.

        Returns:
            Iteration, all parameters at the cell.

        Examples:
            >>> from mindspore import nn
            >>> net = nn.Dense(3, 4)
            >>> parameters = []
            >>> for item in net.get_parameters():
            ...     parameters.append(item)
        """
        for _, param in self.parameters_and_names(expand=expand):
            yield param

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
            name_prefix (str): Namespace. Default: ''.
            expand (bool): If true, yields parameters of this cell and all subcells. Otherwise, only yield parameters
                           that are direct members of this cell. Default: True.

        Returns:
            Iteration, all the names and corresponding parameters in the cell.

        Examples:
            >>> from mindspore import nn
            >>> n = nn.Dense(3, 4)
            >>> names = []
            >>> for m in n.parameters_and_names():
            ...     if m[0]:
            ...         names.append(m[0])
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
                if par.inited_param is not None:
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
            cells (str): Cells to iterate over. Default: None.
            name_prefix (str): Namespace. Default: ''.

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
                for ele in cell.cells_and_names(t_cells, cells_name_prefix):
                    yield ele

    def cells(self):
        """
        Returns an iterator over immediate cells.

        Returns:
            Iteration, the immediate cells in the cell.
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
            Cell_.set_mixed_precision_type(self, MixedPrecisionType.FP16)
        if "fp32" in flags and flags.get("fp32", False):
            Cell_.set_mixed_precision_type(self, MixedPrecisionType.FP32)

    def _add_mixed_precision_flag_recursive(self, **flags):
        """Add mixed precision flag to each cell"""
        if "fp16" in flags and flags.get("fp16", False):
            self._set_mixed_precision_type_recursive(MixedPrecisionType.FP16)
        if "fp32" in flags and flags.get("fp32", False):
            self._set_mixed_precision_type_recursive(MixedPrecisionType.FP32)

    def add_flags(self, **flags):
        """
        Add customized attributes for cell.

        This method is also called when the cell class is instantiated and the class parameter 'flags' is set to True.

        Args:
            flags (dict): Network configuration information, currently it is used for the binding of network and
                dataset. Users can also customize network attributes by this parameter. Default: None.
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
                dataset. Users can also customize network attributes by this parameter. Default: None.
        """
        self.add_flags(**flags)
        self._add_mixed_precision_flag_recursive(**flags)
        for cell in self.cells():
            cell.add_flags_recursive(**flags)
        return self

    def _add_init_args(self, **args):
        if hasattr(self, '_cell_init_args'):
            self._cell_init_args += str({**args})

    def get_flags(self):
        """
        Get the self_defined attributes of the cell, which can be added by `add_flags` method.
        """
        if not hasattr(self, "_func_graph_flags"):
            self._func_graph_flags = {}
        return self._func_graph_flags

    def _set_mixed_precision_type_recursive(self, mixed_type):
        """Set mixed precision type to each cell"""
        Cell_.set_mixed_precision_type(self, mixed_type)
        for cell in self.cells():
            cell._set_mixed_precision_type_recursive(mixed_type)

    def to_float(self, dst_type):
        """
        Add cast on all inputs of cell and child cells to run with certain float type.

        If `dst_type` is `mindspore.dtype.float16`, all the inputs of Cell, including input, Parameter and Tensor, will
        be cast to float16. Please refer to the usage in source code of :func:`mindspore.build_train_network`.

        Note:
            Multiple calls will overwrite.

        Args:
            dst_type (:class:`mindspore.dtype`): Transfer cell to run with dst_type.
                dst_type can be `mstype.float16` or `mstype.float32`.

        Returns:
            Cell, the cell itself.

        Raises:
            ValueError: If dst_type is not mstype.float32 or mstype.float16.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import mindspore.nn as nn
            >>> from mindspore import dtype as mstype
            >>>
            >>> net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
            >>> net.to_float(mstype.float16)
            Conv2d<input_channels=120, output_channels=240, kernel_size=(4, 4), stride=(1, 1), pad_mode=same,
            padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
        """
        if dst_type not in (mstype.float16, mstype.float32):
            raise ValueError("For 'to_float', the argument 'dst_type' must be float32 or float16, "
                             "but got {}.".format(dst_type))
        if dst_type == mstype.float16:
            self._set_mixed_precision_type_recursive(MixedPrecisionType.FP16)
        else:
            self._set_mixed_precision_type_recursive(MixedPrecisionType.FP32)
        flags = {'fp16': dst_type == mstype.float16, 'fp32': dst_type == mstype.float32}
        self._add_init_args(**flags)
        return self

    def set_boost(self, boost_type):
        """
        In order to improve the network performance, configure the network auto enable to
        accelerate the algorithm in the algorithm library.

        If `boost_type` is not in the algorithm library, please view the algorithm in the algorithm library through
        `algorithm library <https://gitee.com/mindspore/mindspore/tree/r1.10/mindspore/python/mindspore/boost>`_.

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
        Sets the cell flag for gradient. In pynative mode, this parameter specifies whether the network requires
        gradients. If true, the backward network needed to compute the gradients will be generated when the forward
        network is executed.

        Args:
            requires_grad (bool): Specifies if the net need to grad, if it is
                true, the cell will construct backward network in pynative mode. Default: True.

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

        Args:
            mode (bool): Specifies whether the model is training. Default: True.

        Returns:
            Cell, the cell itself.
        """
        if mode is False:
            self._phase = 'predict'
        else:
            self._phase = 'train'
        self.add_flags_recursive(training=mode)
        return self

    def set_broadcast_flag(self, mode=True):
        """
        Set parameter broadcast mode for this cell.

        Args:
            mode (bool): Specifies whether the mode is parameter broadcast. Default: True.
        """
        self.add_flags_recursive(broadcast_flag=mode)
        return self

    def set_auto_parallel(self):
        """
        Set the cell to auto parallel mode.

        Note:
            If a cell needs to use the auto parallel or semi auto parallel mode for training, evaluation or prediction,
            this interface needs to be called by the cell.
        """
        self._auto_parallel_mode = True
        self._get_construct_inputs_number_and_name()

    def set_jit_config(self, jit_config):
        """
        Set jit config for cell.

        Args:
            jit_config (JitConfig): Jit config for compile. For details, please refer to :class:`mindspore.JitConfig`.
        """
        if self._jit_config_dict:
            logger.warning("For Cell, jit config can only be set once, ignore this setting.")
        else:
            self._jit_config_dict = jit_config.jit_config_dict

    def flatten_weights(self, fusion_size=0):
        """
        Reset data for weight parameters so that they are using contiguous memory chunks grouped by data type.

        Note:
            By default, parameters with same data type will using a single contiguous memory chunk. but for
            some models with huge number of parameters, splitting a large memory chunk into several smaller
            memory chunks has the potential for performance gains, if this is the case, we can use 'fusion_size'
            to limit the maximum memory chunk size.

        Args:
            fusion_size (int): Maximum memory chunk size in bytes, 0 for unlimited. Default: 0.
        """
        if fusion_size < 0:
            raise ValueError(f"Negative 'fusion_size' {fusion_size} is invalid.")
        Tensor._flatten_tensors(self.trainable_params(), fusion_size)  # pylint: disable=W0212

    def register_forward_pre_hook(self, hook_fn):
        """
        Register forward pre hook function for Cell object.

        Note:
            - The `register_forward_pre_hook(hook_fn)` does not work in graph mode or ms_function.
            - 'hook_fn' must be defined as the following code.
              `cell_id` is the information of registered Cell object, including name and ID. `inputs` is the forward
              input objects passed to the Cell. The 'hook_fn' can modify the forward input objects by returning new
              forward input objects.
            - It should have the following signature:
              hook_fn(cell_id, inputs) -> new input objects or none.
            - In order to prevent running failed when switching to graph mode, it is not recommended to write it in the
              `construct` function of Cell object. In the pynative mode, if the `register_forward_pre_hook` function is
              called in the `construct` function of the Cell object, a hook function will be added at each run time of
              Cell object.

        Args:
            hook_fn (function): Python function. Forward pre hook function.

        Returns:
            Handle, it is an instance of `mindspore.common.hook_handle.HookHandle` and corresponding to the `hook_fn` .
            The handle can be used to remove the added `hook_fn` by calling `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> import mindspore.nn as nn
            >>> from mindspore import Tensor
            >>> from mindspore.ops import GradOperation
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def forward_pre_hook_fn(cell_id, inputs):
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
            >>> grad = GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
            forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1],
                            dtype=Float32, value= [ 1.00000000e+00]))
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32,
            value= [ 2.00000000e+00]))
        """
        if context.get_context("mode") != context.PYNATIVE_MODE:
            logger.warning(f"'register_forward_pre_hook' function is only supported in pynative mode, you can use "
                           f"context.set_context to set pynative mode.")
            return HookHandle()

        if not isinstance(hook_fn, (FunctionType, MethodType)):
            raise TypeError(f"When using 'register_forward_pre_hook(hook_fn)', the type of 'hook_fn' must be python "
                            f"function, but got {type(hook_fn)}.")
        if hook_fn.__code__.co_name == "staging_specialize":
            raise TypeError(f"Decorating hook function {hook_fn.__name__} with '@ms_function' is not supported.")

        self._enable_forward_pre_hook = True
        _pynative_executor.set_hook_changed(self)
        if not hasattr(self, '_forward_pre_hook_key'):
            self._forward_pre_hook_key = -1
        self._forward_pre_hook_key += 1
        self._forward_pre_hook[self._forward_pre_hook_key] = hook_fn
        handle = HookHandle(self, self._forward_pre_hook_key, "_forward_pre_hook")
        return handle

    def _run_forward_pre_hook(self, inputs):
        """
        Running forward pre hook function registered on Cell object.

        Args:
            inputs: The input objects of cell object.

        Returns:
            - **outputs** - New input objects or none.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
        """
        cell_id = self.cls_name + "(" + str(id(self)) + ")"
        for fn in self._forward_pre_hook.values():
            ret = fn(cell_id, inputs)
            if ret is not None:
                if not isinstance(ret, tuple):
                    inputs = (ret,)
                else:
                    inputs = ret
        return inputs

    def register_forward_hook(self, hook_fn):
        """
        Set the Cell forward hook function.

        Note:
            - The `register_forward_hook(hook_fn)` does not work in graph mode or ms_function.
            - 'hook_fn' must be defined as the following code.
              `cell_id` is the information of registered Cell object, including name and ID. `inputs` is the forward
              input objects passed to the Cell. `output` is the forward output object of the Cell. The 'hook_fn' can
              modify the forward output object by returning new forward output object.
            - It should have the following signature:
              hook_fn(cell_id, inputs, output) -> new output object or none.
            - In order to prevent running failed when switching to graph mode, it is not recommended to write it in the
              `construct` function of Cell object. In the pynative mode, if the `register_forward_hook` function is
              called in the `construct` function of the Cell object, a hook function will be added at each run time of
              Cell object.

        Args:
            hook_fn (function): Python function. Forward hook function.

        Returns:
            Handle, it is an instance of `mindspore.common.hook_handle.HookHandle` and corresponding to the `hook_fn` .
            The handle can be used to remove the added `hook_fn` by calling `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> import mindspore.nn as nn
            >>> from mindspore import Tensor
            >>> from mindspore.ops import GradOperation
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def forward_hook_fn(cell_id, inputs, output):
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
            >>> grad = GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32)))
            forward inputs: (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1],
                            dtype=Float32, value= [ 1.00000000e+00]))
            forward output: 2.0
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]), Tensor(shape=[1], dtype=Float32,
            value= [ 2.00000000e+00]))
        """
        if context.get_context("mode") != context.PYNATIVE_MODE:
            logger.warning(f"'register_forward_hook' function is only supported in pynative mode, you can use "
                           f"context.set_context to set pynative mode.")
            return HookHandle()

        if not isinstance(hook_fn, (FunctionType, MethodType)):
            raise TypeError(f"When using 'register_forward_hook(hook_fn)', the type of 'hook_fn' must be python "
                            f"function, but got {type(hook_fn)}.")
        if hook_fn.__code__.co_name == "staging_specialize":
            raise TypeError(f"Decorating hook function {hook_fn.__name__} with '@ms_function' is not supported.")

        self._enable_forward_hook = True
        _pynative_executor.set_hook_changed(self)
        if not hasattr(self, '_forward_hook_key'):
            self._forward_hook_key = -1
        self._forward_hook_key += 1
        self._forward_hook[self._forward_hook_key] = hook_fn
        handle = HookHandle(self, self._forward_hook_key, "_forward_hook")
        return handle

    def _run_forward_hook(self, inputs, output):
        """
        Running forward hook function registered on Cell object.

        Args:
            inputs: The input objects of Cell object.
            output: The output object of Cell object.

        Returns:
            - **output** - New output object or none.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
        """
        cell_id = self.cls_name + "(" + str(id(self)) + ")"
        for fn in self._forward_hook.values():
            ret = fn(cell_id, inputs, output)
            if ret is not None:
                output = ret
        return output

    def register_backward_hook(self, hook_fn):
        """
        Register the backward hook function.

        Note:
            - The `register_backward_hook(hook_fn)` does not work in graph mode or ms_function.
            - The 'hook_fn' must be defined as the following code.
              `cell_id` is the information of registered Cell object, including name and ID. `grad_input` is the
              gradient passed to the Cell. `grad_output` is the gradient computed and passed to the next Cell or
              primitive, which may be modified by returning a new output gradient.
            - The 'hook_fn' should have the following signature:
              hook_fn(cell_id, grad_input, grad_output) -> New output gradient or none.
            - The 'hook_fn' is executed in the python environment. In order to prevent running failed when switching to
              graph mode, it is not recommended to write it in the `construct` function of Cell object. In the pynative
              mode, if the `register_backward_hook` function is called in the `construct` function of the Cell object,
              a hook function will be added at each run time of Cell object.

        Args:
            hook_fn (function): Python function. Backward hook function.

        Returns:
            Handle, it is an instance of `mindspore.common.hook_handle.HookHandle` and corresponding to the `hook_fn` .
            The handle can be used to remove the added `hook_fn` by calling `handle.remove()` .

        Raises:
            TypeError: If the `hook_fn` is not a function of python.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> import mindspore.nn as nn
            >>> from mindspore import Tensor
            >>> from mindspore.ops import GradOperation
            >>> ms.set_context(mode=ms.PYNATIVE_MODE)
            >>> def backward_hook_fn(cell_id, grad_input, grad_output):
            ...     print("backward input: ", grad_input)
            ...     print("backward output: ", grad_output)
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
            >>> grad = GradOperation(get_all=True)
            >>> net = Net()
            >>> output = grad(net)(Tensor(np.ones([1]).astype(np.float32)))
            backward input: (Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]),)
            backward output: (Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]),)
            >>> print(output)
            (Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]),)
        """
        if context.get_context("mode") != context.PYNATIVE_MODE:
            logger.warning(f"'register_backward_hook' function is only supported in pynative mode, you can use "
                           f"context.set_context to set pynative mode.")
            return HookHandle()

        if not isinstance(hook_fn, (FunctionType, MethodType)):
            raise TypeError(f"When using 'register_backward_hook(hook_fn)', the type of 'hook_fn' must be python "
                            f"function, but got {type(hook_fn)}.")
        if self._cell_backward_hook is None:
            self._enable_backward_hook = True
            self._cell_backward_hook = inner.CellBackwardHook(self.cls_name + "(" + str(id(self)) + ")")
            backward_hook_key = self._cell_backward_hook.register_backward_hook(hook_fn)
            handle = HookHandle(self, backward_hook_key, "_cell_backward_hook")
        else:
            backward_hook_key = self._cell_backward_hook.register_backward_hook(hook_fn)
            handle = HookHandle(self, backward_hook_key, "_cell_backward_hook")
        return handle

    def _backward_hook_construct(self, *inputs):
        """
        Backward hook construct method to replace original construct method.

        Args:
            inputs: The input objects of Cell object.

        Returns:
            - **outputs** - The output objects of Cell object.

        Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
        """
        if len(inputs) > 1:
            inputs = self._cell_backward_hook(inputs)
        else:
            inputs = self._cell_backward_hook(*inputs)
        if isinstance(inputs, tuple):
            outputs = self.construct(*inputs)
        else:
            outputs = self.construct(inputs)
        outputs = self._cell_backward_hook(outputs)
        return outputs

    def set_param_ps(self, recurse=True, init_in_server=False):
        """
        Set whether the trainable parameters are updated by parameter server and whether the
        trainable parameters are initialized on server.

        Note:
            It only works when a running task is in the parameter server mode.

        Args:
            recurse (bool): Whether sets the trainable parameters of subcells. Default: True.
            init_in_server (bool): Whether trainable parameters updated by parameter server are
                initialized on server. Default: False.
        """
        params = self.trainable_params(recurse)
        for param in params:
            param.set_param_ps(init_in_server)

    def set_param_fl(self, push_to_server=False, pull_from_server=False, requires_aggr=True):
        """
        Set the way of parameter and server interaction.

        Args:
            push_to_server (bool): Whether the parameter should be pushed to server. Default: False.
            pull_from_server (bool): Whether the parameter should be pulled from server. Default: False.
            requires_aggr (bool): Whether the parameter should be aggregated in the server. Default: True.
        """
        params = self.parameters_and_names()
        for param in params:
            param[1].set_param_fl(push_to_server, pull_from_server, requires_aggr)

    def set_comm_fusion(self, fusion_type, recurse=True):
        """
        Set `comm_fusion` for all the parameters in this cell. Please refer to the description of
        :class:`mindspore.Parameter.comm_fusion`.

        Note:
            The value of attribute will be overwritten when the function is called multiply.

        Args:
            fusion_type (int): The value of `comm_fusion`.
            recurse (bool): Whether sets the trainable parameters of subcells. Default: True.
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
        if context._get_mode() == context.PYNATIVE_MODE:
            raise TypeError("Recompute is not supported in pynative mode currently, you can use "
                            "'context.set_context(mode=context.GRAPH_MODE)' to set graph mode.")
        Validator.check_bool(mode)
        Validator.check_bool(output_recompute)
        if not self._has_config_recompute:
            self._has_config_recompute = True
        else:
            raise RuntimeError("The recompute interface can be configured only once."
                               " When the parent cell is configured, the child cell should not be configured")
        self._set_recompute_scope(mode)
        if mode and not output_recompute:
            self.add_flags(output_no_recompute=True)
        for cell in self.cells():
            cell._recompute(mode, True)

    @args_type_check(mp_comm_recompute=bool, parallel_optimizer_comm_recompute=bool)
    def recompute(self, **kwargs):
        """
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

        Args:
            mp_comm_recompute (bool): Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            parallel_optimizer_comm_recompute (bool): Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
        """
        self._recompute()
        if 'mp_comm_recompute' in kwargs.keys():
            self._mp_comm_recompute(kwargs.get('mp_comm_recompute', False))
        if 'parallel_optimizer_comm_recompute' in kwargs.keys():
            if (kwargs.get('parallel_optimizer_comm_recompute', False) and
                    context.get_auto_parallel_context("pipeline_stages") > 1):
                logger.warning("Currently, the communication operator allgathers introduced by optimizer shard "
                               "are not support recomputation in pipeline parallel.")
            elif context.get_auto_parallel_context("pipeline_stages") == 1:
                self._parallel_optimizer_comm_recompute(kwargs.get('parallel_optimizer_comm_recompute', False))
        if 'recompute_slice_activation' in kwargs.keys():
            self._recompute_slice_activation(kwargs.get('recompute_slice_activation', False))

        for key, _ in kwargs.items():
            if key not in ('mp_comm_recompute', 'parallel_optimizer_comm_recompute', 'recompute_slice_activation'):
                raise ValueError("For 'recompute', keyword '%s' is not recognized! "
                                 "the key kwargs must be 'mp_comm_recompute', "
                                 "'parallel_optimizer_comm_recompute', 'recompute_slice_activation'" % key)

    def infer_param_pipeline_stage(self):
        """
        Infer pipeline stages of all parameters in the cell.

        Note:
            - If a parameter does not belong to any cell which has been set pipeline_stage,
              the parameter should use add_pipeline_stage to add it's pipeline_stage information.
            - If a parameter P has been used by two operators in different stages "stageA" and "stageB",
              the parameter P should use P.add_pipeline_stage(stageA) and P.add_pipeline_stage(stageB)
              to add it's stage information before using infer_param_pipeline_stage.

        Returns:
            The params belong to current stage in pipeline parallel.

        Raises:
            RuntimeError: If there is a parameter does not belong to any stage.
        """
        from mindspore.parallel._utils import _get_global_rank, _get_device_num
        logger.warning(f"This interface may be deleted in the future.")
        stage_num = context.get_auto_parallel_context("pipeline_stages")
        device_num = _get_device_num()
        rank_id = _get_global_rank()
        per_stage_devices = device_num // stage_num
        current_stage = rank_id // per_stage_devices
        params = []
        for param in self.trainable_params():
            if not param._pipeline_stage_list:  # pylint: disable=W0212
                raise RuntimeError("For 'infer_param_pipeline_stage', the parameter {} does not belong to any stage, "
                                   "please check whether the cell where the param locates has been set "
                                   "'pipeline_stage'. Otherwise, the parameter should use 'add_pipeline_stage' "
                                   "to add its stage information".format(param.name))
            if current_stage in param._pipeline_stage_list:
                params.append(param)
        return params

    def _check_compile_dynamic_shape(self, *inputs):
        """
        Check if graph has been compiled with dynamic shape.

        Args:
            inputs (tuple): Inputs of the Cell object.
        """
        set_inputs_len = len(self._dynamic_shape_inputs)
        inputs_len = len(inputs)
        if set_inputs_len != inputs_len:
            raise ValueError("The number of 'set_input' Tensor must be equal to network's inputs."
                             f"but got 'set_inputs': {set_inputs_len} and network's input: {inputs_len}.")
        for index, (net_input, set_input) in enumerate(zip(inputs, self._dynamic_shape_inputs)):
            if isinstance(set_input, Tensor):
                if not isinstance(net_input, Tensor):
                    raise TypeError(
                        f"The {index + 1}th input type of 'set_inputs' must be Tensor, but got {type(net_input)}.")
                if set_input.dtype is not net_input.dtype:
                    raise ValueError(
                        f"The {index + 1}th input type of 'set_inputs' must be the same as network's input, "
                        f"but got 'set_inputs': {set_input.dtype} and network's input: {net_input.dtype}.")
                if net_input.dim() != 0 and set_input.dim() != net_input.dim():
                    raise ValueError(
                        f"The {index + 1}th input dims of 'set_inputs' must be the same as network's input, "
                        f"but got 'set_inputs': {set_input.dim()} and network's input: {net_input.dim()}.")
                if not all([ele1 in (-1, ele2) for ele1, ele2 in zip(set_input.shape, net_input.shape)]):
                    raise ValueError(
                        f"The {index + 1}th input shape of 'set_inputs' must be the same as network's input, "
                        f"but got 'set_inputs': {set_input.shape} and network's input: {net_input.shape}.")


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
            If the parameter does not exist, ignore it. Default: None.
    Raises:
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
        >>>
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
    def __init__(self, graph, params_init=None):
        super(GraphCell, self).__init__(auto_prefix=True)
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

    def __call__(self, *inputs):
        self.phase = "graph_load_from_mindir"
        self._add_attr("graph_load_from_mindir", self.graph)
        return self.compile_and_run(*inputs)


def _check_param_list_tuple(value):
    """
    Check the type of input in list or tuple is Parameter.
    :param value: list or tuple.
    :return: The types of all inputs are parameter.
    """
    for item in value:
        if not isinstance(item, Parameter):
            return False
    return True
