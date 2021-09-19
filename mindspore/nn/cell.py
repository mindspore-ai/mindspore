# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import gc
import inspect
import os
import time
from collections import OrderedDict

import numpy

from mindspore._checkparam import args_type_check
from mindspore import log as logger
from mindspore.common.parameter import PARAMETER_NAME_DEFAULT
from mindspore.common._decorator import deprecated
from mindspore.context import ParallelMode
from .. import context
from .._c_expression import init_pipeline, Cell_, FuncGraph
from .._checkparam import Validator
from ..common import dtype as mstype
from ..common.api import _cell_graph_executor, _pynative_executor
from ..common.parameter import Parameter, ParameterTuple
from ..common.tensor import Tensor
from ..ops.operations import HookBackward, Cast
from ..ops.primitive import Primitive
from ..parallel._tensor import _load_tensor_by_layout


class Cell(Cell_):
    """
    Base class for all neural networks.

    A 'Cell' could be a single neural network cell, such as conv2d, relu, batch_norm, etc. or a composition of
    cells to constructing a network.

    Note:
        In general, the autograd algorithm will automatically generate the implementation of the gradient function,
        but if back-propagation(bprop) method is implemented, the gradient function will be replaced by the bprop.
        The bprop implementation will receive a tensor `dout` containing the gradient of the loss w.r.t.
        the output, and a tensor `out` containing the forward result. The bprop needs to compute the
        gradient of the loss w.r.t. the inputs, gradient of the loss w.r.t. Parameter variables are not supported
        currently. The bprop method must contain the self parameter.

    Args:
        auto_prefix (bool): Recursively generate namespaces. Default: True.
        flags (dict): Network configuration information, currently it is used for the binding of network and dataset.
                      Users can also customize network attributes by this parameter. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> class MyCell(nn.Cell):
        ...    def __init__(self):
        ...        super(MyCell, self).__init__()
        ...        self.relu = ops.ReLU()
        ...
        ...    def construct(self, x):
        ...        return self.relu(x)
    """
    IGNORE_LIST = ['_scope', '_cell_init_args', '_auto_prefix', '_cells', '_params', '_construct_inputs_names',
                   '_construct_inputs_num', '_create_time', '_mindspore_flags', '_parallel_inputs_run',
                   '_parameter_layout_dict', '_params_list', '_tensor_list', '_phase',
                   '_auto_parallel_mode', '_backward_hook', '_bprop_debug', '_is_run', '_param_prefix',
                   '_attr_synced', 'enable_hook', 'pynative', 'requires_grad',
                   '_auto_parallel_compile_and_run', 'cell_type']

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
        self.parameter_broadcast_done = False
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
        self._backward_hook = None
        self.enable_hook = False
        self._bprop_debug = False
        self.cell_type = None
        self._auto_parallel_compile_and_run = False
        self.cast = Cast()
        self._has_config_recompute = False

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
            raise TypeError("The 'bprop debug' value must be a bool type.")
        self._bprop_debug = value

    def update_cell_prefix(self):
        """
        Update the all child cells' self.param_prefix.

        After being invoked, it can get all the cell's children's name prefix by '_param_prefix'.
        """
        cells_name = self.cells_and_names()

        for cell_name, cell in cells_name:
            cell._param_prefix = cell_name

    def update_cell_type(self, cell_type):
        """
        The current cell type is updated when a quantization aware training network is encountered.

        After being invoked, it can set the cell type to 'cell_type'.
        """
        self.cell_type = cell_type

    @cell_init_args.setter
    def cell_init_args(self, value):
        if not isinstance(value, str):
            raise TypeError("The 'cell_init_args' must be a string type.")
        self._cell_init_args = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, str):
            raise TypeError("The 'phase' must be a string type.")
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
            raise TypeError("The 'parameter_layout_dict' must be a dict type.")
        self._parameter_layout_dict = value

    @property
    def parallel_parameter_name_list(self):
        return self._parallel_parameter_name_list

    @parallel_parameter_name_list.setter
    def parallel_parameter_name_list(self, value):
        if not isinstance(value, list):
            raise TypeError("The 'parallel_parameter_name_list' must be a list type.")
        self._parallel_parameter_name_list = value

    @property
    def pipeline_stage(self):
        return self._pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        if isinstance(value, bool):
            raise TypeError("'pipeline_stage' must be int type, but got bool.")
        if not isinstance(value, int):
            raise TypeError("'pipeline_stage' must be int type.")
        if value < 0:
            raise TypeError("'pipeline_stage' can not less than 0.")
        self._pipeline_stage = value
        for item in self.trainable_params():
            item.add_pipeline_stage(value)

    @property
    def parallel_parameter_merge_net_dict(self):
        return self._parallel_parameter_merge_net_dict

    @parallel_parameter_merge_net_dict.setter
    def parallel_parameter_merge_net_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError("The 'parallel_parameter_merge_net_dict' must be a dict type.")
        self._parallel_parameter_merge_net_dict = value

    def get_func_graph_proto(self):
        """Return graph binary proto."""
        exec_id = self.phase + "." + str(self.create_time) + '.' + str(id(self))
        return _cell_graph_executor._get_func_graph_proto(self, exec_id, "anf_ir", True)

    def __getattr__(self, name):
        if '_params' in self.__dict__:
            params = self.__dict__['_params']
            if name in params:
                if context.get_context("mode") == context.PYNATIVE_MODE:
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
        if context.get_context is not None and context.get_context("mode") == context.PYNATIVE_MODE:
            _pynative_executor.del_cell(str(id(self)))
        if hasattr(self, "_create_time"):
            _cell_graph_executor.del_net_res(str(self._create_time))

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
        if self.enable_hook:
            output = self._hook_construct(*cast_inputs)
        else:
            output = self.construct(*cast_inputs, **kwargs)
        return output

    def _check_construct_args(self, *inputs, **kwargs):
        """Check the args needed by the function construct"""
        if kwargs:
            raise ValueError("For 'graph' mode, the outermost network does not support passing "
                             "variable key-value pair parameters.")
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
            raise TypeError(
                f"The function construct needs {positional_args} positional argument, but only provided {len(inputs)}.")

        if len(inputs) > positional_args + default_args:
            raise TypeError(
                f"The function construct needs {positional_args} positional argument and {default_args} default "
                f"argument, but provided {len(inputs)}")

    class CellGuard:
        def __enter__(self):
            _pynative_executor.set_lazy_build(True)
            _pynative_executor.enter_cell()

        def __exit__(self, exc_type, exc_val, exc_tb):
            _pynative_executor.exit_cell()
            if _pynative_executor.is_top_cell():
                _pynative_executor.set_lazy_build(False)

    def __call__(self, *inputs, **kwargs):
        if self.__class__.construct is Cell.construct:
            logger.warning(f"The '{self.__class__}' does not override the method 'construct', "
                           f"will call the super class(Cell) 'construct'.")
        if kwargs:
            bound_args = inspect.signature(self.construct).bind(*inputs, **kwargs)
            inputs = bound_args.args
            kwargs = bound_args.kwargs

        # Run in Graph mode.
        if context.get_context("mode") == context.GRAPH_MODE:
            self._check_construct_args(*inputs, **kwargs)
            if self.enable_hook:
                raise ValueError("The graph mode does not support hook function.")
            out = self.compile_and_run(*inputs)
            return out

        # Run in PyNative mode.
        if _pynative_executor.is_top_cell():
            _pynative_executor.set_lazy_build(True)
            # There many Casts in parameter_broadcast. Enable lazy_build and build faster.
            self._do_parameter_broadcast()

        for item in inputs:
            if isinstance(item, numpy.ndarray):
                raise TypeError("The cell inputs should not be numpy arrays.")
        if self.requires_grad is True:
            _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(self, *inputs, **kwargs)
        cast_inputs = list()
        if hasattr(self, "_mindspore_flags"):
            if self._mindspore_flags.get('fp16'):
                cast_inputs = self._cast_mixed_precision_inputs(inputs, mstype.float16)
            if self._mindspore_flags.get('fp32'):
                cast_inputs = self._cast_mixed_precision_inputs(inputs, mstype.float32)
        if not cast_inputs:
            cast_inputs = inputs

        with self.CellGuard():
            try:
                output = self.run_construct(cast_inputs, kwargs)
            except Exception as err:
                _pynative_executor.clear_res()
                raise err

        if _pynative_executor.is_top_cell():
            _pynative_executor.execute_all_task()

        if isinstance(output, Parameter):
            output = output.data
        _pynative_executor.end_graph(self, output, *inputs, **kwargs)
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
            raise AttributeError("Can not assign params before Cell.__init__() call.")
        if name in self.__dict__:
            if self.__dict__[name] is not None:
                raise TypeError("The type of value should not be Parameter or Cell, but got Parameter.")
            del self.__dict__[name]
        if cells and name in cells:
            raise TypeError("The type of value should be Cell, but got Parameter.")
        self.insert_param_to_cell(name, value)

    def _set_attr_for_parameter_tuple(self, name, value):
        """Set attr for parameter tuple."""
        params = self.__dict__.get('_params')
        params_list = self.__dict__.get('_params_list')
        if params is None:
            raise AttributeError("Can not assign params before Cell.__init__() call.")
        for item in value:
            self.insert_param_to_cell(item.name, item, check_name=False)
        if context.get_context("mode") == context.PYNATIVE_MODE:
            if name in self.__dict__:
                del self.__dict__[name]
            if name in params:
                del params[name]
            params_list[name] = value
        else:
            object.__setattr__(self, name, value)

    def _set_attr_for_cell(self, name, value):
        """Set attr for cell."""
        cells = self.__dict__.get('_cells')
        params = self.__dict__.get('_params')
        if cells is None:
            raise AttributeError("Can not assign cells before Cell.__init__() call.")
        if name in self.__dict__:
            del self.__dict__[name]
        if params and name in params:
            raise TypeError("The type of value should be Parameter, but got Cell.")
        if self._auto_prefix:
            value.update_parameters_name(name + '.')
        cells[name] = value
        if hasattr(self, '_cell_init_args'):
            self.cell_init_args += str({name: value})

    def __setattr__(self, name, value):
        cells = self.__dict__.get('_cells')
        params = self.__dict__.get('_params')
        tensor_list = self.__dict__.get('_tensor_list')
        if isinstance(value, Parameter):
            self._set_attr_for_parameter(name, value)
        elif isinstance(value, ParameterTuple):
            self._set_attr_for_parameter_tuple(name, value)
        elif isinstance(value, Cell):
            self._set_attr_for_cell(name, value)
        elif params and name in params:
            if isinstance(value, Tensor) and self._params[name] is not None:
                self._params[name].set_data(value)
            elif value is not None:
                raise TypeError(f"The type of value should be Parameter or ParameterTuple, "
                                f"but got {type(value).__name__}.")
            else:
                self.insert_param_to_cell(name, None)
        elif cells and name in cells:
            if value is not None:
                raise TypeError(f"The type of value should be cell, but got {type(value).__name__}.")
            self._cells[name] = None
        elif isinstance(value, Tensor):
            if context.get_context("mode") == context.PYNATIVE_MODE:
                if name in self.__dict__:
                    del self.__dict__[name]
                tensor_list[name] = value
            else:
                object.__setattr__(self, name, value)
        else:
            if isinstance(value, Primitive):
                value.set_prim_instance_name(name)
                self._primitives[name] = value
            object.__setattr__(self, name, value)
        if name not in Cell.IGNORE_LIST:
            self._attr_synced = False

    def extend_repr(self):
        """
        Sets the extended representation of the Cell.

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
            raise TypeError("Parameters need OrderedDict type, but got {}.".format(type(params)))

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
        Slice inputs tensors by parallel strategies, and set the sliced inputs to `_parallel_input_run`

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
            raise ValueError(f"Num of inputs must be greater than 0, but got {self._construct_inputs_num}")
        if self._construct_inputs_names[0] != 'self':
            raise ValueError(f"First member of fn function must be self, but got {self._construct_inputs_names[0]}")
        if self._construct_inputs_num - 1 > len(self._construct_inputs_names):
            raise ValueError(f"Num of inputs must be greater than num of fn function members, num of inputs is \
                {self._construct_inputs_names - 1}, num of fn function members is {len(self._construct_inputs_names)}")
        self._construct_inputs_names = self._construct_inputs_names[1:self._construct_inputs_num]
        self._construct_inputs_num = self._construct_inputs_num - 1

    def compile(self, *inputs):
        """
        Compiles cell.

        Args:
            inputs (tuple): Inputs of the Cell object.
        """
        _cell_graph_executor.compile(self, *inputs, phase=self.phase, auto_parallel_mode=self._auto_parallel_mode)

    def compile_and_run(self, *inputs):
        """
        Compiles and runs cell.

        Args:
            inputs (tuple): Inputs of the Cell object.

        Returns:
            Object, the result of executing.
        """
        self._auto_parallel_compile_and_run = True
        self.compile(*inputs)

        new_inputs = []
        for i in inputs:
            if isinstance(i, Tensor):
                new_inputs.append(i)
            elif context.get_context("grad_for_scalar") and isinstance(i, (int, float)):
                new_inputs.append(i)

        if self._auto_parallel_mode:
            if new_inputs and isinstance(new_inputs[0], Tensor) and inputs[0].virtual_flag:
                # get parallel inputs in sink mode, parallel inputs set in _cell_graph_executor.compile
                parallel_inputs_run = self._parallel_inputs_run
            else:
                parallel_inputs_run = new_inputs
            return _cell_graph_executor(self, *parallel_inputs_run, phase=self.phase)
        return _cell_graph_executor(self, *new_inputs, phase=self.phase)

    def auto_parallel_compile_and_run(self):
        """
        Whether or not to execute compile and run.

        Returns:
            bool, `_auto_parallel_compile_and_run` value.
        """
        return self._auto_parallel_compile_and_run

    def exec_checkpoint_graph(self):
        """Executes saving checkpoint graph operation."""
        _cell_graph_executor(self, phase='save')

    def insert_param_to_cell(self, param_name, param, check_name=True):
        """
        Adds a parameter to the current cell.

        Inserts a parameter with given name to the cell. Please refer to the usage in
        source code of `mindspore.nn.Cell.__setattr__`.

        Args:
            param_name (str): Name of the parameter.
            param (Parameter): Parameter to be inserted to the cell.
            check_name (bool): Determines whether the name input is compatible. Default: True.

        Raises:
            KeyError: If the name of parameter is null or contains dot.
            AttributeError: If user did not call init() first.
            TypeError: If the type of parameter is not Parameter.
        """
        if not param_name:
            raise KeyError("The name of parameter should not be null.")
        if check_name and '.' in param_name:
            raise KeyError("The name of parameter should not contain \".\"")
        if '_params' not in self.__dict__:
            raise AttributeError("You need call init() first.")
        if hasattr(self, param_name) and param_name not in self._params:
            raise KeyError("Duplicated parameter name '{}'.".format(param_name))
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError("The type of parameter should be 'Parameter' if not None.")
        if isinstance(param, Parameter) and param.name == PARAMETER_NAME_DEFAULT:
            param.name = param_name
        self._params[param_name] = param

    def cast_param(self, param):
        """
        Cast parameter according to auto mix precision level in pynative mode.

        This interface is currently used in the case of auto mix precision and usually need not to be used explicitly.

        Args:
            param (Parameter): Parameters, the type of which should be cast.

        Returns:
            Parameter, the input parameter with type automatically cast.
        """
        if hasattr(self, "_mindspore_flags"):
            if self._mindspore_flags.get('fp32'):
                param.set_cast_dtype(mstype.float32)
            elif self._mindspore_flags.get('fp16'):
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
            raise KeyError("Child cell name is incorrect.")
        if hasattr(self, child_name) and child_name not in self._cells:
            raise KeyError("Duplicate child name '{}'.".format(child_name))
        if not isinstance(child_cell, Cell) and child_cell is not None:
            raise TypeError("Child cell type is incorrect.")
        self._cells[child_name] = child_cell

    def construct(self, *inputs, **kwargs):
        """
        Defines the computation to be performed. This method must be overridden by all subclasses.

        Returns:
            Tensor, returns the computed result.
        """
        return None

    def remove_redundant_parameters(self):
        """
        Remove the redundant parameters.

        This interface usually need not to be used explicitly.
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
            `init_parameters_data`, do not save these result.

        Args:
            auto_parallel_mode (bool): If running in auto_parallel_mode.

        Returns:
            Dict[Parameter, Parameter], returns a dict of original parameter and replaced parameter.
        """
        replace = dict()

        def _updata(param):
            if param in replace:
                return replace[param]
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
        Gets parameters dictionary.

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
        Updates the names of parameters with given prefix string.

        Adds the given prefix to the names of parameters.

        Args:
            prefix (str): The prefix string. Default: ''.
            recurse (bool): Whether contains the parameters of subcells. Default: True.
        """

        Validator.check_str_by_regular(prefix)
        for name, param in self.parameters_and_names(expand=recurse):
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
            >>> net = Net()
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
                raise ValueError("The value of {} is {}, its name '{}' already exists.".
                                 format(value, param, param.name))
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
            >>> n = Net()
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
        Returns an iterator over all cells in the network.

        Includes the cell's name and itself.

        Args:
            cells (str): Cells to iterate over. Default: None.
            name_prefix (str): Namespace. Default: ''.

        Returns:
            Iteration, all the child cells and corresponding names in the cell.

        Examples:
            >>> n = Net()
            >>> names = []
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
            Iteration, all the child cells in the cell.
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
            yield parent_prefix + "/" + name + (("-" + cell.__class__.__name__)
                                                if reserve_class_name_in_scope else ""), cell

        for name, cell in self.name_cells().items():
            for key, value in cell._children_scope_recursive(parent_prefix + "/" + name +
                                                             (("-" + cell.__class__.__name__)
                                                              if reserve_class_name_in_scope else "")):
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
        Returns an iterator over all cells in the network.

        Include name of the cell and cell itself.

        Returns:
            Dict[String, Cell], all the child cells and corresponding names in the cell.
        """
        value_set = set()
        cells = OrderedDict()
        for name, cell in self._cells.items():
            if cell is not None and cell not in value_set:
                value_set.add(cell)
                cells[name] = cell
        return cells

    def add_flags(self, **flags):
        """
        Add customized attributes for cell.

        This method is also called when the cell class is instantiated and the class parameter 'flag' is set to True.
        """
        if not hasattr(self, "_mindspore_flags"):
            self._mindspore_flags = {}
        self._mindspore_flags.update({**flags})
        self.__dict__.update({**flags})
        return self

    def add_flags_recursive(self, **flags):
        """
        If a cell contains child cells, this method can recursively customize attributes of all cells.
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
        Get the attributes of cell's flags.
        """
        if not hasattr(self, "_mindspore_flags"):
            self._mindspore_flags = {}
        return self._mindspore_flags

    def to_float(self, dst_type):
        """
        Add cast on all inputs of cell and child cells to run with certain float type.

        If `dst_type is mindspore.dtype.float16`, all the inputs of Cell including input, Parameter, Tensor
        as const will be cast to float16. Please refer to the usage in source code of
        `mindspore.train.amp.build_train_network`.

        Note:
            Multiple calls will overwrite.

        Args:
            dst_type (:class:`mindspore.dtype`): Transfer cell to run with dst_type.
                dst_type can be `mindspore.dtype.float16` or `mindspore.dtype.float32`.

        Returns:
            Cell, the cell itself.

        Raises:
            ValueError: If dst_type is not float32 or float16.
        """
        if dst_type not in (mstype.float16, mstype.float32):
            raise ValueError("The dst_type should inside float32 or float16.")
        flags = {'fp16': dst_type == mstype.float16, 'fp32': dst_type == mstype.float32}
        self.add_flags_recursive(**flags)
        self._add_init_args(**flags)
        return self

    def set_boost(self, boost_type):
        """
        In order to improve the network performance, configure the network auto enable to
        accelerate the algorithm in the algorithm library.

        If `boost_type is not in the algorithm library`, Please view the algorithm in the algorithm library
        through `algorithm library`.

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
            raise ValueError("The boost_type is not in the algorithm library.")
        flags = {"less_bn": boost_type == "less_bn"}
        self.add_flags_recursive(**flags)
        return self

    def set_grad(self, requires_grad=True):
        """
        Sets the cell flag for gradient. In pynative mode, this parameter specifies whether the network require
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
        Set the cell to data_parallel mode.

        The cell can be accessed as an attribute using the given name.

        Args:
            mode (bool): Specifies whether the model is data_parallel. Default: True.
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
        self.add_flags(auto_parallel=True)
        self._get_construct_inputs_number_and_name()

    def _hook_construct(self, *inputs):
        """Hook construct method to replace original construct method when hook function enabled."""
        inputs = self._backward_hook(*inputs)
        inputs = self.construct(inputs)
        outputs = self._backward_hook(inputs)
        return outputs

    def register_backward_hook(self, fn):
        """
        Set the cell backward hook function. Note that this function is only supported in pynative mode.

        Note:
            fn must be defined as the following code. `cell_name` is the name of registered cell.
            `grad_input` is gradient passed to the cell. `grad_output` is the gradient computed and passed to the
            next cell or primitive, which may be modified and returned.
            hook_fn(cell_name, grad_input, grad_output) -> Tensor or None.

        Args:
            fn (function): Specifies the hook function with grad as input.

        """
        self._backward_hook = HookBackward(fn, self.cls_name + "(" + str(id(self)) + ")")
        self.enable_hook = True

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
        Set `comm_fusion` for all the parameters in the Net. Please refer to the description of
        `mindspore.common.parameter.comm_fusion`.

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
        if mode is True:
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

    def _recompute(self, mode=True, output_recompute=False):
        """
        Set the cell recomputed.
        """
        if context.get_context("mode") == context.PYNATIVE_MODE:
            raise TypeError("Recompute is not supported in pynative mode currently.")
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
        Set the cell recomputed. All the primitive in the cell will be set recomputed. If a primitive
        set recomputed feeds into some backward nodes for computing gradient, rather than storing the
        intermediate activation computed in forward pass, we will recompute it in backward pass.

        Note:

            - If the computation involves something like randomization or global variable, the equivalence
              is not guaranteed currently.
            - If the recompute api of a primitive in this cell is also called, the recompute mode of this
              primitive is subject to the recompute api of the primitive.
            - The interface can be configured only once.
              Therefore, when the parent cell is configured, the child cell should not be configured.
            - When the memory remains after applying the recompute, configuring 'mp_comm_recompute=False'
              to improve performance if necessary.
            - When the memory still not enough after applying the recompute, configuring
              'parallel_optimizer_comm_recompute=True' to save more memory if necessary.
              Cells in the same fusion group should has the same parallel_optimizer_comm_recompute configures.

        Args:
            mp_comm_recompute (bool): Specifies whether the model parallel communication operators
                in the cell are recomputed in auto parallel or semi auto parallel mode. Default: True.
            parallel_optimizer_comm_recompute (bool): Specifies whether the communication operator allgathers
                introduced by optimizer shard are recomputed in auto parallel or semi auto parallel mode.
                Default: False.
        """
        self._recompute()
        if 'mp_comm_recompute' in kwargs.keys():
            self._mp_comm_recompute(kwargs['mp_comm_recompute'])
        if 'parallel_optimizer_comm_recompute' in kwargs.keys():
            if kwargs['parallel_optimizer_comm_recompute'] and context.get_auto_parallel_context("pipeline_stages") > 1:
                raise ValueError("Currently, the communication operator allgathers introduced by optimizer shard "
                                 "are not support recomputation in pipeline parallel.")
            self._parallel_optimizer_comm_recompute(kwargs['parallel_optimizer_comm_recompute'])

        for key, _ in kwargs.items():
            if key not in ('mp_comm_recompute', 'parallel_optimizer_comm_recompute'):
                raise ValueError("Recompute keyword %s is not recognized!" % key)

    def infer_param_pipeline_stage(self):
        """
        Infer pipeline stages of all parameters in the cell.

        Note:
            - If a parameter does not belong to any cell which has been set pipeline_stage,
              the parameter should use add_pipeline_stage to add it's pipeline_stage information.
            - If a parameter P has been used by two operator in different stages "stageA" and "stageB",
              the parameter P should use P.add_pipeline_stage(stageA) and P.add_pipeline_stage(stageB)
              to add it's stage information before use infer_param_pipeline_stage.

        Returns:
            The params belong to current stage in pipeline parallel.

        Raises:
            RuntimeError: If there is a parameter does not belong to any stage.
        """
        from mindspore.parallel._utils import _get_global_rank, _get_device_num
        stage_num = context.get_auto_parallel_context("pipeline_stages")
        device_num = _get_device_num()
        rank_id = _get_global_rank()
        per_stage_devices = device_num // stage_num
        current_stage = rank_id // per_stage_devices
        params = []
        for param in self.trainable_params():
            if not param._pipeline_stage_list:
                raise RuntimeError("The parameter {} does not belong to any stage, "
                                   "please check whether the cell where the param locates"
                                   " has been set pipeline_stage. "
                                   "Otherwise, the parameter should use add_pipeline_stage "
                                   "to add its stage information".format(param.name))
            if current_stage in param._pipeline_stage_list:
                params.append(param)
        return params


class GraphKernel(Cell):
    """
    Base class for GraphKernel.

    A `GraphKernel` a composite of basic primitives and can be compiled into a fused kernel automatically when
    enable_graph_kernel in context is set to True.

    This class is deprecated from version 1.3 and will be removed in a future version, use Cell instead.

    GraphKernel is not supported user-defined cells anymore, the `GraphKernel` objects will be treated as
    normal `Cell` objects.

    Args:
        auto_prefix (bool): Recursively generate namespaces. Default: True.
        flags (dict) : Set graph flags. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Relu(nn.GraphKernel):
        ...    def __init__(self):
        ...        super(Relu, self).__init__()
        ...        self.max = P.Maximum()
        ...
        ...    def construct(self, x):
        ...        return self.max(P.Fill()(P.DType()(x), P.Shape()(x), 0.0), x)
    """

    @deprecated("1.3", "Cell", True)
    def __init__(self, auto_prefix=True, flags=None):
        super(GraphKernel, self).__init__(auto_prefix, flags)

    def construct(self):
        raise NotImplementedError


class GraphCell(Cell):
    """
    Base class for running the graph loaded from a MindIR.

    This feature is still under development. Currently `GraphCell` do not support modifying the structure of the
    diagram, and can only use data that shape and type are the same as the input when exporting the MindIR.

    Args:
        graph (object): A compiled graph loaded from MindIR.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.nn as nn
        >>> from mindspore import Tensor, export, load
        >>>
        >>> net = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")
        >>> input = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
        >>> export(net, input, file_name="net", file_format="MINDIR")
        >>> graph = load("net.mindir")
        >>> net = nn.GraphCell(graph)
        >>> output = net(input)
        >>> print(output)
        [[[[4. 6. 4.]
           [6. 9. 6.]
           [4. 6. 4.]]]]
    """
    def __init__(self, graph):
        super(GraphCell, self).__init__(auto_prefix=True)
        if not isinstance(graph, FuncGraph):
            raise TypeError(f"graph must be a FuncGraph loaded from MindIR, but got {type(graph)}.")
        self.graph = graph

    def construct(self, *inputs):
        return self.graph(*inputs)

    def __call__(self, *inputs):
        self.phase = "graph_load_from_mindir"
        self._add_attr("graph_load_from_mindir", self.graph)
        return self.compile_and_run(*inputs)
