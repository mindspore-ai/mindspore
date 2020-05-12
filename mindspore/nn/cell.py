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
"""cell"""
import time
import gc
from collections import OrderedDict
from mindspore import log as logger
from .. import context
from ..common import dtype as mstype
from ..common.api import _executor
from .._checkparam import _check_str_by_regular
from ..common.parameter import Parameter, ParameterTuple
from .._c_expression import init_backend
from ..ops.primitive import Primitive
from ..parallel._tensor import _load_tensor_by_layout
from ..parallel._utils import _get_parallel_mode
from ..common.tensor import Tensor


class Cell:
    """
    Base class for all neural network.

    A 'Cell' could be a single neural network cell, such as conv2d, relu, batch_norm, etc. or a composition of
    cells to constructing a network.

    Note:
        In general, the autograd algorithm will automatically generate the implementation of the gradient function,
        but if bprop method is implemented, the gradient function
        will be replaced by the bprop. The bprop implementation will receive a Tensor `dout` containing the gradient
        of the loss w.r.t. the output, and a Tensor `out` containing the forward result. The bprop need to compute the
        gradient of the loss w.r.t. the inputs, gradient of the loss w.r.t. Parameter variables is not supported
        currently.

    Args:
        auto_prefix (bool): Recursively generate namespaces. Default: True.

    Examples:
        >>> class MyCell(Cell):
        >>>    def __init__(self):
        >>>        super(MyCell, self).__init__()
        >>>        self.relu = P.ReLU()
        >>>
        >>>    def construct(self, x):
        >>>        return self.relu(x)
    """
    def __init__(self, auto_prefix=True, flags=None):
        self._params = OrderedDict()
        self._cells = OrderedDict()
        self.training = False
        self.pynative = False
        self._param_perfix = ''
        self._auto_prefix = auto_prefix
        self._scope = None
        self._phase = 'train'
        self._parameter_layout_dict = {}
        self._create_time = int(time.time() * 1e9)
        init_backend()
        # call gc to release GE session resources used by non-used cell objects
        gc.collect()
        self._construct_inputs_num = 0
        self._construct_inputs_names = []
        if _get_parallel_mode() in ["auto_parallel", "semi_auto_parallel"]:
            self._get_construct_inputs_number_and_name()
        self._parallel_inputs_run = None
        if flags:
            self.add_flags(**flags)

    @property
    def create_time(self):
        return self._create_time

    @property
    def cell_init_args(self):
        return self._cell_init_args

    @property
    def param_perfix(self):
        """
        Param perfix is the prfix of curent cell's direct child parameter.
        """
        return self._param_perfix

    def update_cell_prefix(self):
        """
        Update the all child cells' self.param_prefix.

        After invoked, can get all the cell's children's name perfix by '_param_perfix'.
        """
        cells = self.cells_and_names

        for cell_name, cell in cells:
            cell._param_perfix = cell_name

    @cell_init_args.setter
    def cell_init_args(self, value):
        if not isinstance(value, str):
            raise TypeError("'cell_init_args' must be string type.")
        self._cell_init_args = value

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        if not isinstance(value, str):
            raise TypeError("'phase' must be string type.")
        self._phase = value

    @property
    def parameter_layout_dict(self):
        return self._parameter_layout_dict

    @property
    def cls_name(self):
        return self.__class__.__name__

    @parameter_layout_dict.setter
    def parameter_layout_dict(self, value):
        if not isinstance(value, dict):
            raise TypeError("'parameter_layout_dict' must be dict type.")
        self._parameter_layout_dict = value

    def get_func_graph_proto(self):
        """Return graph binary proto."""
        return _executor._get_func_graph_proto(self.phase + "." + str(self.create_time), "anf_ir", True)

    def __getattr__(self, name):
        if '_params' in self.__dict__:
            params = self.__dict__['_params']
            if name in params:
                return params[name]
        if '_cells' in self.__dict__:
            cells = self.__dict__['_cells']
            if name in cells:
                return cells[name]
        raise AttributeError("'{}' object has no attribute '{}'.".format(type(self).__name__, name))

    def __del__(self):
        if hasattr(self, "_create_time"):
            _executor.del_net_res(str(self._create_time))

    def __delattr__(self, name):
        if name in self._params:
            del self._params[name]
        elif name in self._cells:
            del self._cells[name]
        else:
            object.__delattr__(self, name)

    def __call__(self, *inputs):
        if context.get_context("mode") == context.GRAPH_MODE:
            out = self.compile_and_run(*inputs)
            return out
        output = self.construct(*inputs)
        if isinstance(output, Parameter):
            output = output.data
        return output

    def __setattr__(self, name, value):
        cells = self.__dict__.get('_cells')
        params = self.__dict__.get('_params')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("Can not assign params before Cell.__init__() call.")
            if name in self.__dict__:
                if self.__dict__[name] is not None:
                    raise TypeError("Expected type is not in (Parameter, Cell), but got Parameter.")
                del self.__dict__[name]
            if cells and name in cells:
                raise TypeError("Expected type is Cell, but got Parameter.")
            self.insert_param_to_cell(name, value)
        elif isinstance(value, ParameterTuple):
            if params is None:
                raise AttributeError("Can not assign params before Cell.__init__() call.")
            for item in value:
                self.insert_param_to_cell(item.name, item, check_name=False)
            object.__setattr__(self, name, value)
        elif isinstance(value, Cell):
            if cells is None:
                raise AttributeError("Can not assign cells before Cell.__init__() call.")
            if name in self.__dict__:
                del self.__dict__[name]
            if params and name in params:
                raise TypeError("Expected type is Parameter, but got Cell.")
            if self._auto_prefix:
                value.update_parameters_name(name + '.')
            cells[name] = value
        elif params and name in params:
            if value is not None:
                raise TypeError("Expected type in (Parameter, ParameterTuple), but got {}.".format(type(value)))
            self.insert_param_to_cell(name, None)
        elif cells and name in cells:
            if value is not None:
                raise TypeError("Expected type is cell, but got {}.".format(type(value)))
            self._cells[name] = None
        else:
            if isinstance(value, Primitive):
                value.set_prim_instance_name(name)
            object.__setattr__(self, name, value)

    def extend_repr(self):
        """
        Sets the extended representation of the Cell.

        To print customized extended information, re-implement this method in your own cells.
        """
        return ''

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

        Please refer to the usage in source code of `mindspore.common._Executor.compile`.

        Args:
            params (dict): The parameters dictionary used for init data graph.
        """
        if params is None:
            for key in self.parameters_dict():
                tensor = self.parameters_dict()[key].data
                if key not in self.parameter_layout_dict:
                    logger.info("layout dict does not contain the key %s", key)
                    continue
                layout = self.parameter_layout_dict[key]
                new_tensor = _load_tensor_by_layout(tensor, layout)
                self.parameters_dict()[key].set_parameter_data(new_tensor)
        elif isinstance(params, OrderedDict):
            for key in params:
                tensor = params[key].data
                if key not in self.parameter_layout_dict:
                    logger.info("layout dict does not contain the key %s", key)
                    continue
                layout = self.parameter_layout_dict[key]
                new_tensor = _load_tensor_by_layout(tensor, layout)
                params[key].set_parameter_data(new_tensor)
        else:
            raise TypeError('Parameters need OrderedDict type, but got {}'.
                            format(type(params)))

    def _load_inputs(self, *inputs):
        """
        Slice inputs tensors by parallel strategies.

        Args:
            inputs (Function or Cell): inputs of construct method.
        """
        parallel_inputs_run = []
        if len(inputs) > self._construct_inputs_num:
            raise ValueError('Len of inputs: {} is bigger than self._construct_inputs_num: {}.'.
                             format(len(inputs), self._construct_inputs_num))
        for i, tensor in enumerate(inputs):
            key = self._construct_inputs_names[i]
            # if input is not used, self.parameter_layout_dict may not contain the key
            if key not in self.parameter_layout_dict:
                logger.warning("layout dict does not contain the key %s", key)
                parallel_inputs_run.append(tensor)
            else:
                layout = self.parameter_layout_dict[key]
                new_tensor = _load_tensor_by_layout(tensor, layout)
                parallel_inputs_run.append(new_tensor)
        return tuple(parallel_inputs_run)

    def _get_construct_inputs_number_and_name(self):
        """Compute self._construct_inputs_names and self._construct_inputs_num"""
        import inspect
        from mindspore._extends.parse.parser import get_parse_method_of_class

        fn = get_parse_method_of_class(self)
        inspect.getfullargspec(fn)
        self._construct_inputs_num = fn.__code__.co_argcount
        self._construct_inputs_names = fn.__code__.co_varnames

        assert self._construct_inputs_num > 0
        assert self._construct_inputs_names[0] == 'self'
        assert self._construct_inputs_num - 1 <= len(self._construct_inputs_names)
        self._construct_inputs_names = self._construct_inputs_names[1:self._construct_inputs_num]
        self._construct_inputs_num = self._construct_inputs_num - 1

    def compile_and_run(self, *inputs):
        """
        Compiles and runs cell.

        Args:
            inputs (tuple): Input parameters.

        Returns:
            Object, the result of executing.
        """
        _, compile_flag = _executor.compile(self, *inputs, phase=self.phase)

        if _get_parallel_mode() in ["auto_parallel", "semi_auto_parallel"]:
            if inputs and isinstance(inputs[0], Tensor) and inputs[0].virtual_flag and (not compile_flag):
                parallel_inputs_run = self._parallel_inputs_run
            else:
                self._parallel_inputs_run = self._load_inputs(*inputs)
                parallel_inputs_run = self._parallel_inputs_run
            return _executor(self, *parallel_inputs_run, phase=self.phase)
        return _executor(self, *inputs, phase=self.phase)

    def exec_checkpoint_graph(self):
        """Executes saving checkpoint graph operation."""
        _executor(self, phase='save')

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
        self._params[param_name] = param

    def insert_child_to_cell(self, child_name, child):
        """
        Adds a child cell to the current cell.

        Inserts a subcell with given name to current cell.

        Args:
            child_name (str): Name of the child cell.
            child (Cell): The child cell to be inserted.

        Raises:
            KeyError: Child Cell's name is incorrect or duplicated with the other child name.
            TypeError: Child Cell's type is incorrect.
        """
        if not child_name or '.' in child_name:
            raise KeyError("Child cell name is incorrect.")
        if hasattr(self, child_name) and child_name not in self._cells:
            raise KeyError("Duplicate child name '{}'.".format(child_name))
        if not isinstance(child, Cell) and child is not None:
            raise TypeError("Child cell type is incorrect.")
        self._cells[child_name] = child

    def construct(self, *inputs):
        """
        Defines the computation to be performed.

        This method should be overridden by all subclasses.

        Note:
            The inputs of the top cell only allow Tensor.
            Other types (tuple, list, int etc.) are forbidden.

        Returns:
            Tensor, returns the computed result.
        """
        raise NotImplementedError

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
            prefix (str): The prefix string.
            recurse (bool): Whether contains the parameters of subcells. Default: True.
        """

        _check_str_by_regular(prefix)
        for name, param in self.parameters_and_names(expand=recurse):
            if prefix != '':
                param.is_init = False
            param.name = prefix + name

    def trainable_params(self, recurse=True):
        """
        Returns all trainable parameters.

        Returns a list of all trainable parmeters.

        Args:
            recurse (bool): Whether contains the trainable parameters of subcells. Default: True.

        Returns:
            List, the list of trainable parameters.
        """
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=recurse)))

    def untrainable_params(self, recurse=True):
        """
        Returns all untrainable parameters.

        Returns a list of all untrainable parmeters.

        Args:
            recurse (bool): Whether contains the untrainable parameters of subcells. Default: True.

        Returns:
            List, the list of untrainable parameters.
        """
        return list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=recurse)))

    def get_parameters(self, expand=True):
        """
        Returns an iterator over cell parameters.

        Yields parameters of this cell. If `expand` is True, yield parameters of this cell and all subcells.

        Args:
            expand (bool): If True, yields parameters of this cell and all subcells. Otherwise, yields only parameters
                           that are direct members of this cell. Default: True.

        Examples:
            >>> net = Net()
            >>> for item in net.get_parameters():
            >>>     print(item)
        """
        for _, param in self.parameters_and_names(expand=expand):
            yield param

    def check_names(self):
        names = set("")
        for value, param in self.parameters_and_names():
            if param.name in names:
                raise ValueError("The value of {} is {}, its name '{}' already exists.".
                                 format(value, param, param.name))
            names.add(param.name)

    def parameters_and_names(self, name_prefix='', expand=True):
        """
        Returns an iterator over cell parameters.

        Includes the parameter's name  and itself.

        Args:
            name_prefix (str): Namespace. Default: ''.
            expand (bool): If True, yields parameters of this cell and all subcells. Otherwise, yields only parameters
                           that are direct members of this cell. Default: True.

        Examples:
            >>> n = Net()
            >>> names = []
            >>> for m in n.parameters_and_names():
            >>>     if m[0]:
            >>>         names.append(m[0])
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
                if par and par not in params_set:
                    params_set.add(par)

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

        Examples:
            >>> n = Net()
            >>> names = []
            >>> for m in n.cells_and_names():
            >>>     if m[0]:
            >>>         names.append(m[0])
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
        """Returns an iterator over immediate cells."""
        return self.name_cells().values()

    def _set_scope(self, name):
        """Sets the name on the first time."""
        if self._scope is None:
            self._scope = name

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
        """Returns the scope of a cell object in one network."""
        return self._scope

    def generate_scope(self):
        """Generate the scope for every cell object in the network."""
        for name, cell in self._children_scope_recursive():
            cell._set_scope(name)

    def name_cells(self):
        """
        Returns an iterator over all cells in the network.

        Include name of the cell and cell itself.
        """
        value_set = set()
        cells = OrderedDict()
        for name, cell in self._cells.items():
            if cell is not None and cell not in value_set:
                value_set.add(cell)
                cells[name] = cell
        return cells

    def add_flags(self, **flags):
        for x in flags:
            if not isinstance(flags[x], bool):
                raise TypeError(f"Flags (f{x}) must be bool but {type(flags[x])}.")
        if not hasattr(self, "_mindspore_flags"):
            self._mindspore_flags = {}
        self._mindspore_flags.update({**flags})
        self.__dict__.update({**flags})
        return self

    def add_flags_recursive(self, **flags):
        self.add_flags(**flags)
        if hasattr(self, '_cell_init_args'):
            self._cell_init_args += str({**flags})
        for cell in self.cells():
            cell.add_flags_recursive(**flags)
        return self

    def get_flags(self):
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
            Call multiple times will overwrite the previous.

        Args:
            dst_type (:class:`mindspore.dtype`): Transfer Cell to Run with dst_type.
                dst_type can be `mindspore.dtype.float16` or `mindspore.dtype.float32`.

        Raises:
            ValueError: If dst_type is not float32 or float16.
        """
        if dst_type not in (mstype.float16, mstype.float32):
            raise ValueError("dst_type should inside float32 or float16.")
        flags = {'fp16': dst_type == mstype.float16, 'fp32': dst_type == mstype.float32}
        self.add_flags_recursive(**flags)
        return self

    def set_train(self, mode=True):
        """
        Sets the cell to training mode.

        The cell itself and all children cells will be set to training mode.

        Args:
            mode (bool): Specifies whether the model is training. Default: True.
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
