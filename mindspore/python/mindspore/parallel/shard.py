# Copyright 2023 Huawei Technologies Co., Ltd
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
"""shard"""

import copy
import numpy as np
import mindspore as ms
from mindspore import log as logger
from mindspore._c_expression import Shard_


class _DistributedTensorInfo:
    """
    Describe the distributed information of a tensor.

    Args:
        distributed_info (Union[Layout, DeviceMesh]): The distributed information of a tensor.

    Raises:
        TypeError: If `distributed_info` is not a Layout type.

    Examples:
        >>> from mindspore import _DistributedTensorInfo, Layout
        >>> layout = Layout((2, 2), ("dp", "mp"))
        >>> src_layout = layout("dp", "mp")
        >>> distributed_info = _DistributedTensorInfo(src_layout)
        >>> print(distributed_info.sharding_strategy)
        [2, 2]
    """

    def __init__(self, distributed_info):
        if isinstance(distributed_info, Layout):
            self._layout = distributed_info
            self._distributed_info = distributed_info
        else:
            raise TypeError(
                f"DistributedTensorInfo only supports Layout or DeviceMesh as input, but got {type(distributed_info)}")
        self._sharding_strategy = None

    @property
    def layout(self):
        """return layout of current tensor"""
        return self._layout

    @property
    def distributed_info(self):
        """return the distributed info, it depends on user's input """
        return self._distributed_info

    @property
    def sharding_strategy(self):
        """return the sharding strategy of current tensor"""
        if self._sharding_strategy is None:
            layout_info = self._layout.to_dict()
            device_matrix = layout_info["device_matrix"]
            tensor_map = layout_info["tensor_map"]
            sharding_strategy = []
            for map_value in tensor_map:
                if isinstance(map_value, (tuple, list)):
                    shard_size = 1
                    for value in map_value:
                        if value != -1:
                            shard_size *= device_matrix[len(device_matrix) - value - 1]
                    sharding_strategy.append(shard_size)
                else:
                    if map_value != -1:
                        sharding_strategy.append(device_matrix[len(device_matrix) - map_value - 1])
                    else:
                        sharding_strategy.append(1)
            self._sharding_strategy = sharding_strategy
        return self._sharding_strategy


class Layout:
    """
    Topological abstraction describing cluster devices for tensor slice placement on the cluster.

    Note:
        - It is valid only in semi auto parallel or auto parallel mode.
        - The multiplication result of the `device_matrix` must be equal to the device count in a pipeline stage.
        - When the layout function is invoked to construct a sharding strategy, each alias name is only allowed to be
          used once to shard a tensor.

    Args:
        device_matrix (tuple): Describe the shape of devices arrangement, its element type is int.
        alias_name (tuple): The alias name for each axis of device_matrix, its element type is string.
                            When using "interleaved_parallel" as an alias name, the tensor would be split into multiple
                            copies on the corresponding partition dimension on a single card.
        rank_list (list, optional): Data is allocated to the device according to rank_list. Default: ``None``.

    Raises:
        TypeError: `device_matrix` is not a tuple type.
        TypeError: `alias_name` is not a tuple type.
        TypeError: 'rank_list' is not a list type.
        ValueError: `device_matrix` length is not equal to `alias_name` length.
        TypeError: The element of `device_matrix` is not int type.
        TypeError: The element of `alias_name` is not a str type.
        TypeError: The element of `rank_list` is not int type.
        ValueError: The element of `alias_name` is an empty str.
        ValueError: The element of `alias_name` is "None".
        ValueError: `alias_name` contains repeated element.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.parallel import Layout
        >>> layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
        >>> layout0 = layout("dp", "mp")
        >>> print(layout0.to_dict())
        {"device_matrix": (2, 2, 2), "tensor_map": (2, 0), "interleaved_parallel": False,
        'alias_name': {'dp', 'sp', 'mp'}, "rank_list": [0, 1, 2, 3, 4, 5, 6, 7]}
        >>> layout = Layout((2, 2, 2), ("dp", "sp", "interleaved_parallel"))
        >>> layout1 = layout(("dp", "interleaved_parallel"), "sp")
    """

    def __init__(self, device_matrix, alias_name, rank_list=None):
        if not isinstance(device_matrix, tuple):
            raise TypeError(f'device_matrix must be tuple type, but got:{type(device_matrix)}')
        if not isinstance(alias_name, tuple):
            raise TypeError(f'alias_name must be tuple type, but got:{type(alias_name)}')
        if len(device_matrix) != len(alias_name):
            raise ValueError(f'device_matrix length should be equal to alias_name length')
        for in_ele in device_matrix:
            if not isinstance(in_ele, int):
                raise TypeError(f'The element of device_matrix must be int type, but got:{type(in_ele)}')
        for in_ele in alias_name:
            if not isinstance(in_ele, str):
                raise TypeError(f'The element of alias_name must be str type, but got:{type(in_ele)}')
            if not in_ele:
                raise ValueError(f"The element of alias_name can not be empty.")
            if in_ele == "None":
                raise ValueError(f"The element of alias_name can not set 'None', because 'None' means no sharding.")
        if len(set(alias_name)) != len(alias_name):
            raise ValueError(f'Each element of alias_name {alias_name} should be different')
        inter_key = "interleaved_parallel"
        if inter_key in alias_name and alias_name.index(inter_key) != len(alias_name) - 1:
            raise ValueError(f"When alias_name {alias_name} contains keyword 'interleaved_parallel',"
                             f" it should be at the last dim of alias_name, which means the virtual sharding.")
        self._device_shape = device_matrix
        self._alias_name = alias_name
        self._tensor_map = None
        self._rank_list = list(range(np.prod(np.array(self._device_shape))))
        if rank_list is not None:
            if not isinstance(rank_list, list):
                raise TypeError(f"The rank_list should be a list, but got {type(rank_list).__name__}.")
            for in_ele in rank_list:
                if not isinstance(in_ele, int):
                    raise TypeError(f"The element of rank_list should be int, but got {type(in_ele).__name__}.")
            if len(np.array(rank_list).shape) != 1:
                raise ValueError(
                    f"The rank_list should be a 1-D list, but got {len(np.array(rank_list).shape)}-D list.")
            if len(rank_list) != np.prod(np.array(self._device_shape)):
                raise ValueError(f"The length of rank_list should be equal to the product of device_matrix, "
                                 f"but got {len(rank_list)} and {np.prod(np.array(self._device_shape))}.")
            self._rank_list = rank_list

    def __call__(self, *tensor_map):
        self._tensor_map = ()
        writed_map = ()
        for ele in tensor_map:
            if isinstance(ele, tuple):
                ele_map = ()
                for item in ele:
                    if item == "None":
                        ele_map += (-1,)
                        continue
                    if item not in self._alias_name:
                        raise ValueError(f'The axis {item} is not found in {self._alias_name}')
                    if item in writed_map:
                        raise ValueError(f'The axis {item} has been set more than one in {self._alias_name}')
                    ele_map += (len(self._alias_name) - 1 - self._alias_name.index(item),)
                    writed_map += (item,)
                self._tensor_map += (ele_map,)
                continue
            if ele == "None":
                self._tensor_map += (-1,)
                continue
            if ele not in self._alias_name:
                raise ValueError(f'The axis {ele} is not found in {self._alias_name}')
            if ele in writed_map:
                raise ValueError(f'The axis {ele} has been set more than one in {self._alias_name}')
            self._tensor_map += (len(self._alias_name) - 1 - self._alias_name.index(ele),)
            writed_map += (ele,)
        return copy.deepcopy(self)

    def to_dict(self):
        """
        Transform layout to a dictionary.
        """
        if self._device_shape is None:
            raise ValueError("The device_shape of layout is None")
        if self._tensor_map is None:
            raise ValueError("The tensor_map of layout is None")
        interleaved_parallel = "interleaved_parallel" in self._alias_name
        return {"device_matrix": self._device_shape, "tensor_map": self._tensor_map,
                "interleaved_parallel": interleaved_parallel, "alias_name": self._alias_name,
                "rank_list": self._rank_list}


class Shard(Shard_):
    """Shard operation"""

    def __init__(self):
        """Initialize Shard."""
        super().__init__('Shard')
        self.shard_fn = None
        self.fn = None
        self.in_strategy = None
        self.out_strategy = None
        self.parameter_plan = None
        self.device = None
        self.level = None

    def __call__(self, fn, in_strategy, out_strategy=None, parameter_plan=None, device="Ascend", level=0):
        if not isinstance(in_strategy, tuple):
            raise TypeError(
                f"For 'Shard', the 'in_strategy' should be a tuple, but got {type(in_strategy).__name__}.")
        inner_type = self._check_layout_inner_type(in_strategy, "in_strategy")
        if inner_type == "layout":
            in_strategy = self._extract_layout_value(in_strategy, "in_strategy")

        if not isinstance(out_strategy, (type(None), tuple)):
            raise TypeError(f"For 'Shard', the 'out_strategy' should be None or tuple, "
                            f"but got {type(out_strategy).__name__}.")
        if not isinstance(out_strategy, type(None)):
            logger.warning("Out_strategy is not in use currently, will be ignored in the following procedures.")
            inner_type = self._check_layout_inner_type(out_strategy, "out_strategy")
            if inner_type == "layout":
                out_strategy = self._extract_layout_value(out_strategy, "out_strategy")

        if not isinstance(device, str):
            raise TypeError(f"For 'Shard', the 'device' should be a string, "
                            f"but got {type(device).__name__}")
        if not isinstance(level, int):
            raise TypeError(f"For 'Shard', the 'level' should be an integer, "
                            f"but got {type(level).__name__}")

        if ms.get_algo_parameters("fully_use_devices") is True:
            logger.warning("After calling 'shard', the environment variable 'fully_use_devices' "
                           "will be overwritten as False.")
            ms.set_algo_parameters(fully_use_devices=False)

        if self._is_attrs_has_been_set(fn, in_strategy, out_strategy, device, level):
            return self.shard_fn
        shard_ = Shard()

        if isinstance(fn, ms.nn.Cell):
            for param in fn.trainable_params():
                param.param_info.is_in_pynative_shard = True

        # Set parameter layout to corresponding parameter
        self._set_param_layout_into_parameter(fn, parameter_plan)

        def shard_fn(*args):
            @ms.common.jit(hash_args=fn, backend="ms_backend")
            def after_shard(*args):
                return shard_(fn, in_strategy, out_strategy, device, level)(*args)

            return after_shard(*args)

        self.shard_fn = shard_fn
        self.fn = fn
        self.in_strategy = in_strategy
        self.out_strategy = out_strategy
        self.device = device
        self.level = level
        return self.shard_fn

    @staticmethod
    def _search_parameter_by_name(param_name: str, net):
        param_name = param_name.replace("self.", "")
        for param in net.trainable_params():
            if param.name == param_name:
                return param
        return None

    @staticmethod
    def _check_layout_is_valid(param_name, param_shape, param_strategy):
        if len(param_strategy) != len(param_shape):
            raise ValueError(f"For {param_name}, the length of param_strategy: {len(param_strategy)}, "
                             f"is not equal to param_shape len: {len(param_shape)}.")
        for i, _ in enumerate(param_strategy):
            if param_shape[i] % param_strategy[i] != 0:
                raise ValueError(f"For '{param_name}', the param_shape is {param_shape} and "
                                 f"the setting param_strategy is {param_strategy}. "
                                 f"The param_shape[{i}]: {param_shape[i]} cannot be divisible by "
                                 f"param_strategy[{i}]: {param_strategy[i]}.")

    def _set_param_layout_into_parameter(self, fn, parameter_plan):
        """ Set param_strategy into parameter if fn is a Cell and parameter_plan is a dict."""
        if parameter_plan is None:
            return
        if isinstance(parameter_plan, dict):
            if not isinstance(fn, ms.nn.Cell):
                raise TypeError(
                    f"If parameter_plan is set, type of fn must be mindspore.nn.Cell, but got {type(fn)}")
            for k in parameter_plan.keys():
                v = parameter_plan[k]
                if not isinstance(k, str) or not isinstance(v, (tuple, Layout)):
                    raise TypeError(f"For 'Shard', the type of each key and value in 'parameter_plan' must be str and "
                                    f"tuple/Layout, but got {type(k).__name__} and {type(v).__name__}")
        else:
            raise TypeError(f"For 'Shard', the 'parameter_plan' should be a dict or None, "
                            f"but got {type(parameter_plan).__name__}")

        for param_name in parameter_plan.keys():
            param_strategy = parameter_plan[param_name]
            param = self._search_parameter_by_name(param_name, fn)
            if param is None:
                logger.warning(
                    f"{param_name} is not exist, ignored its setting.")
                continue

            has_set = None
            if param.param_info.param_strategy:
                has_set = "strategy"
            if param.param_info.device_matrix:
                has_set = "layout"
            if has_set == "strategy":
                logger.warning(f"The layout of parameter '{param_name}' has been set to "
                               f"{param.param_info.param_strategy}, current setting will be ignored.")
            elif has_set == "layout":
                logger.warning(f"The layout of parameter '{param_name}' has been set, "
                               f"current setting will be ignored.")
            else:
                if isinstance(param_strategy, tuple):
                    self._check_layout_is_valid(param_name, param.shape, param_strategy)
                    param.param_info.param_strategy = param_strategy
                if isinstance(param_strategy, Layout):
                    param_layout = self._extract_layout_value((param_strategy,), "in_strategy")[0]
                    param.param_info.device_matrix = param_layout["device_matrix"]
                    param.param_info.tensor_map = param_layout["tensor_map"]
                    param.param_info.interleaved_parallel = param_layout["interleaved_parallel"]
                    param.param_info.alias_name = param_layout["alias_name"]

    def _is_attrs_has_been_set(self, fn, in_strategy, out_strategy, device, level):
        return self.shard_fn is not None and self.fn == fn and self.in_strategy == in_strategy and \
            self.out_strategy == out_strategy and self.device == device and self.level == level

    def _check_layout_inner_type(self, strategy, log_info):
        """Check inner item type of layout, should be int or ms.Layout."""
        strategy_set = set()
        for stra in strategy:
            if not isinstance(stra, (tuple, Layout)):
                raise TypeError(
                    f"The '{log_info}' should be a tuple(tuple(int)) or tuple(mindspore.parallel.Layout), "
                    f"but got {type(stra).__name__}")
            if isinstance(stra, Layout):
                strategy_set.add("layout")
            elif isinstance(stra, tuple):
                strategy_set.add("tuple")
                self._check_tuple_strategy(stra)
        if len(strategy_set) != 1:
            raise TypeError(
                f"For 'Shard', the strategy can only pass in consistent type for all dimensions.")
        return strategy_set.pop()

    def _extract_layout_value(self, layout, log_info):
        """Extract parallel layout value"""
        layout_value = None
        if layout is not None:
            if not isinstance(layout, tuple):
                raise TypeError(f'{log_info} must be tuple type, but got:{type(layout)}')
            layout_value = ()
            for in_ele in layout:
                if not isinstance(in_ele, Layout):
                    raise TypeError(f"The {log_info} item should be a object of class Layout.")
                layout_value += ({k: v for k, v in in_ele.to_dict().items() if k != "rank_list"},)
        return layout_value

    def _check_tuple_strategy(self, dim_strategy):
        if not all(isinstance(x, int) for x in dim_strategy):
            raise TypeError(
                f"The tuple strategy for each dimension should be tuple(int).")


def shard(fn, in_strategy, out_strategy=None, parameter_plan=None):
    """
    Specify the input and output slicing strategy for a Cell or function.
    In Graph mode, use this method to specify distribution strategy for a Cell,
    strategy for others will be set by sharding propagation.
    in_strategy and out_strategy define the input and output layout respectively.
    in_strategy/out_strategy should be a tuple, each element of which corresponds to the desired layout of
    this input/output, and None represents data_parallel,
    which can refer to the description of :func:`mindspore.ops.Primitive.shard`.
    The parallel strategies of remaining operators are derived from the strategy specified by the input and output.

    Note:
        - It is valid only in semi auto parallel or auto parallel mode.
          In other parallel modes, strategies set here will be ignored.
        - If the input contains Parameter, its strategy should be set in `in_strategy`.

    .. warning::
        The method is currently not supported in PyNative mode.

    Args:
        fn (Union[Cell, Function]): Function to be executed in parallel.
                                    Its arguments and return value must be Tensor.
                                    If `fn` is a Cell with parameters, `fn` needs to be an instantiated object,
                                    otherwise its arguments cannot be accessed.
        in_strategy (tuple): Define the layout of inputs, each element of the tuple should be a tuple(int) or
                             tuple(mindspore.parallel.Layout).
                             Tuple defines the layout of the corresponding input.
        out_strategy (Union[tuple, None], optional): Define the layout of outputs similar with `in_strategy`.
                                           Default: ``None`` .
        parameter_plan (Union[dict, None], optional): Define the layout for the specified parameters.
                                            Each element in dict
                                            defines the layout of the parameter like "param_name: layout".
                                            The key is a parameter name of type 'str'.
                                            The value is a 1-D integer tuple or a 1-D mindspore.parallel.Layout tuple,
                                            indicating the corresponding layout.
                                            If the parameter name is incorrect or the corresponding parameter
                                            has been set, the parameter setting will be ignored. Supported
                                            only when `fn` is a Cell with parameters.
                                            Default: ``None`` .

    Returns:
        Function, return the function that will be executed under auto parallel process.

    Raises:
        AssertionError: If parallel mode is not "auto_parallel" nor "semi_auto_parallel".
        TypeError: If `in_strategy` is not a tuple.
        TypeError: If `out_strategy` is not a tuple or None.
        TypeError: If any element in `in_strategy` is not a tuple(int) or tuple(mindspore.parallel.Layout).
        TypeError: If any element in `out_strategy` is not a tuple(int) or tuple(mindspore.parallel.Layout).
        TypeError: If `parameter_plan` is not a dict or None.
        TypeError: If any key in `parameter_plan` is not a str.
        TypeError: If any value in `parameter_plan` is not a tuple(int) or a tuple(mindspore.parallel.Layout).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, nn, ops
        >>> from mindspore.communication import init
        >>> from mindspore.parallel import shard
        >>> from mindspore.parallel import Layout
        >>> from mindspore.nn.utils import no_init_parameters
        >>> from mindspore.parallel.auto_parallel import AutoParallel
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>>
        >>> # Case 1: cell uses functional
        >>> class BasicBlock(nn.Cell):
        >>>     def __init__(self):
        >>>         super(BasicBlock, self).__init__()
        >>>         self.dense1 = nn.Dense(64, 64)
        >>>         self.gelu = nn.GELU()
        >>>         def my_add(x, y):
        >>>             x = ops.abs(x)
        >>>             return x + y
        >>>         # shard a function with tuple(int) strategies
        >>>         self.shard_my_add = shard(my_add, in_strategy=((2, 2), (1, 4)), out_strategy=((4, 1),))
        >>>
        >>>     def construct(self, x, u):
        >>>         x = self.gelu(x)
        >>>         y = self.gelu(u)
        >>>         y = x * y
        >>>         x = self.dense1(x)
        >>>         x = self.shard_my_add(x, y)
        >>>         return x
        >>>
        >>> class NetForward(nn.Cell):
        >>>     def __init__(self):
        >>>         super(NetForward, self).__init__()
        >>>         self.block1 = BasicBlock()
        >>>         self.block2 = BasicBlock()
        >>>         self.matmul = ops.MatMul()
        >>>
        >>>     def construct(self, x, y):
        >>>         x = self.matmul(x, y)
        >>>         x = self.block1(x, x)
        >>>         x = self.block2(x, x)
        >>>         return x
        >>>
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         # setting cell sharding strategy and parameter_plan by tuple(int)
        >>>         self.layer_net1 = NetForward()
        >>>         self.layer_net1_shard = shard(self.layer_net1, in_strategy=((4, 2), (2, 1)),
        ...                                          parameter_plan={"self.layer_net1.block1.weight": (4, 1)})
        >>>
        >>>         # setting cell sharding strategy and parameter_plan by tuple(ms.Layout)
        >>>         self.layer_net2 = NetForward()
        >>>         layout = Layout((4, 2, 1), ("dp", "mp", "sp"))
        >>>         in_layout = (layout("dp", "mp"), layout("mp", "sp"))
        >>>         param_layout = layout("dp", "sp")
        >>>         self.layer_net2_shard = shard(self.layer_net2, in_strategy=in_layout,
        ...                                          parameter_plan={"self.layer_net2.block2.weight": param_layout})
        >>>         self.flatten = nn.Flatten()
        >>>         self.layer1 = nn.Dense(64, 64)
        >>>         self.layer2 = nn.Dense(64, 32)
        >>>         self.add = ops.Add()
        >>>         self.matmul = ops.MatMul()
        >>>
        >>>     def construct(self, x, y):
        >>>         x = self.flatten(x)
        >>>         y = self.flatten(y)
        >>>         x = self.layer1(x)
        >>>         x = self.layer_net1_shard(x, y)
        >>>         x = self.layer_net2_shard(x, y)
        >>>         x = self.layer2(x)
        >>>         x = self.matmul(x, Tensor(np.ones(shape=(32, 32)), dtype=ms.float32))
        >>>         return x
        >>>
        >>> with no_init_parameters():
        >>>     net = Net()
        >>> x = Tensor(np.ones(shape=(64, 1, 8, 8)), dtype=ms.float32)
        >>> y = Tensor(np.ones(shape=(64, 1, 8, 8)), dtype=ms.float32)
        >>> parallel_net = AutoParallel(net, parallel_mode='sharding_propagation')
        >>> parallel_net(x, y)
        >>>
        >>> # Case 2: function uses functional sharding
        >>> def test_shard(x, y):
        ...     return x + y
        >>> x = Tensor(np.ones(shape=(32, 10)), dtype=ms.float32)
        >>> y = Tensor(np.ones(shape=(32, 10)), dtype=ms.float32)
        >>> output = shard(test_shard, in_strategy=((4, 2), (4, 2)))(x, y)
        >>> print(output.shape)
        (32, 10)

    """
    if ms.communication.management.get_group_size() == 1:
        return fn
    if not isinstance(fn, (ms.nn.Cell)):
        logger.warning("'fn' is not a mindspore.nn.Cell, and its definition cannot involve Parameter; "
                       "otherwise, the result may be incorrect.")

    return Shard()(fn, in_strategy, out_strategy, parameter_plan)
