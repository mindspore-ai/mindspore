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
"""Defines other operators with functional form."""
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import rotary_position_embedding
from mindspore.ops.auto_generate import moe_distribute_dispatch, moe_distribute_combine
from mindspore.ops.auto_generate.gen_ops_prim import moe_init_routing_v2_op

partial_ = P.Partial()
depend_ = P.Depend()
move_to_ = P.MoveTo()


def partial(func, *args):
    """
    Makes a partial function instance. Partial function can be used to derived specialized
    functions from general functions by fixing the value of certain arguments.

    Args:
        func (FunctionType): The incoming function.
        args (Tensor): The arguments of the incoming function.

    Returns:
        FunctionType, partial function bound with arguments.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> def show_input(x, y, z):
        ...     return x, y, z
        >>> partial_show_input = ops.partial(show_input, Tensor(1))
        >>> output1 = partial_show_input(Tensor(2), Tensor(3))
        >>> print(output1)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64,
         value= 3))
        >>> output2 = partial_show_input(Tensor(3), Tensor(4))
        >>> print(output2)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 3), Tensor(shape=[], dtype=Int64,
         value= 4))
    """
    return partial_(func, *args)


def depend(value, expr):
    """
    depend is used for processing dependency operations.

    In most scenarios, if operators have IO side effects or memory side effects,
    they will be executed according to the user's semantics. In some scenarios,
    if the two operators A and B have no order dependency, and A must be executed
    before B, we recommend using Depend to specify their execution order. The
    usage method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = depend(y, a)
                                --->        b = B(y)

    Args:
        value (Tensor): The real value to return for depend operator.
        expr (Expression): The expression to execute with no outputs.

    Returns:
        Tensor, the value passed by last operator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.softmax = ops.Softmax()
        ...
        ...     def construct(self, x, y):
        ...         mul = x * y
        ...         y = ops.depend(y, mul)
        ...         ret = self.softmax(y)
        ...         return ret
        ...
        >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
        >>> print(output)
        [[0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]]
    """
    return depend_(value, expr)


def move_to(input, to="CPU", blocking=True):  # pylint: disable=redefined-outer-name
    """
    Copy tensor to target device synchronously or asynchronously, default synchronously.

    .. note::
        This interface currently only supports Graph mode with jit_level of O0 or O1.

    Args:
        input (Union[Tensor, list[int], tuple[int]]): The input tensor. When the input is a list or a tuple, it will be
                                                      converted to tensor before copying.
        to (str, optional): Specify the target device, with optional values of ``"Ascend"`` and ``"CPU"``.
                            Default ``"CPU"`` .
        blocking (bool, optional): Whether to use synchronous copying. Default ``True``.

    Returns:
        A new tensor on target device.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import nn, ops, Tensor
        >>> mindspore.set_context(mode=mindspore.GRAPH_MODE)
        >>> class MoveToNet(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...
        ...     def construct(self, x):
        ...         cpu_x = ops.move_to(x, "CPU")
        ...         npu_x = ops.move_to(cpu_x, "Ascend")
        ...         return npu_x
        ...
        >>> net = MoveToNet()
        >>> x = Tensor([1, 2, 3], mindspore.int64)
        >>> y = net(x)
        >>> print(y)
        [1 2 3]
    """
    return move_to_(input, to, blocking)


def moe_init_routing_v2(x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                        expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
    """
    Performs routing on the computation result of MoeGatingTopKSoftmaxV2.

    Notes:
        - NUM_ROWS: The number of rows in 'x', which represents the number of original input tokens.
        - H: The number of cols in 'x', which denotes for the hiddens of input tokens.
        - K: The number of experts corresponding to each row of features in the output of MoeGatingTopKSoftmaxV2.
        - Currently, MoeInitRoutingV2 does not support mutable inputs.

    Args:
        x (Tensor): A 2D tensor, which contains the input feature tokens. The shape of the tensor
        is (NUM_ROWS, H). Supported dtypes: Float16, BFloat16, Float32.
        expert_idx (Tensor): A 2D tensor, representing K experts corresponding to each row of features
        in the output of MoeGatingTopKSoftmaxV2. The shape of the tensor is (NUM_ROWS, K).
        Supported dtype: int32. In the Drop/Pad scenario or when the output 'expert_tokens_count_or_cumsum'
        is required in the non-Drop/Pad scenario, the value range of this tensor is [0, 'expert_num' - 1].
        In other scenarios, the value must be greater than or equal to 0.
        active_num (int64): Indicates whether the scenario is Active, this value works only
        when 'drop_pad_mode' = 0. The value must be greater than or equal to 0 where 0 is the Dropless scenario
        and others represent the Active scenario.
        expert_capacity (int64): The number of tokens that each expert can process.
        The value must be greater than or equal to 0. In the Drop/Pad scenario, the value range is (0, NUM_ROWS].
        expert_num (int64): The number of experts. The value must be greater than or equal to 0.
        In the Drop/Pad scenario or when 'expert_tokens_count_or_cumsum_flag' > 0, the value must be greater than 0.
        drop_pad_mode (int64): Indicates whether the scenario is Drop/Pad. The value must be 0 or 1:

          - 0: non-Drop/Pad scenario.
          - 1: Drop/Pad scenario.
        expert_tokens_count_or_cumsum_flag (int64): A flag which controls whether the
        output 'expert_tokens_count_or_cumsum' is required. The value must be 0, 1 or 2:

          - 0: The output 'expert_tokens_count_or_cumsum' is not required.
          - 1: The output 'expert_tokens_count_or_cumsum' is required,
            which represents the accumulated number of tokens processed by each expert.
          - 2: The output 'expert_tokens_count_or_cumsum' is required,
            which represents the number of tokens processed by each expert.
        expert_tokens_before_capacity_flag (bool): A flag which controls whether the
        output 'expert_tokens_before_capacity' is required.

          - False: The output 'expert_tokens_before_capacity' is not required.
          - True: The output 'expert_tokens_before_capacity' is required, which represents the
            number of tokens processed by each expert before the drop.

    Returns:
        A tuple of tensors.
        expanded_x (Tensor): A 2D/3D tensor which indicates features extended based on 'expert_idx'.
        The shape of the tensor depends on scenarios:

          - Dropless scenario: The shape is (NUM_ROWS * K, H).
          - Active scenario: The shape is (min('active_num', NUM_ROWS * K), H).
          - Drop/Pad scenario: The shape is ('expert_num', 'expert_capacity', H).
        Supported dtypes: Float16, BFloat16, Float32.
        expanded_row_idx (Tensor): A 1D tensor which represents the mapping between 'expanded_x' and 'x'.
        The shape of the tensor is (NUM_ROWS * K). Supported dtype: int32.
        expert_tokens_count_or_cumsum (Tensor): A 1D tensor which indicates the statistics on the
        number of tokens processed by each expert and the accumulated value.
        The value of the tensor is valid only in the non-Drop/Pad scenario which is controlled by the
        'expert_tokens_count_or_cumsum_flag'.
        The value of this tensor is dirty data from the memory when it is not required.
        The shape of the tensor is ('expert_num'). Supported dtype: int32.
        expert_tokens_before_capacity (Tensor): A 1D tensor which indicates the statistics on the
        number of tokens processed by each expert before the drop.
        The value of the tensor is valid only in the Drop/Pad scenario which is controlled by the
        'expert_tokens_before_capacity_flag'.
        The value of this tensor is dirty data from the memory when it is not required.
        The shape of the tensor is ('expert_num'). Supported dtype: int32.

    Raises:
        TypeError: If the data type of input Tensor does not match the description in args.
        ShapeError: If the shape of input Tensor does not match the description in args.
        ValueError: If the value of the inputs do not match the description in args.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[0.1, 0.2, 0.3], [0.2, 0.7, 0.8], [0.3, 0.3, 0.5]]), ms.float16)
        >>> expert_idx = Tensor(np.array([[0, 1, 1], [2, 1, 1], [0, 0, 0]]), ms.int32)
        >>> active_num = 3
        >>> expert_capacity = 2
        >>> expert_num = 3
        >>> drop_pad_mode = 1
        >>> out1, out2 = ops.moe_init_routing_v2(x, expert_idx, active_num, expert_capacity,
        expert_num, drop_pad_mode, 0, False)
        >>> print(out1)
        [[[0.1  0.2  0.3]
          [0.3  0.3  0.5]]
         [[0.1  0.2  0.3]
          [0.1  0.2  0.3]]
         [[0.2  0.7  0.8]
          [0.   0.   0. ]]]
        >>> print(out2)
        [ 0  2  3  4 -1 -1  1 -1 -1 ]
    """
    expanded_x, expanded_row_idx, \
    expert_tokens_count_or_cumsum, \
    expert_tokens_before_capacity = moe_init_routing_v2_op(x, expert_idx, active_num, expert_capacity,
                                                           expert_num, drop_pad_mode,
                                                           expert_tokens_count_or_cumsum_flag,
                                                           expert_tokens_before_capacity_flag)
    if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
        return expanded_x, expanded_row_idx, expert_tokens_before_capacity
    if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag != 0:
        return expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum
    return expanded_x, expanded_row_idx

__all__ = [
    'depend',
    'partial',
    'rotary_position_embedding',
    'move_to',
    'moe_init_routing_v2',
    'moe_distribute_dispatch',
    'moe_distribute_combine'
]
__all__.sort()
