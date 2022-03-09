# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Inner operators for reinforcement learning."""

from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ..primitive import prim_attr_register, PrimitiveWithInfer
from ..._checkparam import Rel


class EnvCreate(PrimitiveWithInfer):
    r"""
    Create a built-in reinforcement learning environment. Repeated calls to the operator will return the previously
    created handle. Make sure to create a new operator instance if you want to create a new environment instance.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        name (str): Name of built-in environment.
        kwargs (any): Environment related parameters.

    Inputs:
        No inputs.

    Outputs:
        handle(Tensor): Handle of created environment instance with dtype int and shape (1,).

    Raises:
        TypeError: The environment not supported.
        TypeError: The environment parameters not provided.

    Supported Platforms:
        ``GPU``
    """

    def __init__(self, name, **kwargs):
        super(EnvCreate, self).__init__(self.__class__.__name__)
        self.add_prim_attr('name', name)
        for key in kwargs:
            self.add_prim_attr(key, kwargs[key])

    def infer_shape(self, *args):
        return (1,)

    def infer_dtype(self, *args):
        return mstype.int64


class EnvReset(PrimitiveWithInfer):
    r"""
    Reset reinforcement learning built-in environment.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        handle (int): The handle returned by `EnvCreate` operator.
        state_shape (list[tuple[int]]): The dimensionality of the state.
        state_dtype (list[:class:`mindspore.dtype`]): The type of the state.
        reward_shape (list[tuple[int]]): The dimensionality of the reward.
        reward_dtype (list[:class:`mindspore.dtype`]): The type of the reward.echo

    Inputs:
        No inputs.

    Outputs:
        Tensor, environment observation after reset.

    Raises:
        TypeError: Environment instance not exist.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, handle, state_shape, state_dtype):
        super(EnvReset, self).__init__(self.__class__.__name__)
        validator.check_value_type("handle", handle, [int], self.name)
        validator.check_value_type("state_shape", state_shape, [list, tuple], self.name)

    def infer_shape(self, *args):
        return self.state_shape

    def infer_dtype(self, *args):
        return self.state_dtype


class EnvStep(PrimitiveWithInfer):
    r"""
    Run one environment timestep.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        handle (int): The handle returned by `EnvCreate` operator.
        state_shape (list[tuple[int]]): The dimensionality of the state.
        state_dtype (list[:class:`mindspore.dtype`]): The type of the state.
        reward_shape (list[tuple[int]]): The dimensionality of the reward.
        reward_dtype (list[:class:`mindspore.dtype`]): The type of the reward.

    Inputs:
        - **action** (Tensor) - action

    Outputs:
        - **state** (Tensor) - Environment state after previous action.
        - **reward** (Tensor), - Reward returned by environment.
        - **done** (Tensor), whether the episode has ended.

    Raises:
        TypeError: If dtype of `handle` is not int.
        TypeError: If dtype of `state_shape` is neither tuple nor list.
        TypeError: If dtype of `state_dtype` is not int nor float.
        TypeError: If dtype of `state_shape` is neither tuple nor list.
        TypeError: If dtype of `reward_dtype` is not int nor float.

    Supported Platforms:
        ``GPU``
    """

    @prim_attr_register
    def __init__(self, handle, state_shape, state_dtype, reward_shape, reward_dtype):
        super(EnvStep, self).__init__(self.__class__.__name__)
        validator.check_value_type("handle", handle, [int], self.name)
        validator.check_value_type("state_shape", state_shape, [list, tuple], self.name)
        validator.check_value_type("reward_shape", reward_shape, [list, tuple], self.name)

    def infer_shape(self, action_shape):
        return self.state_shape, self.reward_shape, (self.state_shape[0],)

    def infer_dtype(self, action_dtype):
        return self.state_dtype, self.reward_dtype, mstype.bool_


class DiscountedReturn(PrimitiveWithInfer):
    r"""
    Calculate discounted return.

    Set discounted return as :math:`G`, discounted factor as :math:`\gamma`, reward as :math:`R`,
    timestep as :math:`t`, max timestep as :math:`N`. Then :math:`G_{t} = \Sigma_{t=0}^N{\gamma^tR_{t+1}}`

    For the reward sequence contain multi-episode, :math:`done` is introduced for indicating episode boundary,
    :math:`last\_state\_value` represents value after final step of last episode.

    Args:
        gamma (float): Discounted factor between [0, 1].

    Inputs:
        - **reward** (Tensor) - The reward sequence contains multi-episode.
          Tensor of shape :math:`(Timestep, Batch, ...)`
        - **done** (Tensor) - The episode done flag. Tensor of shape :math:`(Timestep, Batch)`.
          The data type must be bool.
        - **last_state_value** (Tensor) - The value after final step of last episode.
          Tensor of shape :math:`(Batch, ...)`

    Examples:
        >>> net = DiscountedReturn(gamma=0.99)
        >>> reward = Tensor([[1, 1, 1, 1]], dtype=mindspore.float32)
        >>> done = Tensor([[False, False, True, False]])
        >>> last_state_value = Tensor([2.], dtype=mindspore.float32)
        >>> ret = net(reward, done, last_state_value)
        >>> print(output.shape)
        (2, 2)
    """

    @prim_attr_register
    def __init__(self, gamma):
        self.init_prim_io_names(inputs=['reward', 'done', 'last_state_value'], outputs=['output'])
        validator.check_float_range(gamma, 0, 1, Rel.INC_RIGHT, "gamma", self.name)

    def infer_shape(self, reward_shape, done_shape, last_state_value_shape):
        if len(reward_shape) != len(done_shape):
            raise ValueError(f'{self.name} len(reward) and len(done) should be same, ',
                             f'but got {len(reward_shape)} and {len(done_shape)}.')

        if reward_shape[0] != done_shape[0]:
            raise ValueError(f'{self.name} timestep of reward and done should be same, ',
                             f'but got {reward_shape[0]} and {done_shape[0]}.')

        if reward_shape[1:] != last_state_value_shape:
            raise ValueError(f'{self.name} state value shape should be match, ',
                             f'but got {reward_shape[1:]} and {last_state_value_shape}.')
        return reward_shape

    def infer_dtype(self, reward_dtype, done_dtype, last_state_value_dtype):
        valid_dtypes = (mstype.float16, mstype.float32)
        args = {"reward": reward_dtype, "last_state_value": last_state_value_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, valid_dtypes, self.name)
        validator.check_tensor_dtype_valid('done_dtype', done_dtype, [mstype.bool_], self.name)
        return reward_dtype


class CudnnGRU(PrimitiveWithInfer):
    """
    Performs the Stacked GRU (Gated Recurrent Unit) on the input.

    For detailed information, please refer to :class:`mindspore.nn.GRU`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`.
        bidirectional (bool): Specifies whether it is a bidirectional GRU.
        dropout (float): If not 0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. The range of dropout is [0.0, 1.0].

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`) or
          (batch_size, seq_len, `input_size`).
        - **h** (tuple) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **w** (Tensor) - The input tensor which states for weights.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`, `reserve`, `state`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **reserve** (Tensor) - Tensor of shape (r, 1).
        - **state** (Tensor) - Random number generator state and its shape is (s, 1).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_size = 10
        >>> hidden_size = 2
        >>> num_layers = 1
        >>> seq_len = 5
        >>> batch_size = 2
        >>>
        >>> import mindspore.ops.operations._rl_inner_ops as rl_ops
        >>> net = rl_ops.CudnnGRU(input_size, hidden_size, num_layers, True, False, 0.0)
        >>> input_tensor = Tensor(np.ones([seq_len, batch_size, input_size]).astype(np.float32))
        >>> h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        >>> w = Tensor(np.ones([84, 1, 1]).astype(np.float32))
        >>> output, hn,  _, _ = net(input_tensor, h0, w)
        >>> print(output)
        [[[1.  1. ]
          [1.  1. ]]
         [[1.  1. ]
          [1.  1. ]]
         [[1.  1.]
          [1.  1.]]
         [[1.  1. ]
          [1.  1. ]]
         [[1.  1. ]
          [1.  1. ]]]
    """

    @prim_attr_register
    def __init__(self, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout):
        """Initialize GRU."""
        self.input_size = validator.check_positive_int(input_size, "input_size", self.name)
        self.hidden_size = validator.check_positive_int(hidden_size, "hidden_size", self.name)
        self.num_layers = validator.check_positive_int(num_layers, "num_layers", self.name)
        self.has_bias = validator.check_value_type("has_bias", has_bias, (bool,), self.name)
        self.bidirectional = validator.check_value_type("bidirectional", bidirectional, (bool,), self.name)
        self.dropout = validator.check_value_type("dropout", dropout, [float], self.name)
        self.dropout = validator.check_float_range(dropout, 0, 1, Rel.INC_BOTH, 'dropout', self.name)

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

    def infer_shape(self, x_shape, h_shape, w_shape):
        validator.check_equal_int(len(x_shape), 3, "x rank", self.name)
        validator.check_equal_int(x_shape[2], self.input_size, "x[2]", self.name)

        validator.check_equal_int(len(h_shape), 3, "h rank", self.name)

        validator.check_int(h_shape[0], self.num_layers * self.num_directions, Rel.EQ, "h[0]", self.name)
        validator.check_equal_int(h_shape[1], x_shape[1], "h[1]", self.name)
        validator.check_int(h_shape[2], self.hidden_size, Rel.EQ, "h[2]", self.name)

        y_shape = (x_shape[0], x_shape[1], self.hidden_size * self.num_directions)

        # set arbitrary shape for reserved space
        reserved_shape = (1, 1)
        state_shape = (1, 1)
        return y_shape, h_shape, reserved_shape, state_shape

    def infer_dtype(self, x_dtype, h_dtype, w_dtype):
        args = {'x': x_dtype, 'h': h_dtype, 'w': w_dtype}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.float32, mstype.float16), self.name)
        return x_dtype, x_dtype, x_dtype, x_dtype


class PriorityReplayBufferCreate(PrimitiveWithInfer):
    r"""
    PriorityReplayBuffer is experience container used in Deep Q-Networks.
    The algorithm is proposed in `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`.
    Same as the normal replay buffer, it lets the reinforcement learning agents remember and reuse experiences from the
    past. Besides, it replays important transitions more frequently and improve sample effciency.

    Args:
        capcity (int64): Capacity of the buffer. It is recommended that set capacity to pow(2, N).
        alpha (float): The parameter determines how much prioritization is used between [0, 1].
        beta (float): The parameter determines how much compensations for non-uniform probabilities between [0, 1].
        shapes (list[tuple[int]]): The dimensionality of the transition.
        dtypes (list[:class:`mindspore.dtype`]): The type of the transition.
        seed0 (int): Random seed0, must be non-negative. Default: 0.
        seed1 (int): Random seed1, must be non-negative. Default: 0.

    Outputs:
        handle(Tensor): Handle of created priority replay buffer instance with dtype int64 and shape (1,).

    Raises:
        TypeError: The args not provided.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self, capacity, alpha, beta, shapes, dtypes, seed0, seed1):
        """Initialize PriorityReplaBufferCreate."""
        validator.check_int(capacity, 1, Rel.GE, "capacity", self.name)
        validator.check_float_range(alpha, 0.0, 1.0, Rel.INC_BOTH)
        validator.check_float_range(beta, 0.0, 1.0, Rel.INC_BOTH)
        validator.check_value_type("shape of init data", shapes, [tuple, list], self.name)
        validator.check_value_type("dtypes of init data", dtypes, [tuple, list], self.name)
        validator.check_non_negative_int(seed0, "seed0", self.name)
        validator.check_non_negative_int(seed1, "seed1", self.name)

    def infer_shape(self):
        return (1,)

    def infer_dtype(self):
        return mstype.int64


class PriorityReplayBufferPush(PrimitiveWithInfer):
    r"""
    Push a transition to the priority replay buffer.

    Args:
        handle(Tensor): Priority replay buffer instance handle with dtype int64 and shape (1,).

    Outputs:
        handle(Tensor): Priority replay buffer instance handle with dtype int64 and shape (1,).

    Raises:
        TypeError: The priority replay buffer not created before.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self, handle):
        """Initialize PriorityReplaBufferPush."""
        validator.check_int(handle, 0, Rel.GE, "handle", self.name)

    def infer_shape(self, *inputs):
        return (1,)

    def infer_dtype(self, *inputs):
        return mstype.int64


class PriorityReplayBufferSample(PrimitiveWithInfer):
    r"""
    Sample a transition to the priority replay buffer.

    .. warning::
            This is an experimental prototype that is subject to change and/or deletion.

    Args:
        handle(Tensor): Priority replay buffer instance handle with dtype int64 and shape (1,).
        batch_size (int): The size of the sampled transitions.
        shapes (list[tuple[int]]): The dimensionality of the transition.
        dtypes (list[:class:`mindspore.dtype`]): The type of the transition.

    Outputs:
        tuple(Tensor): Transition with its indices and bias correction weights.

    Raises:
        TypeError: The priority replay buffer not created before.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self, handle, batch_size, shapes, dtypes):
        """Initialize PriorityReplaBufferSample."""
        validator.check_int(handle, 0, Rel.GE, "capacity", self.name)
        validator.check_int(batch_size, 1, Rel.GE, "batch_size", self.name)
        validator.check_value_type("shape of init data", shapes, [tuple, list], self.name)
        validator.check_value_type("dtypes of init data", dtypes, [tuple, list], self.name)

    def infer_shape(self):
        output_shape = [(self.batch_size,), (self.batch_size,)]
        for shape in self.shapes:
            output_shape.append((self.batch_size,) + shape)
        # indices, weights, transitions
        return tuple(output_shape)

    def infer_dtype(self):
        return (mstype.int64, mstype.float32) + self.dtypes


class PriorityReplayBufferUpdate(PrimitiveWithInfer):
    r"""
    Update transition prorities.

    Args:
        handle(Tensor): Priority replay buffer instance handle with dtype int64 and shape (1,).

    Inputs:
        - **indices** (Tensor) - transition indices.
        - **priorities** (Tensor) - Transition priorities.

    Outputs:
        Priority replay buffer instance handle with dtype int64 and shape (1,).

    Raises:
        TypeError: The priority replay buffer not created before.

    Supported Platforms:
        ``CPU``
    """

    @prim_attr_register
    def __init__(self, handle):
        """Initialize PriorityReplaBufferUpdate."""
        validator.check_int(handle, 0, Rel.GE, "capacity", self.name)

    def infer_shape(self, indices, priorities):
        return (1,)

    def infer_dtype(self, indices, priorities):
        return mstype.int64
