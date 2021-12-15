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
