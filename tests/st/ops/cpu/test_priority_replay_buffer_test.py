# Copyright 2022 Huawei Technologies Co., Ltd
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

import sys
import pytest
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations._rl_inner_ops import PriorityReplayBufferCreate, PriorityReplayBufferPush
from mindspore.ops.operations._rl_inner_ops import PriorityReplayBufferSample, PriorityReplayBufferUpdate
from mindspore.ops.operations._rl_inner_ops import PriorityReplayBufferDestroy
from mindspore.common.api import _pynative_executor


class PriorityReplayBuffer(nn.Cell):
    def __init__(self, capacity, alpha, sample_size, shapes, dtypes, seed0, seed1):
        super(PriorityReplayBuffer, self).__init__()
        handle = PriorityReplayBufferCreate(capacity, alpha, shapes, dtypes, seed0, seed1)().asnumpy().item()
        self.push_op = PriorityReplayBufferPush(handle).add_prim_attr('side_effect_io', True)
        self.sample_op = PriorityReplayBufferSample(handle, sample_size, shapes, dtypes)
        self.update_op = PriorityReplayBufferUpdate(handle).add_prim_attr('side_effect_io', True)
        self.destroy_op = PriorityReplayBufferDestroy(handle).add_prim_attr('side_effect_io', True)

    def push(self, *transition):
        return self.push_op(transition)

    def sample(self, beta):
        return self.sample_op(beta)

    def update_priorities(self, indices, priorities):
        return self.update_op(indices, priorities)

    def destroy(self):
        return self.destroy_op()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_priority_replay_buffer_ops():
    """
    Feature: PriorityReplayBuffer used in Reinforcement Learning.
    Description: test cases PriorityReplayBuffer.
    Expectation: push, sample, update operators result correct.
    """
    if sys.platform == 'darwin':
        return

    capacity = 200
    batch_size = 32
    state_shape, state_dtype = (17,), mindspore.float32
    action_shape, action_dtype = (6,), mindspore.int32
    shapes = (state_shape, action_shape)
    dtypes = (state_dtype, action_dtype)
    prb = PriorityReplayBuffer(capacity, 1., batch_size, shapes, dtypes, seed0=0, seed1=42)

    # Push 100 timestep transitions to priority replay buffer.
    for i in range(100):
        state = Tensor(np.ones(state_shape) * i, state_dtype)
        action = Tensor(np.ones(action_shape) * i, action_dtype)
        prb.push(state, action)

    # Sample a batch of transitions, the indices should be consist with transition.
    indices, weights, states, actions = prb.sample(1.)
    assert np.all(indices.asnumpy() < 100)
    states_expect = np.broadcast_to(indices.asnumpy().reshape(-1, 1), states.shape)
    actions_expect = np.broadcast_to(indices.asnumpy().reshape(-1, 1), actions.shape)
    assert np.allclose(states.asnumpy(), states_expect)
    assert np.allclose(actions.asnumpy(), actions_expect)

    # Minimize the priority, these transition will not be sampled next time.
    priorities = Tensor(np.ones(weights.shape) * 1e-7, mindspore.float32)
    prb.update_priorities(indices, priorities)

    indices_new, _, states_new, actions_new = prb.sample(1.)
    assert np.all(indices_new.asnumpy() < 100)
    assert np.all(indices.asnumpy() != indices_new.asnumpy())
    states_expect = np.broadcast_to(indices_new.asnumpy().reshape(-1, 1), states.shape)
    actions_expect = np.broadcast_to(indices_new.asnumpy().reshape(-1, 1), actions.shape)
    assert np.allclose(states_new.asnumpy(), states_expect)
    assert np.allclose(actions_new.asnumpy(), actions_expect)

    prb.destroy()
    _pynative_executor.sync()
