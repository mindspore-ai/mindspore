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
""" test_run_config """
import pytest

from mindspore.train import CheckpointConfig


def test_init():
    """ test_init """
    save_checkpoint_steps = 1
    keep_checkpoint_max = 5

    config = CheckpointConfig(save_checkpoint_steps,
                              keep_checkpoint_max)

    assert config.save_checkpoint_steps == save_checkpoint_steps
    assert config.keep_checkpoint_max == keep_checkpoint_max
    policy = config.get_checkpoint_policy()
    assert policy['keep_checkpoint_max'] == keep_checkpoint_max


def test_arguments_values():
    """ test_arguments_values """
    config = CheckpointConfig()
    assert config.save_checkpoint_steps == 1
    assert config.save_checkpoint_seconds is None
    assert config.keep_checkpoint_max == 5
    assert config.keep_checkpoint_per_n_minutes is None

    with pytest.raises(TypeError):
        CheckpointConfig(save_checkpoint_steps='abc')
    with pytest.raises(TypeError):
        CheckpointConfig(save_checkpoint_seconds='abc')
    with pytest.raises(TypeError):
        CheckpointConfig(keep_checkpoint_max='abc')
    with pytest.raises(TypeError):
        CheckpointConfig(keep_checkpoint_per_n_minutes='abc')

    with pytest.raises(ValueError):
        CheckpointConfig(save_checkpoint_steps=-1)
    with pytest.raises(ValueError):
        CheckpointConfig(save_checkpoint_seconds=-1)
    with pytest.raises(ValueError):
        CheckpointConfig(keep_checkpoint_max=-1)
    with pytest.raises(ValueError):
        CheckpointConfig(keep_checkpoint_per_n_minutes=-1)
