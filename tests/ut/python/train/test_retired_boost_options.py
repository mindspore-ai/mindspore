# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Compatibility tests for retired Boost training options."""
from importlib.util import find_spec

import pytest

from mindspore import nn
from mindspore.train import Model, amp
from mindspore.train.loss_scale_manager import FixedLossScaleManager


@pytest.mark.parametrize("options", [{}, {"boost_level": "O0", "acc_level": "O0", "boost_config_dict": {}}])
def test_model_without_boost(options):
    """
    Feature: Model without Boost.
    Description: Construct a model with default or explicitly disabled legacy settings.
    Expectation: The original prediction network remains usable.
    """
    net = nn.ReLU()
    assert Model(net, **options).predict_network is net


@pytest.mark.parametrize("options", [{"boost_level": "O1", "acc_level": "O0"},
                                     {"acc_level": "O2"}, {"boost_config_dict": {"boost": {}}}])
def test_model_rejects_boost(options):
    """
    Feature: Retired Boost training strategies.
    Description: Request a strategy through a level, its old alias, or a configuration.
    Expectation: Model raises an explicit retirement error instead of silently disabling it.
    """
    with pytest.raises(ValueError, match="Boost training has been retired"):
        Model(nn.ReLU(), **options)


def test_amp_rejects_boost_level():
    """
    Feature: AMP training network construction.
    Description: Request the retired Boost level with an ordinary optimizer.
    Expectation: AMP rejects the Boost option explicitly.
    """
    net = nn.Dense(1, 1)
    opt = nn.SGD(net.trainable_params(), learning_rate=0.1)
    with pytest.raises(ValueError, match="Boost training has been retired"):
        amp.build_train_network(net, opt, boost_level="O2")


def test_retired_boost_package_and_training_cell():
    """
    Feature: Ordinary training after retiring the Boost package.
    Description: Build an AMP O0 training cell with SGD and a fixed loss scale.
    Expectation: Boost is absent and the ordinary training cell uses the supplied optimizer and scale.
    """
    assert find_spec("mindspore.boost") is None
    net = nn.Dense(1, 1, weight_init="ones", has_bias=False)
    opt = nn.SGD(net.trainable_params(), learning_rate=0.1, loss_scale=8.0)
    cell = amp.build_train_network(net, opt, nn.MSELoss(), boost_level="O0",
                                   loss_scale_manager=FixedLossScaleManager(8, drop_overflow_update=False))
    assert isinstance(cell, nn.TrainOneStepCell)
    assert cell.optimizer is opt
    assert cell.sens == 8
