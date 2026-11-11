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
"""CPU coverage for ordinary AMP training after Boost retirement."""
import pytest

from mindspore import Tensor, float32, nn
from mindspore.train import amp
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', mem_peak=0.01)
def test_training_step_without_boost():
    """
    Feature: Ordinary training after retiring Boost.
    Description: Run one O0 training step with MSE, SGD and a fixed loss scale on CPU.
    Expectation: The unscaled gradient is 2 and the weight changes from 1 to 0.8.
    """
    net = nn.Dense(1, 1, weight_init="ones", has_bias=False)
    opt = nn.SGD(net.trainable_params(), learning_rate=0.1, loss_scale=8.0)
    cell = amp.build_train_network(net, opt, nn.MSELoss(), boost_level="O0",
                                   loss_scale_manager=FixedLossScaleManager(8, drop_overflow_update=False))
    cell(Tensor([[1.0]], float32), Tensor([[0.0]], float32))
    assert net.weight.asnumpy().item() == pytest.approx(0.8)
