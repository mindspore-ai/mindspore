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
import pytest

import mindspore.context as context
import mindspore.nn as nn
from .weight_decay_utils import dynamic_weight_decay_cmp, WeightDecaySchdule, Net
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_momentum_dynamic_weight_decay(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for Momentum
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    net1, net2 = Net(), Net()
    weight_decay_schedule = WeightDecaySchdule()
    optimizer1 = nn.Momentum(net1.trainable_params(), momentum=0.001, learning_rate=0.001, weight_decay=0.001)
    optimizer2 = nn.Momentum(net2.trainable_params(), momentum=0.001, learning_rate=0.001,
                             weight_decay=weight_decay_schedule)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_momentum_dynamic_weight_decay_group(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for Momentum
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    weight_decay_schedule = WeightDecaySchdule()
    net1, net2 = Net(), Net()

    net1_fc1_params = list(filter(lambda x: 'fc1' in x.name, net1.trainable_params()))
    net1_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net1.trainable_params()))

    net2_fc1_params = list(filter(lambda x: 'fc1' in x.name, net2.trainable_params()))
    net2_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net2.trainable_params()))

    params1 = [{'params': net1_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net1_fc2_params, 'weight_decay': 0.001, 'lr': 0.001}]

    params2 = [{'params': net2_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net2_fc2_params, 'weight_decay': weight_decay_schedule, 'lr': 0.001}]

    optimizer1 = nn.Momentum(params1, momentum=0.001, learning_rate=0.001, weight_decay=0.001)
    optimizer2 = nn.Momentum(params2, momentum=0.001, learning_rate=0.001, weight_decay=0.001)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamweightdecay_dynamic_weight_decay(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for AdamWeightDecay
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    net1, net2 = Net(), Net()
    weight_decay_schedule = WeightDecaySchdule()
    optimizer1 = nn.AdamWeightDecay(net1.trainable_params(), learning_rate=0.001, weight_decay=0.001)
    optimizer2 = nn.AdamWeightDecay(net2.trainable_params(), learning_rate=0.001, weight_decay=weight_decay_schedule)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_adamweightdecay_dynamic_weight_decay_group(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for Momentum
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    weight_decay_schedule = WeightDecaySchdule()
    net1, net2 = Net(), Net()

    net1_fc1_params = list(filter(lambda x: 'fc1' in x.name, net1.trainable_params()))
    net1_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net1.trainable_params()))

    net2_fc1_params = list(filter(lambda x: 'fc1' in x.name, net2.trainable_params()))
    net2_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net2.trainable_params()))

    params1 = [{'params': net1_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net1_fc2_params, 'weight_decay': 0.001, 'lr': 0.001}]

    params2 = [{'params': net2_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net2_fc2_params, 'weight_decay': weight_decay_schedule, 'lr': 0.001}]

    optimizer1 = nn.AdamWeightDecay(params1, learning_rate=0.001, weight_decay=0.001)
    optimizer2 = nn.AdamWeightDecay(params2, learning_rate=0.001, weight_decay=0.001)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_lamb_dynamic_weight_decay_graph_group(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for LAMB
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    weight_decay_schedule = WeightDecaySchdule()
    net1, net2 = Net(), Net()

    net1_fc1_params = list(filter(lambda x: 'fc1' in x.name, net1.trainable_params()))
    net1_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net1.trainable_params()))

    net2_fc1_params = list(filter(lambda x: 'fc1' in x.name, net2.trainable_params()))
    net2_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net2.trainable_params()))

    params1 = [{'params': net1_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net1_fc2_params, 'weight_decay': 0.001, 'lr': 0.001}]

    params2 = [{'params': net2_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net2_fc2_params, 'weight_decay': weight_decay_schedule, 'lr': 0.001}]

    optimizer1 = nn.Lamb(params1, learning_rate=0.001, weight_decay=0.001)
    optimizer2 = nn.Lamb(params2, learning_rate=0.001, weight_decay=0.001)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_lars_dynamic_weight_decay(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for Lars
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    net1, net2 = Net(), Net()
    weight_decay_schedule = WeightDecaySchdule()

    opt1 = nn.Momentum(net1.trainable_params(), momentum=0.001, learning_rate=0.001, weight_decay=0.001)
    opt2 = nn.Momentum(net2.trainable_params(), momentum=0.001, learning_rate=0.001, weight_decay=weight_decay_schedule)
    optimizer1 = nn.LARS(opt1, lars_filter=lambda x: 'LayerNorm' not in x.name)
    optimizer2 = nn.LARS(opt2, lars_filter=lambda x: 'LayerNorm' not in x.name)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_lars_dynamic_weight_decay_group(mode):
    """
    Feature: Dynamic weight decay
    Description: Test dynamic weight decay for Lars
    Expectation: The value of decay changes according to preset weight decay schedule
    """
    context.set_context(mode=mode)
    net1, net2 = Net(), Net()
    weight_decay_schedule = WeightDecaySchdule()

    net1_fc1_params = list(filter(lambda x: 'fc1' in x.name, net1.trainable_params()))
    net1_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net1.trainable_params()))

    net2_fc1_params = list(filter(lambda x: 'fc1' in x.name, net2.trainable_params()))
    net2_fc2_params = list(filter(lambda x: 'fc1' not in x.name, net2.trainable_params()))

    params1 = [{'params': net1_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net1_fc2_params, 'weight_decay': 0.001, 'lr': 0.001}]

    params2 = [{'params': net2_fc1_params, 'weight_decay': 0.01, 'lr': 0.01},
               {'params': net2_fc2_params, 'weight_decay': weight_decay_schedule, 'lr': 0.001}]

    opt1 = nn.Momentum(params1, momentum=0.001, learning_rate=0.001, weight_decay=0.001)
    opt2 = nn.Momentum(params2, momentum=0.001, learning_rate=0.001, weight_decay=0.001)
    optimizer1 = nn.LARS(opt1, lars_filter=lambda x: 'LayerNorm' not in x.name)
    optimizer2 = nn.LARS(opt2, lars_filter=lambda x: 'LayerNorm' not in x.name)
    dynamic_weight_decay_cmp(net1, net2, optimizer1, optimizer2)
