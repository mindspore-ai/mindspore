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
import pytest
import numpy as np
from mindspore import nn
import mindspore as ms
from mindspore.experimental import optim
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_sequential_lr_scheduler(mode):
    """
    Feature: SequentialLR
    Description: Verify the result of SequentialLR
    Expectation: success
    """
    # Graph mode use fallback with list of cell getitem, will be fixed later.
    ms.set_context(mode=mode)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.1)
    scheduler1 = optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=2)
    scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
    expect_list = [[0.01], [0.1], [0.09], [0.081], [0.0729], [0.06561]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(6):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_reduce_lr_on_plateau(mode):
    """
    Feature: ReduceLROnPlateau
    Description: Verify the result of ReduceLROnPlateau
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0)
    expect_list = [[0.1], [0.01], [0.001], [0.001], [0.0001]]
    metrics = [1, 1.5, 1.8, 0.4, 0.5]

    class SchedNet(nn.Cell):
        def construct(self, metric):
            scheduler.step(metric)
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(5):
        current_lr = sched_net(metrics[i])
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_step_lr(mode):
    """
    Feature: StepLR
    Description: Verify the result of StepLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    expect_list = [[0.05], [0.005], [0.005], [0.0005], [0.0005]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(5):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_linear_lr(mode):
    """
    Feature: LinearLR
    Description: Verify the result of LinearLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.05)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=4)
    expect_list = [[0.03125], [0.0375], [0.04375], [0.05], [0.05], [0.05]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(6):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_exponential_lr(mode):
    """
    Feature: ExponentialLR
    Description: Verify the result of ExponentialLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.01)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    expect_list = [[0.005], [0.0025], [0.00125], [0.000625], [0.0003125]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(5):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_polynomial_lr(mode):
    """
    Feature: PolynomialLR
    Description: Verify the result of PolynomialLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.01)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer)
    expect_list = [[0.008], [0.006], [0.004], [0.002], [0], [0]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(6):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_lambdalr_scheduler(mode):
    """
    Feature: LambdaLR
    Description: Verify the result of LambdaLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()

    dense_params = list(filter(lambda x: 'dens' in x.name, net.trainable_params()))
    no_dense_params = list(filter(lambda x: 'dens' not in x.name, net.trainable_params()))
    group_params = [{'params': dense_params},
                    {'params': no_dense_params, 'lr': 1.}]
    optimizer = optim.Adam(group_params, 0.1)

    lambda1 = lambda epoch: epoch // 3
    lambda2 = lambda epoch: 0.9 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    expect_list = [[0.0, 0.9], [0.0, 0.81], [0.1, 0.729],
                   [0.1, 0.6561], [0.1, 0.59049], [0.2, 0.531441]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(6):
        current_lr = sched_net()
        assert np.allclose([float(lr) for lr in current_lr], expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_multiplicative_lr(mode):
    """
    Feature: MultiplicativeLR
    Description: Verify the result of MultiplicativeLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.1)
    lmbda = lambda epoch: 0.9
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    expect_list = [[0.09], [0.081], [0.0729], [0.06561], [0.059049], [0.0531441]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(6):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_multistep_lr(mode):
    """
    Feature: MultiStepLR
    Description: Verify the result of MultiStepLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)
    expect_list = [[0.1], [0.01], [0.01], [0.001], [0.001]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(5):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_constant_lr(mode):
    """
    Feature: ConstantLR
    Description: Verify the result of ConstantLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.Adam(net.trainable_params(), 0.1)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)
    expect_list = [[0.05], [0.05], [0.05], [0.1], [0.1]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(5):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cyclic_lr(mode):
    """
    Feature: CyclicLR
    Description: Verify the result of CyclicLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.SGD(net.trainable_params(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    expect_list = [[0.010045], [0.01009], [0.010135], [0.01018], [0.010225]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(5):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cosine_annealing_warm_restarts(mode):
    """
    Feature: CosineAnnealingWarmRestarts
    Description: Verify the result of CosineAnnealingWarmRestarts
    Expectation: CosineAnnealingWarmRestarts
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.SGD(net.trainable_params(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2)
    expect_list = [[0.1], [0.09330127018922195], [0.07500000000000001], [0.05],
                   [0.025000000000000012], [0.006698729810778076],
                   [0.1], [0.09330127018922194], [0.07500000000000002], [0.05],
                   [0.02499999999999999], [0.006698729810778076]]

    class SchedNet(nn.Cell):
        def construct(self, global_step):
            scheduler.step(global_step)
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    iters = 3
    for epoch in range(4):
        for i in range(iters):
            current_lr = sched_net(epoch + i / iters)
            assert np.allclose(current_lr[0].asnumpy(), expect_list[epoch*iters+i])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cosine_annealing_lr(mode):
    """
    Feature: CosineAnnealingLR
    Description: Verify the result of CosineAnnealingLR
    Expectation: success
    """
    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    net = Net()
    optimizer = optim.SGD(net.trainable_params(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    expect_list = [[0.05], [0.0], [0.05], [0.1], [0.05], [0.0]]

    class SchedNet(nn.Cell):
        def construct(self):
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            return current_lr

    sched_net = SchedNet()
    for i in range(6):
        current_lr = sched_net()
        assert np.allclose(current_lr[0].asnumpy(), expect_list[i], 1e-6, 1e-6)
