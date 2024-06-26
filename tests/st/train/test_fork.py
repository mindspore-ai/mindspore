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
""" test fork. """
import platform
import pytest
import mindspore as ms
import mindspore.ops as ops
import mindspore.multiprocessing as mp
import mindspore.ops.functional as F
import numpy as np
from tests.mark_utils import arg_mark


def subprocess(mode, i, q):
    ms.set_context(mode=mode)
    x = q.get()
    y = ops.log(x)
    assert np.allclose(y.asnumpy(), np.log(2), 1e-3), "subprocess id:{i}"


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fork(mode):
    """
    Feature: Fork test
    Description: Test multiprocessing with fork
    Expectation: No exception
    """
    if platform.system() != 'Linux':
        return
    ms.set_context(mode=mode)
    x = ms.Tensor(2, dtype=ms.float32)
    y = ops.log(x)
    assert np.allclose(y.asnumpy(), np.log(2), 1e-3)

    mp.set_start_method('fork', force=True)
    processes = []
    for i in range(4):
        q = mp.Queue()
        p = mp.Process(target=subprocess, args=(mode, i, q))
        p.start()
        processes.append((p, i, q))

    for (p, i, q) in processes:
        q.put(ms.Tensor(2, dtype=ms.float32))
        p.join(5) # timeout:5s
        assert p.exitcode == 0, f"child process idx:{i}, exitcode:{p.exitcode}"


def child_process(x):
    return ops.log(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_multiprocess_pool(mode):
    """
    Feature: Fork test
    Description: Test multiprocessing with fork
    Expectation: No exception
    """
    if platform.system() != 'Linux':
        return
    ms.set_context(mode=mode)
    x = ms.Tensor(2, dtype=ms.float32)
    y = ops.log(x)
    assert np.allclose(y.asnumpy(), np.log(2), 1e-3)

    mp.set_start_method('fork', force=True)
    with mp.Pool(processes=2) as pool:
        inputs = [ms.Tensor(2.0), ms.Tensor(2.0)]
        outputs = pool.map(child_process, inputs)
        assert np.allclose(outputs[0].asnumpy(), np.log(2), 1e-3)
        assert np.allclose(outputs[1].asnumpy(), np.log(2), 1e-3)


class Net(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 1

    def construct(self, x, y):
        for k in range(1):
            if x != 1:
                for _ in range(1):
                    y = k * x
                    y = self.a + y
                    if x > 5:
                        break
            if x == 5:
                for _ in range(1):
                    y = self.a - y
                    if x == y:
                        continue
        return x + y

def childprocess(mode, i, q):
    ms.set_context(mode=mode)
    x = np.array([-1], np.float32)
    y = np.array([2], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(ms.Tensor(x), ms.Tensor(y))
    assert np.allclose(fgrad[0].asnumpy(), np.array([1.]), 1e-3)
    assert np.allclose(fgrad[1].asnumpy(), np.array([0.]), 1e-3)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fork_subgraphs(mode):
    """
    Feature: Fork test
    Description: Test multiprocessing with fork when net has subgraphs
    Expectation: No exception
    """
    if platform.system() != 'Linux':
        return
    ms.set_context(mode=mode)
    x = np.array([-1], np.float32)
    y = np.array([2], np.float32)
    net = Net()
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(ms.Tensor(x), ms.Tensor(y))
    assert np.allclose(fgrad[0].asnumpy(), np.array([1.]), 1e-3)
    assert np.allclose(fgrad[1].asnumpy(), np.array([0.]), 1e-3)

    mp.set_start_method('fork', force=True)
    processes = []
    for i in range(4):
        q = mp.Queue()
        p = mp.Process(target=childprocess, args=(mode, i, q))
        p.start()
        processes.append((p, i, q))

    for (p, i, q) in processes:
        q.put(ms.Tensor(2, dtype=ms.float32))
        p.join(5) # timeout:5s
        assert p.exitcode == 0, f"child process idx:{i}, exitcode:{p.exitcode}"
