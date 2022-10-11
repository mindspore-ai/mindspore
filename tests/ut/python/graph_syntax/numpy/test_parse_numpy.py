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
""" test_parse_numpy """
import pytest
import numpy as np
from mindspore import nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_use_numpy_constant():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            ret = np.pi
            return ret

    net = Net()
    output = net()
    assert np.allclose(output, np.pi)


def test_use_numpy_method():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            ret = np.linspace(1, 10, 4)
            return ret

    net = Net()
    # Not raise NotImplementedError('Mindspore not supports to use the numpy ...') any more,
    # but raise RuntimeError('Should not use Python object in runtime...'), after support JIT Fallback.
    with pytest.raises(RuntimeError) as err:
        net()
    assert "Should not use Python object in runtime" in str(err.value)


def test_use_numpy_module():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()

        def construct(self):
            ret = np.random.randint(0, 10, [1, 10])
            return ret

    net = Net()
    # Not raise NotImplementedError('Mindspore not supports to use the numpy ...') any more,
    # but raise RuntimeError('Should not use Python object in runtime...'), after support JIT Fallback.
    with pytest.raises(RuntimeError) as err:
        net()
    assert "Should not use Python object in runtime" in str(err.value)
