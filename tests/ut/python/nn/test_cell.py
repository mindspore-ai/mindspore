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
""" test cell """
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.api import _executor
from ..ut_filter import non_graph_engine


class ModA(nn.Cell):
    def __init__(self, tensor):
        super(ModA, self).__init__()
        self.weight = Parameter(tensor, name="weight")

    def construct(self, *inputs):
        pass


class ModB(nn.Cell):
    def __init__(self, tensor):
        super(ModB, self).__init__()
        self.weight = Parameter(tensor, name="weight")

    def construct(self, *inputs):
        pass


class ModC(nn.Cell):
    def __init__(self, ta, tb):
        super(ModC, self).__init__()
        self.mod1 = ModA(ta)
        self.mod2 = ModB(tb)

    def construct(self, *inputs):
        pass


class Net(nn.Cell):
    """ Net definition """
    name_len = 4
    cells_num = 3

    def __init__(self, ta, tb):
        super(Net, self).__init__()
        self.mod1 = ModA(ta)
        self.mod2 = ModB(tb)
        self.mod3 = ModC(ta, tb)

    def construct(self, *inputs):
        pass


class Net2(nn.Cell):
    def __init__(self, ta, tb):
        super(Net2, self).__init__(auto_prefix=False)
        self.mod1 = ModA(ta)
        self.mod2 = ModB(tb)
        self.mod3 = ModC(ta, tb)

    def construct(self, *inputs):
        pass


class ConvNet(nn.Cell):
    """ ConvNet definition """
    image_h = 224
    image_w = 224
    output_ch = 64

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, ConvNet.output_ch, kernel_size=7, stride=2, pad_mode="pad", padding=3)
        self.bn1 = nn.BatchNorm2d(ConvNet.output_ch)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(
            int(ConvNet.image_h*ConvNet.image_w*ConvNet.output_ch/(4*4)),
            num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def test_basic():
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    n = Net(ta, tb)
    names = list(n.parameters_dict().keys())
    assert len(names) == n.name_len
    assert names[0] == "mod1.weight"
    assert names[1] == "mod2.weight"
    assert names[2] == "mod3.mod1.weight"
    assert names[3] == "mod3.mod2.weight"


def test_parameter_name():
    """ test_parameter_name """
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    n = Net(ta, tb)
    names = []
    for m in n.parameters_and_names():
        if m[0]:
            names.append(m[0])
    assert names[0] == "mod1.weight"
    assert names[1] == "mod2.weight"
    assert names[2] == "mod3.mod1.weight"
    assert names[3] == "mod3.mod2.weight"


def test_cell_name():
    """ test_cell_name """
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    n = Net(ta, tb)
    n.insert_child_to_cell('modNone', None)
    names = []
    for m in n.cells_and_names():
        if m[0]:
            names.append(m[0])
    assert names[0] == "mod1"
    assert names[1] == "mod2"
    assert names[2] == "mod3"
    assert names[3] == "mod3.mod1"
    assert names[4] == "mod3.mod2"


def test_cells():
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    n = Net(ta, tb)
    ch = list(n.cells())
    assert len(ch) == n.cells_num


def test_exceptions():
    """ test_exceptions """
    t = Tensor(np.ones([2, 3]))

    class ModError(nn.Cell):
        def __init__(self, tensor):
            self.weight = Parameter(tensor, name="weight")
            super(ModError, self).__init__()

        def construct(self, *inputs):
            pass

    with pytest.raises(AttributeError):
        ModError(t)

    class ModError1(nn.Cell):
        def __init__(self, tensor):
            super().__init__()
            self.weight = Parameter(tensor, name="weight")
            self.weight = None
            self.weight = ModA(tensor)

        def construct(self, *inputs):
            pass

    with pytest.raises(TypeError):
        ModError1(t)

    class ModError2(nn.Cell):
        def __init__(self, tensor):
            super().__init__()
            self.mod = ModA(tensor)
            self.mod = None
            self.mod = tensor

        def construct(self, *inputs):
            pass

    with pytest.raises(TypeError):
        ModError2(t)

    m = nn.Cell()
    with pytest.raises(NotImplementedError):
        m.construct()


def test_del():
    """ test_del """
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    n = Net(ta, tb)
    names = list(n.parameters_dict().keys())
    assert len(names) == n.name_len
    del n.mod1
    names = list(n.parameters_dict().keys())
    assert len(names) == n.name_len - 1
    with pytest.raises(AttributeError):
        del n.mod1.weight
    del n.mod2.weight
    names = list(n.parameters_dict().keys())
    assert len(names) == n.name_len - 2
    with pytest.raises(AttributeError):
        del n.mod


def test_add_attr():
    """ test_add_attr """
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    p = Parameter(ta, name="weight")
    m = nn.Cell()
    m.insert_param_to_cell('weight', p)

    with pytest.raises(TypeError):
        m.insert_child_to_cell("network", p)

    with pytest.raises(KeyError):
        m.insert_param_to_cell('', p)
    with pytest.raises(KeyError):
        m.insert_param_to_cell('a.b', p)
    m.insert_param_to_cell('weight', p)
    with pytest.raises(KeyError):
        m.insert_child_to_cell('', ModA(ta))
    with pytest.raises(KeyError):
        m.insert_child_to_cell('a.b', ModB(tb))

    with pytest.raises(TypeError):
        m.insert_child_to_cell('buffer', tb)
    with pytest.raises(TypeError):
        m.insert_param_to_cell('w', ta)
    with pytest.raises(TypeError):
        m.insert_child_to_cell('m', p)

    class ModAddCellError(nn.Cell):
        def __init__(self, tensor):
            self.mod = ModA(tensor)
            super().__init__()

        def construct(self, *inputs):
            pass

    with pytest.raises(AttributeError):
        ModAddCellError(ta)


def test_train_eval():
    m = nn.Cell()
    assert not m.training
    m.set_train()
    assert m.training
    m.set_train(False)
    assert not m.training


def test_stop_update_name():
    ta = Tensor(np.ones([2, 3]))
    tb = Tensor(np.ones([1, 4]))
    n = Net2(ta, tb)
    names = list(n.parameters_dict().keys())
    assert names[0] == "weight"
    assert names[1] == "mod1.weight"
    assert names[2] == "mod2.weight"


class ModelName(nn.Cell):
    def __init__(self, tensor):
        super(ModelName, self).__init__()
        self.w2 = Parameter(tensor, name="weight")
        self.w1 = Parameter(tensor, name="weight")
        self.w3 = Parameter(tensor, name=None)
        self.w4 = Parameter(tensor, name=None)

    def construct(self, *inputs):
        pass


def test_cell_names():
    ta = Tensor(np.ones([2, 3]))
    mn = ModelName(ta)
    with pytest.raises(ValueError):
        _executor.compile(mn)
