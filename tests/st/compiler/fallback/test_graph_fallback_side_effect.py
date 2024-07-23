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
""" test graph JIT Fallback runtime feature """

import os
import shutil
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore import jit
import mindspore.amp as amp
import mindspore.nn as nn
from mindspore.common.initializer import One, Normal
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


class UserDefinedNet:
    def __init__(self):
        self.value = 10

    def __call__(self, x):
        return self.value * x


class UNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

    def construct(self, x):
        out = x * self.para
        print("out:", out)
        out = self.net(x) + self.para
        self.para = 2 * x
        return out, self.para + 10


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_side_effect_assign():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """
    net = UNet(UserDefinedNet())
    x = np.array(10, np.float64)
    output = net(ms.Tensor(x))
    print("output:", output)
    assert output[0].asnumpy() == 102
    assert output[1].asnumpy() == 30


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_side_effect_dict():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')
            self.para2 = Parameter(Tensor(4, dtype=ms.float64), name='para2')

        def construct(self, x):
            out = x * self.para
            print("out:", out)
            x = {'a': Tensor(1, dtype=ms.float64), 'b': Tensor(2, dtype=ms.float64)}
            y = x.get('a') + out
            z = dict(a=y + self.para - self.para2)
            self.para = 2 * y
            return z, self.para + 2

    net = Net()
    x = np.array(10, np.float64)
    out = net(ms.Tensor(x))
    print("out:", out)
    assert out[0] == {'a': 19}
    assert out[1] == 44


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_side_effect_dict_2():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def construct(self, x):
            out = x * self.para
            x = {'a': Tensor(1, dtype=ms.float64), 'b': Tensor(2, dtype=ms.float64)}
            self.para = x.get('a') + out
            out = x.get('b') - self.para
            y = {'c': 3, 'b': 4, 'd': self.para + 1}
            x.update(y)
            return self.para + out, x

    net = Net()
    x = np.array(10, np.float64)
    out = net(ms.Tensor(x))
    print("out:", out)
    assert out[0] == 2
    assert out[1] == {'a': 1, 'b': 4, 'c': 3, 'd': 22}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_side_effect_nested_net():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    class Inner:
        def __init__(self):
            self.number = ms.Tensor(2, dtype=ms.float64)

        def act(self, x, y):
            return self.number * (x + y)

    @ms.jit_class
    class InnerNet:
        def __init__(self):
            self.inner = Inner()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def renew_para(self, x, y):
            self.para = x + y
            return self.para

    class NestedNet(ms.nn.Cell):
        @ms.jit
        def construct(self, x, y):
            out = InnerNet().inner.act(InnerNet().renew_para(x, y) + x, y)
            out = out + (InnerNet().renew_para(out, y) * 1)
            return out

    x = ms.Tensor(2, dtype=ms.float64)
    y = ms.Tensor(4, dtype=ms.float64)
    net = NestedNet()
    output = net(x, y)
    print("output:", output)
    assert output == 52


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_control_flow():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def construct(self, x):
            out = x * self.para
            x = {'a': Tensor(1, dtype=ms.float64), 'b': Tensor(2, dtype=ms.float64)}
            self.para = x.get('a')
            if self.para > 0:
                out = x.get('b') - self.para
            return self.para, out, x

    net = Net()
    x = np.array(10, np.float64)
    out = net(ms.Tensor(x))
    print("out:", out)
    assert out[0] == 1
    assert out[1] == 1
    assert out[2] == {'a': 1, 'b': 2}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_side_effect_asnumpy():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    @jit
    def loss_scale():
        loss_scaler = amp.DynamicLossScaler(scale_value=2 ** 10, scale_factor=2, scale_window=1)
        grads = (Tensor(np.array([np.log(-1), 1.0]).astype(np.float32)), Tensor(np.array([0.2]).astype(np.float32)))
        unscaled_grads = loss_scaler.unscale(grads)
        is_finite = amp.all_finite(unscaled_grads)
        loss_scaler.adjust(is_finite)
        param_value = loss_scaler.scale_value * 1
        return param_value.asnumpy()

    out = loss_scale()
    assert out == 512


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_side_effect_assign_1():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def construct(self, x):
            out = x * self.para
            self.para = 2 * x
            param_value = self.para * 1
            return out, param_value.asnumpy()

    net = Net()
    x = np.array(10, np.float64)
    out = net(ms.Tensor(x))
    assert out[1] == 20


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_side_effect_assign_2():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def construct(self, x):
            self.para = 2 * x
            param_value = self.para * 1
            return param_value.asnumpy()

    net = Net()
    x = np.array(10, np.float64)
    out = net(ms.Tensor(x))
    assert out == 20


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_side_effect_dict_3():
    """
    Feature: Fallback runtime side effect.
    Description: Test execution order in Fallback runtime.
    Expectation: No error.
    """
    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.para = Parameter(Tensor(2, dtype=ms.float64), name='para')

        def construct(self, input_x):
            self.para = input_x + 1
            x = {'a': Tensor(1, dtype=ms.float64), 'b': self.para}
            self.para = x.get('a') * 2
            y = x.get('b') + self.para * 2
            self.para = self.para + 10
            z = {'c': 3, 'b': self.para * 2, 'd': self.para + 1}
            return x.get('b') + 1, y, z


    net = Net()
    x = np.array(10, np.float64)
    out = net(ms.Tensor(x))
    assert out[0] == 13
    assert out[1] == 6
    assert out[2] == {'c': 3, 'b': 24, 'd': 13}


class PrintPyExecuteNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def construct(self, x):
        out = x * x
        print("out1:", out)
        out = self.net(x) + out
        print("out2:", out)
        return out


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_print_pyexecute():
    """
    Feature: Side effect in Fallback runtime.
    Description: Side effect in Fallback runtime.
    Expectation: No error.
    """
    net = PrintPyExecuteNet(UserDefinedNet())
    x = np.array([10], np.float64)
    output = net(ms.Tensor(x))
    print(output)
    assert output == 200


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_dtype_is_cond():
    """
    Feature: Side effect in Fallback runtime.
    Description: Side effect in Fallback runtime.
    Expectation: No error.
    """
    @ms.jit
    def func(x):
        dtype = x.dtype
        if dtype:
            return 0
        return 1

    x = ms.Tensor(True)
    out = func(x)
    assert out == 0


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_if_after_for_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class UserDefinedNet2:
        def __init__(self):
            self.x = np.array([1, 2])
            self.y = np.array([3, 4])
            self.z = np.array([1, 2, 3, 4])

        def __call__(self, x):
            return self.x

    class PyExecuteNet(ms.nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self):
            x = self.net.x
            y = self.net.y
            z = self.net.z
            if len(x) + len(y) == len(z):
                return Tensor(y)
            return Tensor(z)

    net = PyExecuteNet(UserDefinedNet2())
    output = net()
    assert (output.asnumpy() == [3, 4]).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_save_load():
    """
    Feature: JIT Fallback
    Description: Test fallback with side effect operate from third-party module.
    Expectation: No exception.
    """
    class LinearNet(nn.Cell):
        def __init__(self, path_file):
            super(LinearNet, self).__init__()
            self.path_file = path_file
            self.dense = nn.Dense(10, 1)

        def construct(self, x):
            x = self.dense(x)
            np.save(self.path_file, x.asnumpy())
            np.load('/tmp/test_fallback_save_load/tensor1.npy')
            return x

    file_name = 'tensor1'
    path = f"/tmp/test_fallback_save_load/"
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)
    model = LinearNet(os.path.join(path, file_name))
    input_x = Tensor(shape=(1, 10), dtype=ms.float32, init=One())
    output = model(input_x)
    print("output:", output)
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_save_load_with_annotation():
    """
    Feature: JIT Fallback
    Description: Test fallback with side effect operate from third-party module.
    Expectation: No exception.
    """
    class LinearNet3(nn.Cell):
        def __init__(self, path_file):
            super(LinearNet3, self).__init__()
            self.path_file = path_file
            self.dense = nn.Dense(10, 1)

        def construct(self, x):
            x = self.dense(x)
            np.save(self.path_file, x.asnumpy())
            np.load('/tmp/test_fallback_save_load_with_annotation/tensor2.npy')  # @jit.typing: side_effect
            return x

    file_name = 'tensor2'
    path = f"/tmp/test_fallback_save_load_with_annotation/"
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    os.makedirs(path)
    model = LinearNet3(os.path.join(path, file_name))
    input_x = Tensor(shape=(1, 10), dtype=ms.float32, init=One())
    output = model(input_x)
    print("output:", output)
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


class TransformerNet(nn.Cell):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.transformer = nn.Transformer(d_model=10, nhead=2)
        self.dense = nn.Dense(10, 1)

    def construct(self, x, tgt):
        transformer_out = self.transformer(x, tgt)
        dense_output = self.dense(transformer_out[-1])
        np.save(self.file, dense_output.asnumpy())
        return dense_output.asnumpy(), Tensor(np.array([1, 2]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_save_with_side_effect():
    """
    Feature: JIT Fallback
    Description: Test fallback with side effect operate from third-party module.
    Expectation: No exception.
    """
    file_path = './Tensor'
    model = TransformerNet(file_path)
    input1_1 = Tensor(shape=(20, 32, 10), dtype=ms.float32, init=One())
    input1_2 = Tensor(shape=(20, 32, 10), dtype=ms.float32, init=Normal())

    dense_output, tensor_out = model(input1_1, input1_2)
    assert np.allclose(dense_output, np.load('./Tensor.npy'))
    os.remove('./Tensor.npy')
    assert (tensor_out.asnumpy() == [1, 2]).all()
