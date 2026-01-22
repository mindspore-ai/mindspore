# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test recompute under grad jit """
import os
import subprocess
import glob
import shutil
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
from mindspore.graph.lazy_inline import lazy_inline
import mindspore.ops.operations as P
from mindspore import ops, nn, jit, context, recompute
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.train.model import Model
from tests.st.compiler.utils import FakeData
from tests.st.pi_jit.share.utils import allclose_nparray
from tests.mark_utils import arg_mark


class Grad(Cell):
    def __init__(self, net):
        super().__init__()
        self.grad = ops.GradOperation()
        self.net = net

    def construct(self, x):
        grad_net = self.grad(self.net)
        return grad_net(x)


class LeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()
        self.batch_size = 32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init="normal")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init="normal")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120, weight_init="normal", bias_init="zeros")
        self.fc2 = nn.Dense(120, 84, weight_init="normal", bias_init="zeros")
        self.fc3 = nn.Dense(84, 10, weight_init="normal", bias_init="zeros")

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_block_recompute1():
    """
    Feature: Sub cell recompute under gradjit
    Description: LeNet block is set recompute by the cell recompute api.
    Expectation: Run successfully.
    """

    class NetRecompute(Cell):
        def __init__(self):
            super().__init__()
            self.block = LeNet()
            self.block.recompute()

        @jit
        def construct(self, x):
            out = self.block(x)
            out = ops.Abs()(out)
            return out

    save_graphs_path = "./test_recompute_under_gradjit1"
    context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    x = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    net = NetRecompute()
    grad_net = Grad(net)
    grad_net(x)

    para = '= Conv2D(%'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_block_recompute1_with_func_api():
    """
    Feature: Sub cell recompute under gradjit
    Description: LeNet block is set recompute by the cell recompute api.
    Expectation: Run successfully.
    """

    class NetFunctional(Cell):
        def __init__(self):
            super().__init__()
            self.block = LeNet()

        @jit
        def construct(self, x):
            out = recompute(self.block, x)
            out = ops.Abs()(out)
            return out

    save_graphs_path = "./test_recompute_block_recompute1_with_func_api"
    context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    x = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    net = NetFunctional()
    grad_net = Grad(net)
    grad_net(x)

    para = '= Conv2D(%'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_block_recompute2():
    """
    Feature: Top cell recompute under gradjit
    Description: LeNet block is set recompute by the cell recompute api.
    Expectation: Run successfully.
    """

    class NetRecompute(Cell):
        def __init__(self):
            super().__init__()
            self.block = LeNet()
            self.recompute()

        @jit
        def construct(self, x):
            out = self.block(x)
            out = ops.Abs()(out)
            return out

    save_graphs_path = "./test_recompute_under_gradjit2"
    context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    x = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    net = NetRecompute()
    grad_net = Grad(net)
    grad_net(x)

    para = '= Conv2D(%'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "0"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_lazy_inline():
    """
    Feature: Recompute with lazyinline
    Description: Recompute with lazyinline
    Expectation: Run successfully.
    """
    class Block1(nn.Cell):
        def __init__(self, weight_shape=(32, 32)):
            super().__init__()
            self.mul = P.MatMul()
            self.add = P.Add()
            self.relu = P.ReLU()
            self.mul_weight = Parameter(Tensor(np.ones(weight_shape).astype(np.float32)), name="mul_weight")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.relu(out)
            out = self.add(out, out)
            return out

    class NetLazyInline(nn.Cell):
        @lazy_inline
        def __init__(self):
            super().__init__()
            self.blocks = nn.CellList()
            b = Block1()
            self.blocks.append(b)

        @jit(backend="ms_backend")
        def construct(self, x):
            out = x
            out = self.blocks[0](out)
            return out

    net1 = NetLazyInline()
    net2 = NetLazyInline()
    net1.blocks[0].recompute()
    x = Tensor(np.random.randint(low=0, high=64, size=(32, 32)).astype(np.float32))
    grad1 = ops.grad(net1)(x)
    grad2 = ops.grad(net2)(x)
    assert np.allclose(grad1, grad2)


def check_ir(ir_name, ir_path, expect_dict, ir_num):
    try:
        ir_files = sorted(glob.glob(os.path.join(ir_path, ir_name)))

        file = ir_files[ir_num]
        for key in expect_dict:
            cmd = f"grep '{key}' {file} | wc -l"
            output = subprocess.check_output(cmd, shell=True)
            output = str(output, 'utf-8').strip()
            assert int(output) == expect_dict[key]

    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')


def save_ir(ir_path):
    cmd = "rm -rf %s" % ir_path
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Remove ir path failed. Error message: ", e)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "1"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path


def net_train(input_x, net, dataset):
    epoch_size = 4
    loss = SoftmaxCrossEntropyWithLogits(sparse=False)
    opt = Momentum(learning_rate=0.1, momentum=0.9,
                   params=net.trainable_params())
    model = Model(net, loss, opt)
    model.train(epoch_size, dataset, dataset_sink_mode=False)
    out = model.predict(input_x)
    return out.asnumpy()


class Conv2dAddReluMean(nn.Cell):
    def __init__(self):
        super().__init__()
        self.block = Net()
        self.mean = P.ReduceMean(keep_dims=False)

    @jit(backend="ms_backend")
    def construct(self, x):
        x = self.block(x)
        x = self.mean(x, (2, 3))
        return x


class Net(nn.Cell):
    def __init__(self, has_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, weight_init="ones",
                              bias_init='zeros', has_bias=has_bias)
        self.add = P.Add()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.add(x, x)
        x = self.relu(x)
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_jit_recompute_001():
    """
    Feature: Recompute with net work train
    Description: Recompute with net work train
    Expectation: Run successfully.
    """
    case_name = "test_grad_jit_recompute_001"
    ir_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), case_name)
    save_ir(ir_path)

    seed = np.random.randint(2 ** 32)
    np.random.seed(seed)
    input_x = Tensor(np.random.randint(low=0, high=64,
                                       size=(16, 3, 32, 32)).astype(np.float32))
    dataset1 = FakeData(size=32, batch_size=16,
                        image_size=(3, 32, 32), num_classes=12)
    dataset2 = FakeData(size=32, batch_size=16,
                        image_size=(3, 32, 32), num_classes=12)
    net1 = Conv2dAddReluMean()
    net1.block.recompute()
    net2 = Conv2dAddReluMean()

    infer1 = net_train(input_x, net1, dataset1)
    infer2 = net_train(input_x, net2, dataset2)
    allclose_nparray(infer1, infer2, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_jit_recompute_003():
    """
    Feature: Recompute with net work train
    Description: Recompute with net work train
    Expectation: Run successfully.
    """
    class Block1(Cell):
        def __init__(self, weight_shape=(128, 128)):
            super().__init__()
            self.mul = P.MatMul()
            self.mul1 = P.MatMul()
            self.mul2 = P.MatMul()
            self.add = P.Add()
            self.relu = P.ReLU()
            self.relu1 = P.ReLU()
            self.softmax = P.Softmax(-1)
            self.softmax1 = P.Softmax(-1)
            self.gelu = P.GeLU()
            self.mul_weight = Parameter(
                Tensor(np.ones(weight_shape).astype(np.float32)), name="mul_weight")
            self.matmul_weight1 = Parameter(Tensor(np.ones((weight_shape[1], 2048)).astype(np.float32)),
                                            name="matmul_weight1")
            self.matmul_weight2 = Parameter(Tensor(np.ones((2048, weight_shape[1])).astype(np.float32)),
                                            name="matmul_weight2")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.mul1(out, self.matmul_weight1)
            out = self.mul2(out, self.matmul_weight2)
            out = self.add(out, out)
            out = self.relu(out)
            out = self.softmax(out)
            out = self.relu1(out)
            out = self.softmax1(out)
            out = self.gelu(out)
            return out

    class Net2(Cell):
        def __init__(self):
            super().__init__()
            self.blocks = nn.CellList()
            for _ in range(8):
                b = Block1()
                self.blocks.append(b)

        @jit(backend="ms_backend")
        def construct(self, x):
            out = x
            for i in range(8):
                out = self.blocks[i](out)
            return out

    case_name = "test_grad_jit_recompute_003"
    ir_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), case_name)
    save_ir(ir_path)

    seed = np.random.randint(2 ** 32)
    np.random.seed(seed)
    input_x = Tensor(np.random.randn(128, 128).astype(np.float32))

    net1 = Net2()
    for i in range(8):
        net1.blocks[i].recompute()
    net2 = Net2()
    dataset1 = FakeData(size=256, batch_size=128,
                        image_size=(128,), num_classes=128)
    dataset2 = FakeData(size=256, batch_size=128,
                        image_size=(128,), num_classes=128)
    infer1 = net_train(input_x, net1, dataset1)
    infer2 = net_train(input_x, net2, dataset2)
    allclose_nparray(infer1, infer2, 0.0001, 0.0001)
    # 第一个ir图中equiv开头的算子，第二张ir图中没有用到
    check_ir('opt_forward_[0-9]*.ir', ir_path, {"ReLU(": 16}, 0)
    check_ir('opt_backward_[0-9]*.ir', ir_path, {"ReLU(": 16}, 0)
    check_ir('opt_forward_[0-9]*.ir', ir_path, {"Softmax(": 16}, 0)
    check_ir('opt_forward_[0-9]*.ir', ir_path, {"Softmax(": 16}, 0)

    check_ir('opt_backward_[0-9]*.ir', ir_path, {"ReLU(": 0}, 1)
    check_ir('opt_backward_[0-9]*.ir', ir_path, {"ReLU(": 16}, 0)
    check_ir('opt_backward_[0-9]*.ir', ir_path, {"Softmax(": 0}, 1)
    check_ir('opt_backward_[0-9]*.ir', ir_path, {"Softmax(": 16}, 0)
