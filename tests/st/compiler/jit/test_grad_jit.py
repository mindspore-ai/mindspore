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
"""Test grad jit prune"""
import os
import glob
import shutil
import subprocess
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops.functional import grad
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.train.model import Model
from tests.st.auto_parallel.utils.dataset_utils import FakeData
from tests.mark_utils import arg_mark


def check_ir(ir_name, ir_path, expect_dict, ir_num=0):
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
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "1"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_gradjit_prune_001():
    """
    Feature: gradjit
    Description: Test gradjit
    Expectation: No exception.
    """
    case_name = "test_gradjit_prune_001"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    save_ir(ir_path)

    class Net(nn.Cell):
        def construct(self, x, y, z):
            x = x / 1
            y = y / 1
            z = z - 1
            x[1] = 1 / Tensor([1])
            y[1] = 2 / Tensor([2])
            z[0] = 2 * Tensor([2])
            return x + y + z

    ms.set_context(mode=ms.PYNATIVE_MODE)
    x = np.ones([4, 8]).astype(np.float32)
    y = np.ones([4, 8]).astype(np.float32)
    z = np.ones([4, 8]).astype(np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y), Tensor(z))
    grad_out = grad(net, (0, 1))(Tensor(x), Tensor(y), Tensor(z))

    net.construct = ms.jit(net.construct)
    out_jit = net(Tensor(x), Tensor(y), Tensor(z))
    grad_out_jit = grad(net, (0, 1))(Tensor(x), Tensor(y), Tensor(z))

    assert np.allclose(out[0].asnumpy(), out_jit[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(out[1].asnumpy(), out_jit[1].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(grad_out[0].asnumpy(), grad_out_jit[0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(grad_out[1].asnumpy(), grad_out_jit[1].asnumpy(), 0.0001, 0.0001)

    check_ir("filtered_output_grad*.ir", ir_path, {"PrimFunc_Div(": 2, "TensorScatterUpdate(": 2})
    check_ir("opt_backward_[0-9]*.ir", ir_path, {"PrimFunc_Div(": 2, "TensorScatterUpdate(": 3})


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_jit_recompute_002():
    """
    Feature: gradjit
    Description: Test gradjit with recompute
    Expectation: No exception.
    """
    class Net1(nn.Cell):
        def __init__(self, has_bias=True):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=1, weight_init="ones",
                                bias_init='zeros', has_bias=has_bias)
            self.flatten = ops.Flatten()
            self.add = ops.Add()
            self.fc = nn.Dense(in_channels=12 * 32 * 32, out_channels=12,
                            weight_init='ones', bias_init='zeros', has_bias=True)

        def construct(self, x):
            x = self.conv(x)
            x = self.flatten(x)
            x = self.fc(x)
            x = self.add(x, x)
            return x

    class Conv2dAddReluMean1(nn.Cell):
        def __init__(self):
            super().__init__()
            self.block = Net1()
            self.relu = nn.ReLU()

        @ms.jit
        def construct(self, x):
            x = self.block(x)
            x = self.relu(x)
            return x

    def net_train(input_x, net, dataset):
        epoch_size = 4
        loss = SoftmaxCrossEntropyWithLogits(sparse=False)
        opt = Momentum(learning_rate=0.1, momentum=0.9, params=net.trainable_params())
        model = Model(net, loss, opt)
        model.train(epoch_size, dataset, dataset_sink_mode=False)
        out = model.predict(input_x)
        return out.asnumpy()

    ms.set_context(mode=ms.PYNATIVE_MODE)
    case_name = "test_grad_jit_recompute_002"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    save_ir(ir_path)

    seed = np.random.randint(2 ** 32)
    np.random.seed(seed)
    input_x = Tensor(np.random.randint(low=0, high=64,
                                       size=(16, 3, 32, 32)).astype(np.float32))
    dataset1 = FakeData(size=32, batch_size=16, image_size=(3, 32, 32), num_classes=12)
    dataset2 = FakeData(size=32, batch_size=16, image_size=(3, 32, 32), num_classes=12)
    net1 = Conv2dAddReluMean1()
    net1.block.recompute()
    net2 = Conv2dAddReluMean1()

    infer1 = net_train(input_x, net1, dataset1)
    infer2 = net_train(input_x, net2, dataset2)
    assert np.allclose(infer1, infer2, 0.0001, 0.0001)

    check_ir('opt_backward_[0-9]*.ir', ir_path, {"Conv2D(": 0}, 1)
    check_ir('opt_backward_[0-9]*.ir', ir_path, {"Conv2D(": 1}, 0)
    check_ir('opt_forward_[0-9]*.ir', ir_path, {"Conv2D(": 1}, 0)

    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
