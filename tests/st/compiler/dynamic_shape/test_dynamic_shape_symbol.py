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
"""Test dynamic shape with symbol"""

import os
import torch
import shutil
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
from tests.mark_utils import arg_mark


def check_ir_symbolic_shape(dir_path, file_multi_targets, expect_num=None):
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"'{dir_path}' is not a valid directory.")
    validate_files = sorted([
        f for f in os.listdir(dir_path)
        if 'validate' in f and os.path.isfile(os.path.join(dir_path, f))
    ])
    ir_validate_num = len(validate_files)
    expected_count = expect_num if expect_num is not None else len(file_multi_targets)
    assert ir_validate_num == expected_count, \
        f"Expect {expected_count} IR files, but found {ir_validate_num}: {validate_files}"
    for item in file_multi_targets:
        key, targets = item
        file_path = None
        if isinstance(key, int):
            filename = validate_files[key]
            file_path = os.path.join(dir_path, filename)
        elif isinstance(key, str):
            matched_files = [f for f in validate_files if key in f]
            filename = matched_files[0]
            file_path = os.path.join(dir_path, filename)
        else:
            raise TypeError("Key must be int (index) or str (substring in filename)")
        with open(file_path, 'r', encoding="utf-8") as f:
            content = f.read()
        for target_str in targets:
            assert target_str in content, \
                f"Target string '{target_str}' not found in IR file: {filename}"


def save_ir(ir_path):
    if os.path.exists(ir_path):
        shutil.rmtree(ir_path)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "1"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'validate'


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_dynamic_shape_symbol():
    """
    Feature: Dynamic shape
    Description: Test dynamic shape with ms.Symbol
    Expectation: No exception.
    """
    case_name = "test_dynamic_shape_symbol"
    ir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
    save_ir(ir_path)

    s1 = ms.Symbol(min=2, max=10)

    class Net(nn.Cell):
        @ms.jit(dynamic=1, backend="ms_backend")
        @ms.enable_dynamic(y=Tensor(shape=[s1, s1], dtype=ms.float32))
        def construct(self, x, y):
            return x * y

    class Mod(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    class GradOfAllInputs(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

        def construct(self, *inputs):
            grad_net = self.grad_op(self.net)
            return grad_net(*inputs)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    x1 = y1 = np.random.randn(3, 4).astype(np.float32)
    x2 = y2 = np.random.randn(2, 6).astype(np.float32)
    x3 = y3 = np.random.randn(6, 2).astype(np.float32)

    input_x_list = [x1, x2, x3]
    input_y_list = [y1, y2, y3]
    net = Net()
    net.set_grad()
    mod = Mod()

    for i in range(3):
        ms_x = Tensor(input_x_list[i])
        ms_y = Tensor(input_y_list[i])
        ms_out = net(ms_x, ms_y)
        pt_x = torch.from_numpy(input_x_list[i])
        pt_y = torch.from_numpy(input_y_list[i])
        pt_x.requires_grad = True
        pt_y.requires_grad = True
        pt_out = mod(pt_x, pt_y)
        assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 1e-4, 1e-4)

        out_np_shape = ms_out.shape
        if not out_np_shape:
            out_np = np.array(1).astype(np.float32)
        else:
            out_np = np.random.randn(*out_np_shape).astype(np.float32)
        grad_net = GradOfAllInputs(net)
        grad_net.set_train()
        grad_ms = grad_net(ms_x, ms_y, Tensor(out_np))

        output_grad = torch.from_numpy(out_np.copy())
        out = mod(pt_x, pt_y)
        out.backward(gradient=output_grad)
        grad_pt = [pt_x.grad, pt_y.grad]

        for i in range(2):
            input_grad_mindspore = grad_ms[i].asnumpy()
            input_grad_pytorch = grad_pt[i].numpy()
            assert np.allclose(input_grad_pytorch, input_grad_mindspore, 1e-4, 1e-4)

    check_ir_symbolic_shape(
        dir_path=ir_path,
        file_multi_targets=[
            (0,
             ["%para1_x: <Tensor[Float32], (3, 4)> : [3, 4]",
              "%para2_y: <Tensor[Float32], (-1, -1)> : [s1<[2,10]>, s2<[2,10]>]"]),
            (1,
             ["%para1_x: <Tensor[Float32], (2, 6)> : [2, 6]",
              "%para2_y: <Tensor[Float32], (-1, -1)> : [s3<[2,10]>, s4<[2,10]>]"]),
            (2,
             ["%para1_x: <Tensor[Float32], (-1, -1)> : [s5<[1,inf]>, s6<[1,inf]>]",
              "%para2_y: <Tensor[Float32], (-1, -1)> : [s7<[2,10]>, s8<[2,10]>]"])
        ],
        expect_num=3
    )
