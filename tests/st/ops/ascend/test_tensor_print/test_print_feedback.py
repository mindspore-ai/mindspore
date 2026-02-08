# Copyright 2024 Huawei Technologies Co., Ltd
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

import os
import numpy as np
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.ops.operations as op
from mindspore import nn, Tensor


class PrintVeryTimes(nn.Cell):
    def __init__(self):
        super().__init__()
        self.print = op.Print()

    def construct(self, x11, y11, z1, a1, b1, c1, x1, y1):
        while x1 > y1:
            self.print("m_print", x11)
            self.print("m_print_2:", y11)
            self.print("m_print_4:", z1)
            self.print("m_print_6:", a1)
            self.print("m_print_8:", b1)
            self.print("m_print_10:", c1)
            self.print("m_print", x11)
            self.print("m_print_2:", y11)
            self.print("m_print_4:", z1)
            self.print("m_print_6:", a1)
            self.print("m_print_8:", b1)
            self.print("m_print_10:", c1)
            self.print("m_print", x11)
            self.print("m_print_2:", y11)
            self.print("m_print_4:", z1)
            self.print("m_print_6:", a1)
            self.print("m_print_8:", b1)
            self.print("m_print_10:", c1)
            self.print("m_print_36:", x11, "m_print_38:", y11, "m_print_40:", z1, "m_print_42:",
                       a1, "m_print_44:", b1, "m_print_46:", c1,)
            self.print("m_print_48:", x11, "m_print_50:", y11, "m_print_52:", z1, "m_print_54:",
                       a1, "m_print_56:", b1, "m_print_58:", c1,)
            self.print("m_print_60:", x11, "m_print_62:", y11, "m_print_64:", z1, "m_print_66:",
                       a1, "m_print_68:", b1, "m_print_70:", c1,)
            self.print("m_print_72:", x11, "m_print_74:", y11, "m_print_76:", z1, "m_print_78:",
                       a1, "m_print_80:", b1, "m_print_82:", c1,)
            y1 = y1 + 1
        return y1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_print_tensor_many_times():
    """
    Feature: tensor print feedback to device operator.
    Description: test tensor print feedback to device operator.
    Expectation: the result match with expected result.
    """
    os.environ['MS_DUMP_SLICE_SIZE'] = '2048'
    os.environ['MS_DUMP_WAIT_TIME'] = '53'
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    print_file = os.getcwd() + "/test_print_tensor_many_times.pb"
    ms.set_context(print_file_path=print_file)
    x = np.full((1024, 512, 7, 7), 1.22222121, np.float32)
    y = np.full((1024, 512, 7, 7), 1.22222121, np.float16)
    a = np.full((1024, 512, 7, 7), 12222, np.int32)
    b = np.full((1024, 512, 7, 7), False, np.bool_)
    c = np.full((1024, 512, 7, 7), 1.222222222222, np.float64)
    input1 = Tensor(x)
    input2 = Tensor(y)
    input3 = Tensor(127, ms.uint32)
    input4 = Tensor(a)
    input5 = Tensor(b)
    input6 = Tensor(c)
    x1 = Tensor(10, ms.int32)
    y1 = Tensor(0, ms.int32)
    net = PrintVeryTimes()
    out = net(input1, input2, input3, input4, input5, input6, x1, y1)
    os.system(f'rm -f {print_file}')
    assert out.asnumpy() == 10


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=6144.01)
def test_print_tensor_many_times2():
    """
    Feature: tensor print feedback to device operator.
    Description: test tensor print feedback to device operator.
    Expectation: the result match with expected result.
    """
    class PrintManyTimes(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = op.Print()

        def construct(self, x11, y11, x1, y1):
            while x1 > y1:
                self.print("====================current loop", y1, x1)
                self.print("m_print_1", x11)
                self.print("m_print_2:", y11)
                y1 = y1 + 1
            return y1

    os.environ['MS_DUMP_SLICE_SIZE'] = '2048'
    os.environ['MS_DUMP_WAIT_TIME'] = '53'
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    x = np.full((1024, 512, 256, 4), 1.22222121, np.float32)
    y = np.full((1024, 512, 256, 4), 1.22222121, np.float16)
    input1 = Tensor(x)
    input2 = Tensor(y)
    loop_cnt = 5
    x1 = Tensor(loop_cnt, ms.int32)
    y1 = Tensor(0, ms.int32)
    net = PrintManyTimes()
    out = net(input1, input2, x1, y1)
    assert out.asnumpy() == loop_cnt
