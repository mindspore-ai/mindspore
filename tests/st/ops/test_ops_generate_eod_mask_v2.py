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
import pytest
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


def set_mode(mode):
    context.set_context(device_target="Ascend", save_graphs=False)
    if mode == "GE":
        context.set_context(mode=context.GRAPH_MODE,
                            jit_config={"jit_level": "O2"})
    elif mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE,
                            jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


class ForwardNet(nn.Cell):
    """
    ForwardNet
    """
    def __init__(self):
        super(ForwardNet, self).__init__()
        self.op = ops.auto_generate.GenerateEodMaskV2()
        self.cur_step = ms.Parameter(Tensor(-1., ms.int64))
        self.d_step = Tensor(1., ms.int64)

    def construct(self, input_tensor, ele_pos, seed, offset, start=0, steps=1, error_mode='cycle',
                  flip_mode='bitflip', multiply_factor=0., bit_pos=0, flip_probability=0.):
        self.cur_step = self.cur_step + self.d_step
        return self.op(input_tensor, ele_pos, self.cur_step, seed, offset, start, steps,
                       error_mode, flip_mode, multiply_factor, bit_pos, flip_probability)


def insert_gem_v2_in_backward(input_tensor, ele_pos, cur_step, seed, offset, start=0, steps=1, error_mode='cycle',
                              flip_mode='bitflip', multiply_factor=0., bit_pos=0, flip_probability=0.):
    op = ops.auto_generate.InsertGemV2InBackward()
    return op(input_tensor, ele_pos, cur_step, seed, offset, start, steps, error_mode, flip_mode,
              multiply_factor, bit_pos, flip_probability)


class BackwardNet(nn.Cell):
    """
    BackwardNet
    """
    def __init__(self):
        super(BackwardNet, self).__init__()
        self.cur_step = ms.Parameter(Tensor(-1., ms.int64))
        self.d_step = Tensor(1., ms.int64)

    def construct(self, input_tensor, ele_pos, seed, offset, start=0, steps=1, error_mode='cycle',
                  flip_mode='bitflip', multiply_factor=0., bit_pos=0, flip_probability=0.):
        self.cur_step = self.cur_step + self.d_step
        return ops.grad(insert_gem_v2_in_backward, grad_position=(0))(input_tensor, ele_pos,
                                                                      self.cur_step, seed, offset,
                                                                      start, steps, error_mode,
                                                                      flip_mode, multiply_factor,
                                                                      bit_pos, flip_probability)


def compare_result(src_val, out_val, multiply_factor, flip_mode):
    if flip_mode == 'bitflip':
        assert src_val == -out_val
    elif flip_mode == 'bitflip_designed':
        assert src_val != out_val
    else:
        assert src_val * multiply_factor == out_val


def run_generate_eod_mask_v2_on_step(data, ele_pos, start, steps, error_mode='specific',
                                     flip_mode='bitflip', multiply_factor=0., flip_probability=0.,
                                     bit_pos=0, changed_poses=(0,)):
    """
    Feature: Test GenerateEodMaskV2.
    Description: Test multi dtype inputs
    Expectation: Successful graph compilation.
    """
    seed = Tensor(0, ms.int64)
    offset = Tensor(0, ms.int64)
    for direction in ["Forward", "Backward"]:
        print(f"\nStart {direction} Testing, error_mode '{error_mode}', flip_mode '{flip_mode}', "
              f"flip_probability '{flip_probability}', ele_pos '{ele_pos}'")
        source_data = data.asnumpy()
        if direction == "Forward":
            net = ForwardNet()
        else:
            net = BackwardNet()
            source_data = np.ones(source_data.shape).astype(source_data.dtype)
        for i in range(10):
            out = net(data, Tensor(ele_pos, ms.int64), seed, offset,
                      start, steps, error_mode, flip_mode, multiply_factor,
                      bit_pos, flip_probability)
            print(f"loop {i} the output is:", out)
            if error_mode == "specific":
                if i in steps:
                    for pos in changed_poses:
                        compare_result(source_data.flatten()[pos], out.asnumpy().flatten()[pos],
                                       multiply_factor, flip_mode)
                else:
                    assert (source_data == out.asnumpy()).all()
            else:
                if i != 0 and i % steps[0] == 0:
                    for pos in changed_poses:
                        compare_result(source_data.flatten()[pos], out.asnumpy().flatten()[pos],
                                       multiply_factor, flip_mode)
                else:
                    assert (source_data == out.asnumpy()).all()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", ["GE", "KBK", "PYBOOST"])
def test_generate_eod_mask_v2(mode):
    """
    Feature: test op GenerateEodMoaskV2
    Description: test op GenerateEodMoaskV2
    Expectation: expect results
    """
    set_mode(mode)
    test_data = Tensor([[[0.1]], [[-0.2]]], dtype=ms.float32)

    # Test bit flip with 'bitflip' flip_mode on the first element on 0 bits on each step in 10 steps
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[1],
                                     error_mode='cycle', bit_pos=0, changed_poses=[0])

    # Test bit flip with 'bit_designed' flip_mode on the first element on the
    # first non-zero exponential bit on each step in 10 steps
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[1], error_mode='cycle',
                                     flip_mode="bitflip_designed", bit_pos=0, changed_poses=[0])

    # Test bit flip with 'multiply' flip_mode on the first element on each step in 10 steps
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[1], error_mode='cycle',
                                     flip_mode="multiply", multiply_factor=2., bit_pos=0, changed_poses=[0])

    # Test bit flip with 'multiply_max' flip_mode on the element with max abs value on each step in 10 steps
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[1], error_mode='cycle',
                                     flip_mode="multiply_max", multiply_factor=2., bit_pos=0, changed_poses=[1])

    # Test bit flip on the first element on 0 bits on specified steps in 10 steps
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[4, 5, 8],
                                     error_mode='specific', bit_pos=0, changed_poses=[0])

    # Test bit flip with 'bitflip' flip_mode on all elements with
    # 'flip_probability' on 0 bits on each step in 10 steps
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[1],
                                     error_mode='cycle', flip_probability=1.,
                                     bit_pos=0, changed_poses=[0, 1])
