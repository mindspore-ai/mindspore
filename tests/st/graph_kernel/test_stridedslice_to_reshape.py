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
from tests.mark_utils import arg_mark
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, new_axis_mask, shrink_axis_mask):
        super(Net, self).__init__()
        self.stridedslice = P.StridedSlice(new_axis_mask=new_axis_mask, shrink_axis_mask=shrink_axis_mask)

    def construct(self, x, begin, end, slices):
        return self.stridedslice(x, begin, end, slices)


def get_output(x, begin, end, slices, new_axis_mask, shrink_axis_mask, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net(new_axis_mask, shrink_axis_mask)
    output = net(x, begin, end, slices)
    return output


def compare_stridedslice_result(shape, dtype, axis_mask, mask_type):
    x = Tensor(np.random.random(shape).astype(dtype))
    begin = (0,) * len(shape)
    end = shape
    slices = (1,) * len(shape)
    if mask_type == "new_axis_mask":
        expect = get_output(x, begin, end, slices, axis_mask, 0, False)
        output = get_output(x, begin, end, slices, axis_mask, 0, True)
        expect_np = expect.asnumpy().copy()
        output_np = output.asnumpy().copy()
        assert np.allclose(expect_np, output_np, 0.0001, 0.0001)
    elif mask_type == "shrink_axis_mask":
        expect = get_output(x, begin, end, slices, 0, axis_mask, False)
        output = get_output(x, begin, end, slices, 0, axis_mask, True)
        expect_np = expect.asnumpy().copy()
        output_np = output.asnumpy().copy()
        assert np.allclose(expect_np, output_np, 0.0001, 0.0001)
    else:
        raise ValueError('mask_type must be "new_axis_mask" or "shrink_axis_mask"')


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_stridedslice_to_reshape_gpu():
    """
    Feature: Test stridedslice replacement in arithmetic_simplify pass.
    Description: Verify the correctness of the replacement, all stridedslices should be replaced with
    reshape in arithmetic_simplify pass.
    Expectation: No exception
    """
    context.set_context(mode=context.GRAPH_MODE)
    compare_stridedslice_result((1, 255, 256), np.float32, 0, "new_axis_mask")
    compare_stridedslice_result((1, 255, 256), np.float32, 0, "shrink_axis_mask")
    compare_stridedslice_result((1, 255, 256), np.float32, 1, "new_axis_mask")
    compare_stridedslice_result((1, 255, 256), np.float32, 1, "shrink_axis_mask")
    compare_stridedslice_result((255, 1, 256), np.float32, 2, "shrink_axis_mask")
