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
# ==============================================================================
import os
import tempfile
import pytest
import scipy
import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.train.summary.summary_record import SummaryRecord
from tests.summary_utils import SummaryReader

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class AssignAddNet(nn.Cell):
    def __init__(self, para):
        super(AssignAddNet, self).__init__()
        self.para = Parameter(para, name="para")
        self.assign_add = P.AssignAdd()

    def construct(self, value):
        self.assign_add(self.para, value)
        return self.para


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_add():
    x = Tensor(1, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    expect = Tensor(3, dtype=mstype.int32)
    net = AssignAddNet(x)
    out = net(y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


class AssignSubNet(nn.Cell):
    def __init__(self, para):
        super(AssignSubNet, self).__init__()
        self.para = Parameter(para, name="para")
        self.assign_sub = P.AssignSub()

    def construct(self, value):
        self.assign_sub(self.para, value)
        return self.para


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_assign_sub():
    x = Tensor(3, dtype=mstype.int32)
    y = Tensor(2, dtype=mstype.int32)
    expect = Tensor(1, dtype=mstype.int32)
    net = AssignSubNet(x)
    out = net(y)
    np.testing.assert_array_equal(out.asnumpy(), expect.asnumpy())


class ScatterAddNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterAddNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_add = P.ScatterAdd()

    def construct(self, indices, updates):
        self.scatter_add(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_add():
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]), mstype.float32)
    expect = Tensor(np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]), mstype.float32)
    net = ScatterAddNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterSubNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterSubNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_sub = P.ScatterSub()

    def construct(self, indices, updates):
        self.scatter_sub(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_sub():
    input_x = Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]), mstype.float32)
    expect = Tensor(np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]), mstype.float32)
    net = ScatterSubNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterMulNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterMulNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_mul = P.ScatterMul()

    def construct(self, indices, updates):
        self.scatter_mul(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_mul():
    input_x = Tensor(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]), mstype.float32)
    expect = Tensor(np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]), mstype.float32)
    net = ScatterMulNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterDivNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterDivNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_div = P.ScatterDiv()

    def construct(self, indices, updates):
        self.scatter_div(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_div():
    input_x = Tensor(np.array([[6.0, 6.0, 6.0], [2.0, 2.0, 2.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 1]]), mstype.int32)
    updates = Tensor(np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]), mstype.float32)
    expect = Tensor(np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]), mstype.float32)
    net = ScatterDivNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterMaxNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterMaxNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_max = P.ScatterMax()

    def construct(self, indices, updates):
        self.scatter_max(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_max():
    input_x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]) * 88, mstype.float32)
    expect = Tensor(np.array([[88.0, 88.0, 88.0], [88.0, 88.0, 88.0]]), mstype.float32)
    net = ScatterMaxNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterMinNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterMinNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_min = P.ScatterMin()

    def construct(self, indices, updates):
        self.scatter_min(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_min():
    input_x = Tensor(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.ones([2, 2, 3]), mstype.float32)
    expect = Tensor(np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]), mstype.float32)
    net = ScatterMinNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterUpdateNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterUpdateNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_update = P.ScatterUpdate()

    def construct(self, indices, updates):
        self.scatter_update(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_update():
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([[[1.0, 2.2, 1.0], [2.0, 1.2, 1.0]], [[2.0, 2.2, 1.0], [3.0, 1.2, 1.0]]]), mstype.float32)
    expect = Tensor(np.array([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]]), mstype.float32)
    net = ScatterUpdateNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNdAddNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterNdAddNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_nd_add = P.ScatterNdAdd()

    def construct(self, indices, updates):
        self.scatter_nd_add(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_nd_add():
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    expect = Tensor(np.array([1, 10, 9, 4, 12, 6, 7, 17]), mstype.float32)
    net = ScatterNdAddNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNdSubNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterNdSubNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_nd_sub = P.ScatterNdSub()

    def construct(self, indices, updates):
        self.scatter_nd_sub(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_nd_sub():
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    expect = Tensor(np.array([1, -6, -3, 4, -2, 6, 7, -1]), mstype.float32)
    net = ScatterNdSubNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNdUpdateNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterNdUpdateNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_nd_update = P.ScatterNdUpdate()

    def construct(self, indices, updates):
        self.scatter_nd_update(self.input_x, indices, updates)
        return self.input_x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_nd_update():
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mstype.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), mstype.int32)
    updates = Tensor(np.array([1.0, 2.2]), mstype.float32)
    expect = Tensor(np.array([[1., 0.3, 3.6], [0.4, 2.2, -3.2]]), mstype.float32)
    net = ScatterNdUpdateNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class ScatterNonAliasingAddNet(nn.Cell):
    def __init__(self, input_x):
        super(ScatterNonAliasingAddNet, self).__init__()
        self.input_x = Parameter(input_x, name="para")
        self.scatter_non_aliasing_add = P.ScatterNonAliasingAdd()

    def construct(self, indices, updates):
        out = self.scatter_non_aliasing_add(self.input_x, indices, updates)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scatter_non_aliasing_add():
    input_x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mstype.float32)
    indices = Tensor(np.array([[2], [4], [1], [7]]), mstype.int32)
    updates = Tensor(np.array([6, 7, 8, 9]), mstype.float32)
    expect = Tensor(np.array([1.0, 10.0, 9.0, 4.0, 12.0, 6.0, 7.0, 17.0]), mstype.float32)
    net = ScatterNonAliasingAddNet(input_x)
    out = net(indices, updates)
    np.testing.assert_almost_equal(out.asnumpy(), expect.asnumpy())


class SummaryNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.scalar_summary = P.ScalarSummary()
        self.image_summary = P.ImageSummary()
        self.tensor_summary = P.TensorSummary()
        self.histogram_summary = P.HistogramSummary()

    def construct(self, image_tensor):
        self.image_summary("image", image_tensor)
        self.tensor_summary("tensor", image_tensor)
        self.histogram_summary("histogram", image_tensor)
        scalar = image_tensor[0][0][0][0]
        self.scalar_summary("scalar", scalar)
        return scalar


def train_summary_record(test_writer, steps):
    """Train and record summary."""
    net = SummaryNet()
    out_me_dict = {}
    for i in range(0, steps):
        image_tensor = Tensor(np.array([[[[i]]]]).astype(np.float32))
        out_put = net(image_tensor)
        test_writer.record(i)
        out_me_dict[i] = out_put.asnumpy()
    return out_me_dict


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_summary():
    with tempfile.TemporaryDirectory() as tmp_dir:
        steps = 2
        with SummaryRecord(tmp_dir) as test_writer:
            train_summary_record(test_writer, steps=steps)

            file_name = os.path.realpath(test_writer.full_file_name)
        with SummaryReader(file_name) as summary_writer:
            for _ in range(steps):
                event = summary_writer.read_event()
                tags = set(value.tag for value in event.summary.value)
                assert tags == {'tensor', 'histogram', 'scalar', 'image'}


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_igamma():
    class IGammaTest(nn.Cell):
        def __init__(self):
            super().__init__()
            self.igamma = nn.IGamma()

        def construct(self, x, a):
            return self.igamma(a=a, x=x)

    x = 4.22
    a = 2.29
    net = IGammaTest()
    out = net(Tensor(x, mstype.float32), Tensor(a, mstype.float32))
    expect = scipy.special.gammainc(a, x)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-5, atol=1e-5, equal_nan=True)
