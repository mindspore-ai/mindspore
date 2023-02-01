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
""" test_tensor_slice """
import numpy as np
import pytest

from mindspore import Tensor
from mindspore import Parameter
from mindspore import context
from mindspore import dtype as mstype
from mindspore.nn import Cell
from mindspore.common.parameter import ParameterTuple
from mindspore.ops import composite as C


grad_by_list_with_sens = C.GradOperation(get_by_list=True, sens_param=True)


def setup_module():
    context.set_context(mode=context.PYNATIVE_MODE)


class NetWorkSlicePositive(Cell):
    def __init__(self):
        super(NetWorkSlicePositive, self).__init__()
        self.tensor_ret0 = Tensor(np.ones([1, 2, 3], np.int32))
        self.tensor_ret1 = Tensor(np.ones([4, 8, 10], np.int32))
        self.tensor_ret2 = Tensor(np.ones([6, 8, 10], np.int32))
        self.tensor_ret3 = Tensor(np.ones([3, 8, 10], np.int32))

    def construct(self, tensor):
        ret0 = tensor[3:4:1, 1:5:2, 3:6:1] + self.tensor_ret0
        ret1 = tensor[-6:4:1, 0:8:1, ::1] + self.tensor_ret1
        ret2 = tensor[::, ::, ::] + self.tensor_ret2
        ret3 = tensor[::2] + self.tensor_ret3
        return ret0, ret1, ret2, ret3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice_positive():
    net = NetWorkSlicePositive()
    input_np = np.arange(6*8*10).reshape(6, 8, 10).astype(np.int32)
    input_0 = Tensor(input_np)
    output0, output1, output2, output3 = net(input_0)
    assert np.all(output0.asnumpy() == input_np[3:4:1, 1:5:2, 3:6:1] + np.ones([1, 2, 3]))
    assert np.all(output1.asnumpy() == input_np[-6:4:1, 0:8:1, ::1] + np.ones([4, 8, 10]))
    assert np.all(output2.asnumpy() == input_np[::, ::, ::] + np.ones([6, 8, 10]))
    assert np.all(output3.asnumpy() == input_np[::2] + np.ones([3, 8, 10]))


class NetWorkSliceEllipsis(Cell):
    def __init__(self):
        super(NetWorkSliceEllipsis, self).__init__()
        self.tensor_ret0 = Tensor(np.ones([2, 7, 8], np.int32))
        self.tensor_ret1 = Tensor(np.ones([6, 7, 8, 9], np.int32))
        self.tensor_ret2 = Tensor(np.ones([1, 6, 7, 8, 9], np.int32))

    def construct(self, tensor):
        ret0 = tensor[0:4:2, ..., 1] + self.tensor_ret0
        ret1 = tensor[...] + self.tensor_ret1
        ret2 = tensor[None] + self.tensor_ret2
        ret3 = tensor[True] + self.tensor_ret2
        return ret0, ret1, ret2, ret3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_slice_ellipsis():
    net = NetWorkSliceEllipsis()
    input_np = np.arange(6*7*8*9).reshape(6, 7, 8, 9).astype(np.int32)
    input_0 = Tensor(input_np)
    output0, output1, output2, output3 = net(input_0)
    assert np.all(output0.asnumpy() == input_np[0:4:2, ..., 1] + np.ones([2, 7, 8]))
    assert np.all(output1.asnumpy() == input_np[...] + np.ones([6, 7, 8, 9]))
    assert np.all(output2.asnumpy() == input_np[None] + np.ones([6, 7, 8, 9]))
    assert np.all(output3.asnumpy() == input_np[True] + np.ones([1, 6, 7, 8, 9]))


class NetWorkReduceDimension(Cell):
    def __init__(self):
        super(NetWorkReduceDimension, self).__init__()
        self.tensor_ret1 = Tensor(np.ones([3, 10], np.int32))
        self.tensor_ret2 = Tensor(np.ones([6, 8], np.int32))
        self.tensor_ret3 = Tensor(np.array(8, np.int32))
        self.tensor_ret4 = Tensor(np.ones([8, 10], np.int32))

    def construct(self, tensor):
        ret1 = tensor[::2, 1, ::1] + self.tensor_ret1
        ret2 = tensor[::, ::, 0] + self.tensor_ret2
        ret3 = tensor[3, 2, 5] + self.tensor_ret3
        ret4 = tensor[1] + self.tensor_ret4
        return ret1, ret2, ret3, ret4


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reduce_dimension():
    net = NetWorkReduceDimension()
    input_np = np.arange(6*8*10).reshape(6, 8, 10).astype(np.int32)
    input_0 = Tensor(input_np)
    output1, output2, output3, output4 = net(input_0)
    assert np.all(output1.asnumpy() == input_np[::2, 1, ::1] + np.ones([3, 10]))
    assert np.all(output2.asnumpy() == input_np[::, ::, 0] + np.ones([6, 8]))
    assert np.all(output3.asnumpy() == input_np[3, 2, 5] + np.array(8, np.int32))
    assert np.all(output4.asnumpy() == input_np[1] + np.ones([8, 10]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
class NetWorkSliceStep(Cell):
    def __init__(self):
        super(NetWorkSliceStep, self).__init__()
        self.tensor_ret1 = Tensor(np.ones([6, 5, 10], np.int32))
        self.tensor_ret2 = Tensor(np.ones([3, 5, 5], np.int32))

    def construct(self, tensor):
        ret1 = tensor[::1, -5::, ::-1] + self.tensor_ret1
        ret2 = tensor[::2, -5::, ::2] + self.tensor_ret2
        return ret1, ret2


@pytest.mark.level1
# ascend op stridedslice has bug, and has not been fixed.
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_step_negative():
    net = NetWorkSliceStep()
    input_np = np.arange(6*8*10).reshape(6, 8, 10).astype(np.int32)
    input_0 = Tensor(input_np)
    output1, output2 = net(input_0)
    assert np.all(output1.asnumpy() == input_np[::1, -5::, ::-1] + np.ones([6, 5, 10]))
    assert np.all(output2.asnumpy() == input_np[::2, -5::, ::2] + np.ones([3, 5, 5]))


class TensorGetItemByThreeTensors(Cell):
    def __init__(self):
        super(TensorGetItemByThreeTensors, self).__init__()
        self.const0 = Tensor(np.ones((4, 5, 8, 10)), mstype.int32)
        self.const1 = Tensor(np.ones((3, 4, 5, 10)), mstype.int32)
        self.const2 = Tensor(np.ones((5, 3, 4, 5)), mstype.int32)

    def construct(self, x, index_0, index_1, index_2):
        ret0 = x[index_0] + self.const0
        ret1 = x[index_0, index_1] + self.const1
        ret2 = x[index_0, index_1, index_2] + self.const2
        return ret0, ret1, ret2


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_getitem_by_tensors():
    """This testcase may encounter a sync stream error occasionally"""
    net = TensorGetItemByThreeTensors()
    input_x = np.arange(6*8*10).reshape(6, 8, 10).astype(np.int32)
    index_0 = np.random.randint(6, size=(3, 4, 5)).astype(np.int32)
    index_1 = np.random.randint(6, size=(4, 5)).astype(np.int32)
    index_2 = np.random.randint(6, size=(5, 3, 4, 5)).astype(np.int32)
    input_x_ms = Tensor(input_x)
    index_0_ms = Tensor(index_0)
    index_1_ms = Tensor(index_1)
    input_2_ms = Tensor(index_2)
    output0, output1, output2 = net(input_x_ms, index_0_ms, index_1_ms, input_2_ms)
    assert np.all(output0.asnumpy() == input_x[index_0] + np.ones([4, 5, 8, 10]))
    assert np.all(output1.asnumpy() == input_x[index_0, index_1] + np.ones([3, 4, 5, 10]))
    assert np.all(output2.asnumpy() == input_x[index_0, index_1, index_2] + np.ones([5, 3, 4, 5]))


class TensorGetItemByMixedTensorsBasicCase(Cell):
    def __init__(self, c0, c1, c2, c3, c4, c5):
        super(TensorGetItemByMixedTensorsBasicCase, self).__init__()
        self.const0 = Tensor(c0)
        self.const1 = Tensor(c1)
        self.const2 = Tensor(c2)
        self.const3 = Tensor(c3)
        self.const4 = Tensor(c4)
        self.const5 = Tensor(c5)

    def construct(self, tensor, index_0, index_1):
        ret0 = tensor[index_0, index_1, 0:3] + self.const0
        ret1 = tensor[0:3, index_0, ...] + self.const1
        ret2 = tensor[0, index_0, index_1] + self.const2
        ret3 = tensor[..., index_0, 0:3] + self.const3
        ret4 = tensor[0:2, index_0, index_1] + self.const4
        ret5 = tensor[..., index_0, index_1] + self.const5
        return ret0, ret1, ret2, ret3, ret4, ret5


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_getitem_by_mixed_tensors():
    const0 = np.ones((3, 4, 5, 3), np.float32)
    const1 = np.ones((3, 3, 4, 5, 5), np.float32)
    const2 = np.ones((3, 4, 5), np.float32)
    const3 = np.ones((3, 3, 4, 5, 3), np.float32)
    const4 = np.ones((2, 3, 4, 5), np.float32)
    const5 = np.ones((3, 3, 4, 5), np.float32)
    net = TensorGetItemByMixedTensorsBasicCase(const0, const1, const2, const3, const4, const5)
    input_np = np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32)
    input_ms = Tensor(input_np, mstype.float32)
    index_np_0 = np.random.randint(3, size=(3, 4, 5)).astype(np.int32)
    index_np_1 = np.random.randint(4, size=(4, 5)).astype(np.int32)
    index_0 = Tensor(index_np_0, mstype.int32)
    index_1 = Tensor(index_np_1, mstype.int32)
    out0, out1, out2, out3, out4, out5 = net(input_ms, index_0, index_1)
    assert np.all(out0.asnumpy() == (input_np[index_np_0, index_np_1, 0:3] + const0))
    assert np.all(out1.asnumpy() == (input_np[0:3, index_np_0, ...] + const1))
    assert np.all(out2.asnumpy() == (input_np[0, index_np_0, index_np_1] + const2))
    assert np.all(out3.asnumpy() == (input_np[..., index_np_0, 0:3] + const3))
    assert np.all(out4.asnumpy() == (input_np[0:2, index_np_0, index_np_1] + const4))
    assert np.all(out5.asnumpy() == (input_np[..., index_np_0, index_np_1] + const5))


class TensorItemByNone(Cell):
    def construct(self, tensor):
        ret = tensor.item()
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_item_by_none():
    net = TensorItemByNone()
    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = Tensor(input_1d_np, mstype.float32)
    input_3d_np = np.random.randint(3, size=(3, 4, 5)).astype(np.int32)
    input_3d_ms = Tensor(input_3d_np, mstype.float32)

    output_ms = net(input_1d_ms)
    assert np.all(output_ms.asnumpy() == input_1d_np.item())

    with pytest.raises(ValueError):
        net(input_3d_ms)


class TensorItemByItem(Cell):
    def construct(self, tensor, index):
        ret = tensor.item(index)
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_item_by_int():
    net = TensorItemByItem()
    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = Tensor(input_1d_np, mstype.float32)

    input_3d_np = np.random.randint(3, size=(3, 4, 5)).astype(np.int32)
    input_3d_ms = Tensor(input_3d_np, mstype.float32)

    index_np_1, index_np_2, index_np_3, index_np_4 = 0, 1.0, 30, 60

    output_1d_ms = net(input_1d_ms, index_np_1)
    output_3d_ms_1 = net(input_3d_ms, index_np_1)
    output_3d_ms_2 = net(input_3d_ms, index_np_3)

    assert np.all(output_1d_ms.asnumpy() == input_1d_np.item(index_np_1))
    assert np.all(output_3d_ms_1.asnumpy() == input_3d_np.item(index_np_1))
    assert np.all(output_3d_ms_2.asnumpy() == input_3d_np.item(index_np_3))

    with pytest.raises(TypeError):
        net(input_1d_ms, index_np_2)

    with pytest.raises(IndexError):
        net(input_1d_ms, index_np_3)

    with pytest.raises(TypeError):
        net(input_3d_ms, index_np_2)

    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_4)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_item_by_tuple():
    net = TensorItemByItem()
    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = Tensor(input_1d_np, mstype.float32)
    input_3d_np = np.random.randint(3, size=(3, 4, 5)).astype(np.int32)
    input_3d_ms = Tensor(input_3d_np, mstype.float32)

    index_np_1 = (0,)
    index_np_2 = (1, 2)
    index_np_3 = (1, 2, 3)
    index_np_4 = (3, 4, 4)
    index_np_5 = (1, 2, 3, 4)

    output_1d_ms = net(input_1d_ms, index_np_1)
    output_3d_ms = net(input_3d_ms, index_np_3)
    assert np.all(output_1d_ms.asnumpy() == input_1d_np.item(index_np_1))
    assert np.all(output_3d_ms.asnumpy() == input_3d_np.item(index_np_3))

    with pytest.raises(ValueError):
        net(input_1d_ms, index_np_2)

    with pytest.raises(ValueError):
        net(input_3d_ms, index_np_2)

    with pytest.raises(IndexError):
        net(input_3d_ms, index_np_4)

    with pytest.raises(ValueError):
        net(input_3d_ms, index_np_5)


class TensorSetItemByMixedTensors_0(Cell):
    def __init__(self, value):
        super(TensorSetItemByMixedTensors_0, self).__init__()
        self.const = Tensor(np.ones((3, 4, 5), np.float32))
        self.param = Parameter(Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)),
                                      mstype.float32),
                               name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[0:2, index_0, index_1] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_mixed_tensors_0():
    value = 88.0
    net = TensorSetItemByMixedTensors_0(value)
    index_0 = np.random.randint(3, size=(3, 4, 5))
    index_1 = np.random.randint(4, size=(4, 5))
    index_2 = np.random.randint(3, size=(2, 1, 4, 5))
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    input_np = np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32)
    const = np.ones((3, 4, 5), np.float32)
    out = net(index_0_ms, index_1_ms, index_2_ms)
    input_np[0:2, index_0, index_1] = value
    assert np.all(out.asnumpy() == (input_np + const))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
class TensorSetItemByMixedTensors_1(Cell):
    def __init__(self, value):
        super(TensorSetItemByMixedTensors_1, self).__init__()
        self.const = Tensor(np.ones((3, 4, 5), np.float32))
        self.param = Parameter(Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)), mstype.float32),
                               name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[0:2, index_0, ...] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_mixed_tensors_1():
    value = 88.0
    net = TensorSetItemByMixedTensors_1(value)
    index_0 = np.random.randint(3, size=(3, 4, 5))
    index_1 = np.random.randint(4, size=(4, 5))
    index_2 = np.random.randint(3, size=(2, 1, 4, 5))
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    input_np = np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32)
    const = np.ones((3, 4, 5), np.float32)
    out = net(index_0_ms, index_1_ms, index_2_ms)
    input_np[0:2, index_0, ...] = value
    assert np.all(out.asnumpy() == (input_np + const))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
class TensorSetItemByMixedTensors_2(Cell):
    def __init__(self, value):
        super(TensorSetItemByMixedTensors_2, self).__init__()
        self.const = Tensor(np.ones((3, 4, 5), np.float16))
        self.param = Parameter(Tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5)), mstype.float16),
                               name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[..., index_0, 1] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_mixed_tensors_2():
    value = 88.0
    net = TensorSetItemByMixedTensors_2(value)
    index_0 = np.random.randint(3, size=(3, 4, 5))
    index_1 = np.random.randint(4, size=(4, 5))
    index_2 = np.random.randint(3, size=(2, 1, 4, 5))
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    input_np = np.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(np.float32)
    const = np.ones((3, 4, 5), np.float32)
    out = net(index_0_ms, index_1_ms, index_2_ms)
    input_np[..., index_0, 1] = value
    assert np.all(out.asnumpy() == (input_np + const))


class TensorGetItemByMixedTensorsIndexError(Cell):
    def construct(self, x, index_0, index_1):
        ret = x[index_0, index_1, 0:3, ..., 0:5, [1, 2, 3, 4]]
        return ret


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_getitem_by_mixed_tensor_exception():
    input_ms = Tensor(np.arange(3 * 4 * 5 * 6 * 7 * 8 * 9).reshape((3, 4, 5, 6, 7, 8, 9)), mstype.int32)
    index_0 = Tensor(np.random.randint(3, size=(3, 4, 5)), mstype.int32)
    index_1 = Tensor(np.random.randint(4, size=(3, 4, 5)), mstype.int32)
    net1 = TensorGetItemByMixedTensorsIndexError()
    with pytest.raises(ValueError):
        net1(input_ms, index_0, index_1)


class TensorSetItemByOneTensorWithNumber(Cell):
    def __init__(self, value):
        super(TensorSetItemByOneTensorWithNumber, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index):
        self.param[index] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_one_tensor_with_number():
    value = 0.0
    net = TensorSetItemByOneTensorWithNumber(value)
    index_np = np.random.randint(4, size=(5, 4))
    index = Tensor(index_np, mstype.int32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8))
    const = np.ones((6, 7, 8)).astype(np.float32)
    out = net(index)
    input_data[index_np] = value
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByOneTensorWithTensor(Cell):
    def __init__(self):
        super(TensorSetItemByOneTensorWithTensor, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index, value):
        self.param[index] = value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_one_tensor_with_tensor():
    net = TensorSetItemByOneTensorWithTensor()
    index_np = np.random.randint(4, size=(5, 4))
    index = Tensor(index_np, mstype.int32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8))
    const = np.ones((6, 7, 8)).astype(np.float32)
    value = np.zeros((4, 7, 8)).astype(np.float32)
    value_ms = Tensor(value, mstype.float32)
    out = net(index, value_ms)
    input_data[index_np] = value
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByOneTensorWithTupleOfNumber(Cell):
    def __init__(self, value):
        super(TensorSetItemByOneTensorWithTupleOfNumber, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index):
        self.param[index] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_one_tensor_with_tuple_number():
    value = (0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7)
    net = TensorSetItemByOneTensorWithTupleOfNumber(value)
    input_np = np.random.randint(5, size=(5, 4))
    input_ms = Tensor(input_np, mstype.int32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8)).astype(np.float32)
    const = np.ones((6, 7, 8)).astype(np.float32)
    out = net(input_ms)
    input_data[input_np] = value
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByOneTensorWithTupleOfTensor(Cell):
    def __init__(self):
        super(TensorSetItemByOneTensorWithTupleOfTensor, self).__init__()
        self.const = Tensor(np.ones((6, 3, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 3 * 8).reshape((6, 3, 8)), mstype.float32), name="x")

    def construct(self, index, value_0, value_1, value_2):
        self.param[index] = (value_0, value_1, value_2)
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_one_tensor_with_tuple_tensors():
    net = TensorSetItemByOneTensorWithTupleOfTensor()
    input_np = np.random.randint(6, size=(5, 4)).astype(np.int32)
    input_ms = Tensor(input_np, mstype.int32)
    input_data = np.arange(6 * 3 * 8).reshape((6, 3, 8)).astype(np.float32)
    value_0_np = np.zeros((8,), np.float32)
    value_1_np = np.ones((8,), np.float32)
    value_2_np = np.ones((8,), np.float32)*2
    value_0 = Tensor(value_0_np)
    value_1 = Tensor(value_1_np)
    value_2 = Tensor(value_2_np)
    const = np.ones((6, 3, 8)).astype(np.float32)
    out = net(input_ms, value_0, value_1, value_2)
    input_data[input_np] = (value_0_np, value_1_np, value_2_np)
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByTensorsWithNumber(Cell):
    def __init__(self, value):
        super(TensorSetItemByTensorsWithNumber, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[index_0, index_1, index_2] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.level0
def test_setitem_by_tensors_with_number():
    value = 0.0
    net = TensorSetItemByTensorsWithNumber(value)
    index_0 = np.random.randint(6, size=(3, 4, 5))
    index_1 = np.random.randint(7, size=(4, 5))
    index_2 = np.random.randint(8, size=(5, 3, 4, 5))
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    out = net(index_0_ms, index_1_ms, index_2_ms)
    const = np.ones((6, 7, 8)).astype(np.float32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8)).astype(np.float32)
    input_data[index_0, index_1, index_2] = value
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByTensorsWithTensor(Cell):
    def __init__(self):
        super(TensorSetItemByTensorsWithTensor, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, value):
        self.param[index_0, index_1, index_2] = value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_tensors_with_tensor():
    net = TensorSetItemByTensorsWithTensor()
    index_0 = np.random.randint(6, size=(3, 4, 5))
    index_1 = np.random.randint(7, size=(4, 5))
    index_2 = np.random.randint(8, size=(5, 3, 4, 5))
    value = np.zeros((4, 5)).astype(np.float32)
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    value_ms = Tensor(value, mstype.float32)
    out = net(index_0_ms, index_1_ms, index_2_ms, value_ms)
    const = np.ones((6, 7, 8)).astype(np.float32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8)).astype(np.float32)
    input_data[index_0, index_1, index_2] = value
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByTensorsWithTensorNumberError(Cell):
    def __init__(self):
        super(TensorSetItemByTensorsWithTensorNumberError, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, index_3, value):
        self.param[index_0, index_1, index_2, index_3] = value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_tensors_with_tensor_error():
    index_0 = Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32)
    index_1 = Tensor(np.random.randint(7, size=(4, 5)), mstype.int32)
    index_2 = Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)
    index_3 = Tensor(np.random.randint(8, size=(1, 3, 4, 5)), mstype.int32)
    value = Tensor(np.zeros((2, 5)), mstype.float32)
    net = TensorSetItemByTensorsWithTensorNumberError()
    with pytest.raises(IndexError):
        net(index_0, index_1, index_2, index_3, value)


class TensorSetItemByTensorsWithTupleOfNumber(Cell):
    def __init__(self, value):
        super(TensorSetItemByTensorsWithTupleOfNumber, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")
        self.value = value

    def construct(self, index_0, index_1, index_2):
        self.param[index_0, index_1, index_2] = self.value
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
# GPU op has bug, and has not been fixed.
@pytest.mark.env_onecard
def test_setitem_by_tensors_with_tuple_of_number():
    value = (0.0, 1.1, 2.2, 3.3, 4.4)
    net = TensorSetItemByTensorsWithTupleOfNumber(value)
    index_0 = np.random.randint(6, size=(3, 4, 5))
    index_1 = np.random.randint(7, size=(4, 5))
    index_2 = np.random.randint(8, size=(5, 3, 4, 5))
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8)).astype(np.float32)
    input_data[index_0, index_1, index_2] = value
    const = np.ones((6, 7, 8)).astype(np.float32)
    out = net(index_0_ms, index_1_ms, index_2_ms)
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByTensorsWithTupleOfTensor(Cell):
    def __init__(self):
        super(TensorSetItemByTensorsWithTupleOfTensor, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, value_0, value_1, value_2):
        self.param[index_0, index_1, index_2] = (value_0, value_1, value_2)
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
# GPU op has bug, and has not been fixed.
@pytest.mark.env_onecard
def test_setitem_by_tensors_with_tuple_of_tensor():
    value_0 = np.zeros((4, 5))
    value_1 = np.ones((4, 5))
    value_2 = np.ones((4, 5)) * 2
    value_0_ms = Tensor(value_0, mstype.float32)
    value_1_ms = Tensor(value_1, mstype.float32)
    value_2_ms = Tensor(value_2, mstype.float32)
    net = TensorSetItemByTensorsWithTupleOfTensor()
    index_0 = np.random.randint(6, size=(3, 4, 5))
    index_1 = np.random.randint(7, size=(4, 5))
    index_2 = np.random.randint(8, size=(5, 3, 4, 5))
    index_0_ms = Tensor(index_0, mstype.int32)
    index_1_ms = Tensor(index_1, mstype.int32)
    index_2_ms = Tensor(index_2, mstype.int32)
    input_data = np.arange(6 * 7 * 8).reshape((6, 7, 8)).astype(np.float32)
    input_data[index_0, index_1, index_2] = (value_0, value_1, value_2)
    const = np.ones((6, 7, 8)).astype(np.float32)
    out = net(index_0_ms, index_1_ms, index_2_ms, value_0_ms, value_1_ms, value_2_ms)
    assert np.all(out.asnumpy() == (input_data + const))


class TensorSetItemByTensorsWithTupleOfTensorNumberError(Cell):
    def __init__(self):
        super(TensorSetItemByTensorsWithTupleOfTensorNumberError, self).__init__()
        self.const = Tensor(np.ones((6, 7, 8)), mstype.float32)
        self.param = Parameter(Tensor(np.arange(6 * 7 * 8).reshape((6, 7, 8)), mstype.float32), name="x")

    def construct(self, index_0, index_1, index_2, value_0, value_1):
        self.param[index_0, index_1, index_2] = (value_0, value_1)
        ret = self.param + self.const
        return ret


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_by_tensor_with_tuple_of_tensor_error():
    net = TensorSetItemByTensorsWithTupleOfTensorNumberError()
    index_0_ms = Tensor(np.random.randint(6, size=(3, 4, 5)), mstype.int32)
    index_1_ms = Tensor(np.random.randint(7, size=(4, 5)), mstype.int32)
    index_2_ms = Tensor(np.random.randint(8, size=(5, 3, 4, 5)), mstype.int32)
    value_0 = np.zeros((4, 5))
    value_1 = np.ones((4, 5))
    value_0_ms = Tensor(value_0, mstype.float32)
    value_1_ms = Tensor(value_1, mstype.float32)
    with pytest.raises(ValueError):
        net(index_0_ms, index_1_ms, index_2_ms, value_0_ms, value_1_ms)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_setitem_grad():
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.weight = Parameter(
                Tensor(np.ones([4, 4, 5]), dtype=mstype.float32), "b1", requires_grad=True)

        def construct(self, a, b):
            a[1:3:1, ::] = b
            c = a + self.weight
            return c

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = ParameterTuple(net.trainable_params())

        def construct(self, x, y, sens):
            return grad_by_list_with_sens(self.net, self.weights)(x, y, sens)
    net = GradNet(Net())
    x = Tensor(np.ones([4, 4, 5]).astype(np.float32), mstype.float32)
    y = Tensor(np.array([3]).astype(np.float32), mstype.float32)
    sens = Tensor(np.ones([4, 4, 5]).astype(np.float32), mstype.float32)
    net(x, y, sens)


class TensorAssignWithSliceError1(Cell):
    def construct(self, a, b):
        a[1:3:-1, ::] = b
        return a


class TensorAssignWithSliceError2(Cell):
    def construct(self, a, b):
        a[1:3:-1] = b
        return a


class TensorAssignWithSlice2(Cell):
    def construct(self, a, b, ck):
        a[1:5] = b
        a[3:4] = 5
        a[-1:1:-1] = b
        a[-1:3:-1] = 5
        a[::] = b
        a[::] = 9
        z = a + ck
        return z


class TensorAssignWithSlice(Cell):
    def __init__(self):
        super(TensorAssignWithSlice, self).__init__()
        self.c = 2.0

    def construct(self, a, b, ck):
        a[1:3, ::] = b
        a[2:3:, 3:] = b
        a[::] = b
        a[::] = self.c
        a[::, ::] = b
        a[::, ::] = self.c
        a[2:3:, 0:, 4:1:-1] = b
        a[2:3:, 0:, 4:1:-1] = self.c
        z = a + ck
        return z


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_assign_slice_value_1():
    net = TensorAssignWithSlice()
    a = np.arange(60).reshape(3, 4, 5)
    b = np.array([1]).astype(np.float32)  # Tensor([1], dtype=mstype.float32)
    ck = np.arange(60).reshape(3, 4, 5)
    ta = Tensor(a, dtype=mstype.float32)
    tb = Tensor(b, dtype=mstype.float32)
    tck = Tensor(ck, dtype=mstype.float32)
    out = net(ta, tb, tck)
    a[1:3, ::] = b
    a[2:3:, 3:] = b
    a[::] = b
    a[::] = 2.0
    a[::, ::] = b
    a[::, ::] = 2.0
    a[2:3:, 0:, 4:1:-1] = b
    a[2:3:, 0:, 4:1:-1] = 2.0
    z = a + ck
    assert np.all(z == out.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_assign_slice_value_2():
    net2 = TensorAssignWithSlice2()
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    ck = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    b = np.array([1]).astype(np.float32)  # Tensor([1], dtype=mstype.float32)
    tb = Tensor(b, dtype=mstype.float32)
    ta = Tensor(a, dtype=mstype.float32)
    tck = Tensor(ck, dtype=mstype.float32)
    out = net2(ta, tb, tck)
    a[1:5] = b
    a[3:4] = 5
    a[-1:1:-1] = b
    a[-1:3:-1] = 5
    a[::] = b
    a[::] = 9
    z = a + ck
    assert np.all(z == out.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_assign_exception():
    net = TensorAssignWithSlice()
    net2 = TensorAssignWithSlice2()
    # The test case is no longer appropriate since x[1:3:-1] = np.array(2) does
    # not incur an error in numpy, which leaves the original array unchanged after
    # the assign operation.
    # net_e1 = TensorAssignWithSliceError1()
    # net_e2 = TensorAssignWithSliceError2()
    a = np.arange(60).reshape(3, 4, 5)
    ck = np.arange(60).reshape(3, 4, 5)
    b = Tensor([1], dtype=mstype.float32)
    Ta = Tensor(a, dtype=mstype.float32)
    Tck = Tensor(ck, dtype=mstype.float32)
    Ta4d = Tensor(a.reshape(1, 3, 4, 5), dtype=mstype.float32)
    Ta4d_ck = Tensor(ck.reshape(1, 3, 4, 5), dtype=mstype.float32)
    Tb = Tensor([1, 3], dtype=mstype.float32)
    Tc = Tensor([], dtype=mstype.float32)
    t = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mstype.float32)
    tck = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=mstype.float32)
    # Error for A[Slice] = Number
    # 1. A[Slice] = Number,  Slice error
    # with pytest.raises(ValueError):
    #     net_e2(t, 2)

    # Error for A[Slice] = U, U is a Tensor
    # 1. A[Slice] = U,  u.size is error
    with pytest.raises(ValueError):
        net2(t, Tb, tck)
    # 2. A[Slice] = U, U is empty
    with pytest.raises(ValueError):
        net2(t, Tc, tck)
    # 3. A[Slice] = U, U.size error
    with pytest.raises(ValueError):
        net2(t, Tb, tck)

    # Error for A[Tuple(Slice...)] = Tensor
    # 1. A[Tuple(Slice...)] = U, U is empty
    with pytest.raises(ValueError):
        net(Ta, Tc, Tck)
    # 2. A[Tuple(Slice...)] = U, U.size error
    with pytest.raises(ValueError):
        net(Ta, Tb, Tck)
    # 3. A[Tuple(Slice...)] = U,  Slice error
    # with pytest.raises(IndexError):
    #     net_e1(Ta, b)

    # Error for A[Tuple(Slice...)] = Number
    # 1. A[Tuple(Slice...)] = Number,  Slice error
    # with pytest.raises(IndexError):
    #     net_e1(Ta, 2)

    net = TensorAssignWithInteger()
    # Error for A[Number] = scalar/Tensor
    # 1. A[Number] = U, U is a Tensor, u.size not match
    with pytest.raises(ValueError):
        net(Ta, Tb, Tck)
    with pytest.raises(ValueError):
        net(Ta, Tc, Tck)
    # 2. A[Number] = U, the number index error
    with pytest.raises(IndexError):
        net(Ta4d, b, Ta4d_ck)

    # Error for A[(n,m)] = scalar/Tensor
    # 1. A[(n,m)] = U, U is a tensor. u.size not match
    net = TensorAssignWithTupleInteger()
    with pytest.raises(ValueError):
        net(Ta, Tc, Tck)
    with pytest.raises(ValueError):
        net(Ta, Tb, Tck)
    # 2. A[(n,m)] = U, the number index error
    with pytest.raises(IndexError):
        net(Ta4d, b, Ta4d_ck)

    # Error for  A[...] = U or A[1:, ...] = u
    # 1. A[...] = scalar/tensor
    net = TensorAssignWithEllipsis()
    net(Ta, Ta4d)
    with pytest.raises(ValueError):
        net(Ta, Tc)
    with pytest.raises(ValueError):
        net(Ta, Tb)
    # 2. A[::, 1:, ...] = scalar/tensor
    net = TensorAssignWithTupleEllipsis()
    net(Ta, b)
    with pytest.raises(ValueError):
        net(Ta, Tb)


class TensorAssignWithTupleEllipsis2(Cell):
    def construct(self, a, b):
        a[1:, ..., ::] = b
        return a


class TensorAssignWithTupleEllipsis(Cell):
    def construct(self, a, b):
        a[:2, ...] = 1.0
        a[1:, ...] = b
        return a


class TensorAssignWithEllipsis(Cell):
    def construct(self, a, b):
        a[...] = 1
        a[...] = b
        return a


class TensorAssignWithInteger(Cell):
    def construct(self, a, b, ck):
        a[1] = 1
        a[0] = b
        z = a + ck
        return z


class TensorAssignWithTupleInteger(Cell):
    def construct(self, a, b, ck):
        a[(1)] = 1
        a[(1)] = b
        a[(1, 1)] = b
        a[(1, 1)] = 1
        z = a + ck
        return z


class TensorAssignWithBoolTensorIndex(Cell):
    def __init__(self):
        super(TensorAssignWithBoolTensorIndex, self).__init__()
        self.t = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
        self.u_scalar = 5

    def construct(self, a, b, c, u_tensor):
        a[c] = self.u_scalar
        a[b] = u_tensor
        z = a + self.t
        return z


class TensorAssignWithBoolTensorIndexError(Cell):
    def construct(self, a, b, c, u_tensor):
        a[b][c] = u_tensor
        return a


class TensorAssignWithBoolTensorIndex2(Cell):
    def __init__(self):
        super(TensorAssignWithBoolTensorIndex2, self).__init__()
        self.t = Tensor(np.ones([3, 4, 5]), dtype=mstype.float32)
        self.u_scalar = 5

    def construct(self, a, u_tensor):
        a[a > 8] = u_tensor
        a[a >= 6] = self.u_scalar
        a[a < 3] = self.u_scalar
        a[a <= 5] = u_tensor
        a[a == 5] = self.u_scalar
        z = a + self.t
        return z


class TensorAssignWithBoolTensorIndex2Error(Cell):
    def construct(self, a, u_tensor):
        a[a > 8][a > 5] = u_tensor
        return a


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_assign_bool_index_0():
    a = np.arange(60).reshape(3, 4, 5)
    b = a > 5
    c = a < 3
    Ta = Tensor(a, dtype=mstype.float32)
    Tb = Tensor(b)
    Tc = Tensor(c)
    u_tensor = Tensor([1], dtype=mstype.float32)
    net1 = TensorAssignWithBoolTensorIndex()
    out = net1(Ta, Tb, Tc, u_tensor)
    res = np.arange(60).reshape(3, 4, 5)
    res[c] = 5
    res[b] = 1
    res = res + np.ones([3, 4, 5])
    assert np.all(out.asnumpy() == res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_assign_bool_index_1():
    a = np.arange(60).reshape(3, 4, 5)
    Ta = Tensor(a, dtype=mstype.float32)
    u_tensor = Tensor([1], dtype=mstype.float32)
    net2 = TensorAssignWithBoolTensorIndex2()
    out = net2(Ta, u_tensor)
    res = np.arange(60).reshape(3, 4, 5)
    res[res > 8] = 1
    res[res >= 6] = 5
    res[res < 3] = 5
    res[res <= 5] = 1
    res[res == 5] = 5
    res = res + np.ones([3, 4, 5])
    assert np.all(out.asnumpy() == res)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_assign_bool_index_exception():
    a = np.arange(60).reshape(3, 4, 5)
    b = a > 5
    c = a < 3
    Ta = Tensor(a, dtype=mstype.float32)
    Tb = Tensor(b)
    Tc = Tensor(c)
    Td = Tensor([True, True])
    u_tensor = Tensor([1], dtype=mstype.float32)
    u_tensor_error = Tensor([1, 2], dtype=mstype.float32)
    u_scalar = 5
    net1 = TensorAssignWithBoolTensorIndex()
    net2 = TensorAssignWithBoolTensorIndex2()
    with pytest.raises(ValueError):
        net1(Ta, Td, Tc, u_tensor)
    with pytest.raises(IndexError):
        net1(Ta, u_tensor, Tc, u_tensor)
    with pytest.raises(ValueError):
        net1(Ta, Tb, Td, u_tensor)
    with pytest.raises(IndexError):
        net1(Ta, Tb, Ta, u_tensor)
    with pytest.raises(ValueError):
        net1(Ta, Tb, Tc, u_tensor_error)
    # net1(Ta, u_tensor, Tc, u_tensor_error, u_scalar)
    with pytest.raises(ValueError):
        net2(Ta, u_tensor_error)
    net3 = TensorAssignWithBoolTensorIndexError()
    with pytest.raises(IndexError):
        net3(Ta, Tb, Tc, u_tensor)
    with pytest.raises(IndexError):
        net3(Ta, Tb, Tc, u_scalar)
    net4 = TensorAssignWithBoolTensorIndex2Error()
    with pytest.raises(IndexError):
        net4(Ta, u_tensor)
    with pytest.raises(IndexError):
        net4(Ta, u_scalar)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_slice_reduce_out_of_bounds_neg():
    class NetWork(Cell):
        def __init__(self):
            super(NetWork, self).__init__()
            self.tensor_ret = Tensor(np.array(9, np.int32))

        def construct(self, tensor):
            ret = tensor[-7, 3, 4]
            return ret

    input_tensor = Tensor(np.ones([6, 8, 10], np.int32))
    net = NetWork()
    with pytest.raises(IndexError) as ex:
        net(input_tensor)
    assert "'begin[0]' must be in [-6, 6) when 'shrink_axis_mask' is greater than 0, " \
           "but got 'shrink_axis_mask': 7, 'strides[0]': 1, 'begin[0]': -7." in str(ex.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_slice_reduce_out_of_bounds_positive():
    class NetWork(Cell):
        def __init__(self):
            super(NetWork, self).__init__()
            self.tensor_ret = Tensor(np.array(9, np.int32))

        def construct(self, tensor):
            ret = tensor[6, 3, 4]
            return ret

    input_tensor = Tensor(np.ones([6, 8, 10], np.int32))
    net = NetWork()
    with pytest.raises(IndexError) as ex:
        net(input_tensor)
    assert "'begin[0]' must be in [-6, 6) when 'shrink_axis_mask' is greater than 0, " \
           "but got 'shrink_axis_mask': 7, 'strides[0]': 1, 'begin[0]': 6." in str(ex.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tensor_range():
    a = np.arange(4*5*6).reshape(4, 5, 6).astype(np.float32)
    ta = Tensor(a, mstype.float32)
    ms_out = []
    for item in ta:
        ms_out.append(item)
    np_out = []
    for item in a:
        np_out.append(item)
    for i, elem in enumerate(ms_out):
        assert np.all(elem.asnumpy() == np_out[i])
