# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
import mindspore as ms
from mindspore import ops, nn, ParameterTuple, context, set_seed, Tensor

from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds
from mindspore.common.initializer import One

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
set_seed(2)


def _exec_preprocess(network, is_train, dataset, dataset_sink_mode, epoch_num, sink_size):
    if dataset_sink_mode and not is_train:
        dataset.__loop_size__ = 1

    dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)

    if dataset_sink_mode:
        network = connect_network_with_dataset(network, dataset_helper)

    return dataset_helper, network


def dynamic_shape_sink_process(network, dataset, is_train=True, sink_size=1):
    # epoch_num=1 sink_size=1: exec one step
    dataset_sink_mode = True
    epoch_num = 1
    dataset_helper, network = _exec_preprocess(network, is_train, dataset, dataset_sink_mode, epoch_num, sink_size)
    network.set_train(is_train)
    for inputs in dataset_helper:
        outputs = network(*inputs)
        return outputs


def fixed_shape_process(network, dataset, is_train=True):
    network.set_train(is_train)
    for inputs in dataset.create_tuple_iterator():
        outputs = network(*inputs)
        return outputs


def dataset_generator(data_list):
    for data in data_list:
        yield data


def compare(output, expect):
    if isinstance(output, (tuple, list)):
        assert isinstance(expect, (tuple, list))
        for output_, expect_ in zip(output, expect):
            if not compare(output_, expect_):
                return False
    else:
        if not np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1.0e-4, atol=1.0e-4):
            return False
    return True


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.params = ParameterTuple(net.trainable_params())

    def construct(self, *inputs):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(*inputs)


def common_func(dynamic_range, input_shape, data_type, op_net):
    data_list = []
    for i in dynamic_range:
        cur_data = []
        for data_shape in input_shape:
            cur_shape = [dim if dim is not None else i for dim in data_shape]
            cur_data.append(np.random.random(cur_shape).astype(data_type))
        data_list.append(tuple(cur_data))

    def np_type_to_ms(data_type):
        if data_type == np.float32:
            return ms.float32
        if data_type == np.float64:
            return ms.float64
        if data_type == np.int32:
            return ms.int32
        if data_type == np.int64:
            return ms.int64
        raise ValueError("Unsupportted datatype: {}".format(data_type))

    dynamic_data_map = {}
    dyn_tensors = []
    for i, val in enumerate(input_shape):
        dynamic_data_map["data" + str(i + 1)] = val
        if None in val:
            dyn_tensors.append(Tensor(dtype=np_type_to_ms(data_type), shape=val))
        else:
            dyn_tensors.append(Tensor(dtype=np_type_to_ms(data_type), shape=val, init=One()))

    dataset = ds.GeneratorDataset(data_list, list(dynamic_data_map.keys()))
    net = GradNetWrtX(op_net)
    net.set_inputs(*dyn_tensors)

    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


class LayerNormNet(nn.Cell):
    def __init__(self, last_dim):
        super(LayerNormNet, self).__init__()
        self.layernorm = nn.LayerNorm([last_dim])

    def construct(self, x):
        return self.layernorm(x)


class Conv2dNet(nn.Cell):
    def __init__(self):
        super(Conv2dNet, self).__init__()
        self.conv = nn.Conv2d(3, 10, 4, pad_mode="valid", has_bias=False, weight_init='normal')

    def construct(self, x):
        return self.conv(x)


class DropoutNet(nn.Cell):
    def __init__(self):
        super(DropoutNet, self).__init__()
        self.drop = nn.Dropout(p=0.5)
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return self.relu(self.drop(x))


class ReduceSumNet(nn.Cell):
    def __init__(self, axis=()):
        super(ReduceSumNet, self).__init__()
        self.reduce = ops.ReduceSum()
        self.axis = axis

    def construct(self, x):
        return self.reduce(x, self.axis)


class AddNet(nn.Cell):
    def construct(self, x, y):
        return ops.add(x, y)


class ShapeTensorNet(nn.Cell):
    def __init__(self):
        super(ShapeTensorNet, self).__init__()
        self.reshape = ops.Reshape()
        self.tensor_shape = ops.TensorShape()
        self.shape = ops.Shape()
        self.strided_slice = ops.StridedSlice()
        self.mul = ops.Mul()
        self.tensor_scatter_update = ops.TensorScatterUpdate()

    def construct(self, x, y):
        res = self.tensor_shape(x)
        res = self.strided_slice(res, (1,), (4,), (1,))
        res = self.mul(res, 4)
        y = self.reshape(x, res)
        y_shape = self.shape(y)
        indice = Tensor(np.array([[0], [1]]).astype(np.int32))
        update = Tensor(np.array([32, 32]).astype(np.int64))
        res = self.tensor_scatter_update(res, indice, update)
        z = self.reshape(y, res)
        res_shape = self.shape(z)
        return y_shape, res_shape


class SoftmaxNet(nn.Cell):
    def construct(self, x):
        return ops.Softmax(axis=-1)(x)


class BatchNormNet(nn.Cell):
    def __init__(self, channels):
        super(BatchNormNet, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def construct(self, x):
        return self.bn(x)


class SquareSumAllNet(nn.Cell):
    def __init__(self):
        super(SquareSumAllNet, self).__init__()
        self.square_sum_all = ops.SquareSumAll()

    def construct(self, x, y):
        return self.square_sum_all(x, y)


class HSwishNet(nn.Cell):
    def __init__(self):
        super(HSwishNet, self).__init__()
        self.hswish = ops.HSwish()

    def construct(self, x):
        return self.hswish(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_layernorm():
    """
    Feature: Test LayerNorm and its backward. The input shape is dynamic.
    Description: The second dim of input is unknown.
    Expectation: Assert that results are consistent with fixed shape.
    """
    last_dim = 32
    batch_size = 16
    dynamic_range = range(20, 23)
    data_type = np.float32
    input_shape = [(batch_size, None, last_dim), (batch_size, None, last_dim)]
    net = LayerNormNet(last_dim)
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_conv2d():
    """
    Feature: Test Conv2d and its backward. The input shape is dynamic.
    Description: Input dim of `H `or `W` is unknown. Conv2d's attr[pad] set to "valid".
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    dynamic_range = range(220, 224)
    data_type = np.float32
    input_shape = [(batch_size, 3, None, 112), (batch_size, 10, 219, 109)]
    net = Conv2dNet()
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_dropout():
    """
    Feature: Test Dropout and its backward.
    Description: The input shape is dynamic.
    Expectation: Dropout result is random, assert gradient shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i, 256).astype(np.float32),
                          np.random.rand(batch_size, i, 256).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    t0 = Tensor(dtype=ms.float32, shape=[batch_size, None, 256])
    t1 = Tensor(dtype=ms.float32, shape=[batch_size, None, 256])
    net = GradNetWrtX(DropoutNet())
    net.set_inputs(t0, t1)
    net.set_train()
    gradients = dynamic_shape_sink_process(net, dataset)
    assert gradients[0][0].shape == (batch_size, 49, 256)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_reducesum1():
    """
    Feature: Test ReduceSum and its backward. The input shape is dynamic.
    Description: axis=(), result of reduce sum is a scalar, gradient shape is the same as input, value is all one.
    Expectation: Assert gradient shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i, i + 2).astype(np.float32),
                          np.array(1).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    t0 = Tensor(dtype=ms.float32, shape=[batch_size, None, None])
    t1 = Tensor(dtype=ms.float32, shape=[], init=One())
    net = GradNetWrtX(ReduceSumNet())
    net.set_inputs(t0, t1)
    gradients = dynamic_shape_sink_process(net, dataset)
    assert gradients[0][0].shape == (batch_size, 49, 51)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_reducesum2():
    """
    Feature: Test ReduceSum and its backward. The input shape is dynamic.
    Description: axis is a scalar, not tuple.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i, i + 2).astype(np.float32),
                          np.random.rand(batch_size, i + 2).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    net = GradNetWrtX(ReduceSumNet(1))

    t0 = Tensor(dtype=ms.float32, shape=[batch_size, None, None])
    t1 = Tensor(dtype=ms.float32, shape=[batch_size, None])
    net.set_inputs(t0, t1)
    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_add1():
    """
    Feature: Test Add and its backward. The input shape is dynamic.
    Description: Second input is a scalar. Shape of forward result is the same as first input.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i).astype(np.float32),
                          np.array(1).astype(np.float32),
                          np.random.rand(batch_size, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2", "data3"])
    net = GradNetWrtX(AddNet())

    t0 = Tensor(dtype=ms.float32, shape=[batch_size, None])
    t1 = Tensor(dtype=ms.float32, shape=[], init=One())
    t2 = Tensor(dtype=ms.float32, shape=[batch_size, None])
    net.set_inputs(t0, t1, t2)
    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_add2():
    """
    Feature: Test Add and its backward. The input shape is dynamic.
    Description: Shape of forward result is the same as first input. The axis of reduce_sum in add's bprop will be a
                 empty Tensor.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, 2, i).astype(np.float32),
                          np.random.rand(2, i).astype(np.float32),
                          np.random.rand(batch_size, 2, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2", "data3"])
    net = GradNetWrtX(AddNet())

    t0 = Tensor(dtype=ms.float32, shape=[batch_size, 2, None])
    t1 = Tensor(dtype=ms.float32, shape=[2, None])
    t2 = Tensor(dtype=ms.float32, shape=[batch_size, 2, None])
    net.set_inputs(t0, t1, t2)
    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


@pytest.mark.skip(reason='Operator Shape is not support in backend yet.')
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tensor_shape_value_infer():
    """
    Feature: Test shape tensor infer mechanism.
    Description: Shape value of tensor should be inferred precisely.
    Expectation: Assert that results are consistent with fixed shape.
    """
    data_list = []
    for i in range(42, 50):
        data_list.append((np.random.rand(64, 16, 4, i).astype(np.float32),
                          np.random.rand(64, 16, 4, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    net = ShapeTensorNet()
    input_0 = Tensor(shape=[64, 16, 4, None], dtype=ms.float32)
    input_1 = Tensor(shape=[64, 16, 4, None], dtype=ms.float32)
    net.set_inputs(input_0, input_1)
    res = dynamic_shape_sink_process(net, dataset, True, 3)
    assert res == ((64, 16, -1), (32, 32, -1))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_softmax():
    """
    Feature: Test Softmax and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    dynamic_range = range(48, 50)
    data_type = np.float32
    input_shape = [(batch_size, 2, None), (batch_size, 2, None)]
    net = SoftmaxNet()
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.skip(reason="his bug")
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm():
    """
    Feature: Test Batchnorm2D and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 1
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(batch_size, 256, None, 12), (batch_size, 256, None, 12)]
    net = BatchNormNet(256)
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_square_sum_all():
    """
    Feature: Test SquareSumAll. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(1, 4):
        data_list.append((np.random.rand(batch_size, i).astype(np.float32),
                          np.random.rand(batch_size, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    net = SquareSumAllNet()

    t0 = Tensor(dtype=ms.float32, shape=[batch_size, None])
    t1 = Tensor(dtype=ms.float32, shape=[batch_size, None])
    net.set_inputs(t0, t1)
    out = dynamic_shape_sink_process(net, dataset)
    out_expect = fixed_shape_process(net, dataset)
    assert compare(out, out_expect)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('dtype', [np.float32])
@pytest.mark.env_onecard
def test_dynamic_hswish(dtype):
    """
    Feature: Test HSwish and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    dynamic_range = range(48, 50)
    input_shape = [(batch_size, 2, None), (batch_size, 2, None)]
    net = HSwishNet()
    common_func(dynamic_range, input_shape, dtype, net)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_reshape():
    """
    Feature: dynamic shape for reshape
    Description: This case tests the dynamic shape for op reshape on ascend and gpu.
    Expectation: success
    """
    class MyReLU(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = ops.ReLU()

        def construct(self, x):
            return self.relu(x)

    class ReshapeNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = ops.Reshape()
            self.sin = ops.Sin()
            self.relu = MyReLU()

        def construct(self, x):
            res = self.relu(x)
            res = self.reshape(res, (128, -1))
            res = self.sin(res)
            res = self.reshape(res, (32, 16, 4, -1))
            res = self.relu(res)
            return res

    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(32, 16, 4, i).astype(np.float32),
                          np.random.rand(32, 16, 4, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    net = ReshapeNet()
    net.add_flags_recursive(defer_inline=True)
    grad_net = GradNetWrtX(net)
    t0 = Tensor(dtype=ms.float32, shape=[32, 16, 4, None])
    t1 = Tensor(dtype=ms.float32, shape=[32, 16, 4, None])
    grad_net.set_inputs(t0, t1)
    gradients = dynamic_shape_sink_process(grad_net, dataset)
    print(gradients)
