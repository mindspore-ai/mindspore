# Copyright 2022 Huawei Technologies Co., Ltd
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

from mindspore import ops, nn, ParameterTuple, context, set_seed, Tensor
from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds
import mindspore as ms
from mindspore.common.initializer import One

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
set_seed(2)


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


def _exec_preprocess(network, is_train, dataset, dataset_sink_mode, epoch_num, sink_size):
    if dataset_sink_mode and not is_train:
        dataset.__loop_size__ = 1

    dataset_helper = DatasetHelper(
        dataset, dataset_sink_mode, sink_size, epoch_num)

    if dataset_sink_mode:
        network = connect_network_with_dataset(network, dataset_helper)

    return dataset_helper, network


def dynamic_shape_sink_process(network, dataset, is_train=True):
    dataset_sink_mode = True
    sink_size = 1
    epoch_num = 1
    dataset_helper, network = _exec_preprocess(
        network, is_train, dataset, dataset_sink_mode, epoch_num, sink_size)
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


def get_columns(tensor_num):
    columns = []
    for i in range(tensor_num):
        columns.append("data" + str(i))
    return columns


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
        self.grad_op = ops.GradOperation(
            get_all=True, get_by_list=True, sens_param=True)
        self.params = ParameterTuple(net.trainable_params())

    def construct(self, *inputs):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(*inputs)


def comm_func(dyn_range, input_shp, data_type, op_net, num=None, output_compare_idx=None):
    list_data = []
    for i in dyn_range:
        tmp_data = []
        for data_shp in input_shp:
            if num is None:
                cur_shp = [dim if dim is not None else i for dim in data_shp]
            else:
                cur_shp = []
                k = 0
                for dim in data_shp:
                    if dim is not None:
                        cur_shp.append(dim)
                    elif k == 1:
                        cur_shp.append(num)
                    else:
                        cur_shp.append(i)
                    k = k + 1
            tmp_data.append(np.random.random(cur_shp).astype(data_type))
        list_data.append(tuple(tmp_data))

    data_map = {}
    dyn_tensors = []
    for i, val in enumerate(input_shp):
        data_map["data" + str(i + 1)] = val
        if None in val:
            dyn_tensors.append(Tensor(dtype=np_type_to_ms(data_type), shape=val))
        else:
            dyn_tensors.append(Tensor(dtype=np_type_to_ms(data_type), shape=val, init=One()))

    dataset = ds.GeneratorDataset(list_data, list(data_map.keys()))
    op_net.set_inputs(*dyn_tensors)

    gradient = dynamic_shape_sink_process(op_net, dataset)
    gradient_cmp = fixed_shape_process(op_net, dataset)
    if output_compare_idx is None:
        assert compare(gradient, gradient_cmp)
    else:
        assert compare(gradient[output_compare_idx], gradient_cmp[output_compare_idx])


class ConcatNet(nn.Cell):
    def __init__(self, axis):
        super(ConcatNet, self).__init__()
        self.op = ops.Concat(axis)

    def construct(self, x1, x2):
        return self.op((x1, x2))


def dynamic_concat_run(is_grad):
    axis = 1
    dtype = np.float32
    data_list = []
    for i in [2, 64]:
        data = []
        data.append(np.random.rand(16, i).astype(dtype))
        data.append(np.random.rand(16, i).astype(dtype))
        if is_grad:
            data.append(np.random.rand(16, i*2).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    dynamic_columns = {column_names[0]: [
        16, None], column_names[1]: [16, None]}
    if is_grad:
        dynamic_columns[column_names[-1]] = [16, None]

    dyn_tensors = []
    for val in dynamic_columns.values():
        dyn_tensors.append(Tensor(dtype=ms.float32, shape=val))

    net = ConcatNet(axis)
    if is_grad:
        net = GradNetWrtX(net)

    net.set_inputs(*dyn_tensors)
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_concat_forward():
    """
    Feature: Test Concat.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_concat_run(False)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_concat_backward():
    """
    Feature: Test backward of Concat.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_concat_run(True)


class BatchNormNet(nn.Cell):
    def __init__(self, c):
        super(BatchNormNet, self).__init__()
        self.bn = nn.BatchNorm1d(c)

    def construct(self, input_data):
        x = self.bn(input_data)
        return x


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_bachnorm():
    """
    Feature: Test BatchNorm and its backward.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    c = 256
    dtype = np.float32
    data_list = []
    for i in [2, 64]:
        data = []
        data.append(np.random.rand(i, c).astype(dtype))
        data.append(np.random.rand(i, c).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    t0 = Tensor(dtype=ms.float32, shape=[None, c])
    t1 = Tensor(dtype=ms.float32, shape=[None, c])
    net = GradNetWrtX(BatchNormNet(c))
    net.set_inputs(t0, t1)
    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


class ReshapeNet(nn.Cell):
    def construct(self, x, y):
        shape_of_y = ops.TensorShape()(y)
        return ops.Reshape()(x, shape_of_y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_reshape():
    """
    Feature: Test Reshape.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dtype = np.float32
    data_list = []
    for i in [2, 96]:
        data = []
        data.append(np.random.rand(i, 64, 1).astype(dtype))
        data.append(np.random.rand(i, 64).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    net = ReshapeNet()

    t0 = Tensor(dtype=ms.float32, shape=[None, 64, 1])
    t1 = Tensor(dtype=ms.float32, shape=[None, 64])
    net.set_inputs(t0, t1)
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


class ReduceSumInputAxisNet(nn.Cell):
    def __init__(self):
        super(ReduceSumInputAxisNet, self).__init__()
        self.reduce = ops.ReduceSum()

    def construct(self, x, y):
        return self.reduce(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_reduce_sum_input_axis():
    """
    Feature: Test ReduceSum with axis is input.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with result of the numpy compute
    """
    dtype = np.float32
    data_list = []
    for i in [2, 96]:
        data = []
        data.append(np.random.rand(i, 256).astype(dtype))
        data.append(np.array([1], dtype=np.int64))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    net = ReduceSumInputAxisNet()
    t0 = Tensor(dtype=ms.float32, shape=[None, 256])
    t1 = Tensor(dtype=ms.int64, shape=[1], init=One())
    net.set_inputs(t0, t1)
    output = dynamic_shape_sink_process(net, dataset)
    # Currently, the parameter axis of ReduceSum operator is dynamic(tensor) is
    # not supported under the fixed shape, so numpy is used for comparison
    inputs = data_list[0]
    output_cmp = np.sum(inputs[0], inputs[1][0])
    assert np.allclose(output.asnumpy(), output_cmp, rtol=1.0e-4, atol=1.0e-4)


class NopNet(nn.Cell):
    def construct(self, x):
        x1 = ops.squeeze(x)
        y1 = ops.expand_dims(x1, 1)
        return ops.sub(y1, x1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_nop():
    """
    Feature: Test Nop.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dtype = np.float32
    data_list = []
    for i in [2, 64]:
        data = []
        data.append(np.random.rand(i, 1).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    net = NopNet()
    t0 = Tensor(dtype=ms.float32, shape=[None, 1])
    net.set_inputs(t0)
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


class ReduceSumNet(nn.Cell):
    def __init__(self, axis=()):
        super(ReduceSumNet, self).__init__()
        self.reduce = ops.ReduceSum()
        self.axis = axis

    def construct(self, x):
        return self.reduce(x, self.axis)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_reduce_sum():
    """
    Feature: Test ReduceSum and its backward.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with result of with fixed shape.
    """
    dtype = np.float32
    data_list = []
    for i in [2, 96]:
        data = []
        data.append(np.random.rand(i, 256).astype(dtype))
        data.append(np.array(1).astype(np.float32))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    net = GradNetWrtX(ReduceSumNet())
    t0 = Tensor(dtype=ms.float32, shape=[None, 256])
    t1 = Tensor(dtype=ms.float32, shape=[], init=One())
    net.set_inputs(t0, t1)
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


class AddNet(nn.Cell):
    def construct(self, x, y):
        return ops.add(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_add():
    """
    Feature: Test add and its backward.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with result of with fixed shape.
    """
    dtype = np.float32
    data_list = []
    for i in [2, 96]:
        data = []
        data.append(np.random.rand(i, 256).astype(dtype))
        data.append(np.random.rand(i, 256).astype(dtype))
        data.append(np.random.rand(i, 256).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    net = GradNetWrtX(AddNet())
    t0 = Tensor(dtype=ms.float32, shape=[None, 256])
    t1 = Tensor(dtype=ms.float32, shape=[None, 256])
    t2 = Tensor(dtype=ms.float32, shape=[None, 256])
    net.set_inputs(t0, t1, t2)
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


class BatchNorm(nn.Cell):
    def __init__(self):
        super(BatchNorm, self).__init__()
        self.batch_norm = ops.BatchNorm()

    def construct(self, input_x, scale, bias, mean, variance):
        out = self.batch_norm(input_x, scale, bias, mean, variance)
        return out


class MaxPool(nn.Cell):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = ops.MaxPool(pad_mode="VALID", kernel_size=2, strides=1)

    def construct(self, x):
        out = self.maxpool(x)
        return out


class SigmoidCrossEntropyWithLogits(nn.Cell):
    def __init__(self):
        super(SigmoidCrossEntropyWithLogits, self).__init__()
        self.op = ops.SigmoidCrossEntropyWithLogits()

    def construct(self, x, y):
        out = self.op(x, y)
        return out


class Sigmoid(nn.Cell):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.op = ops.Sigmoid()

    def construct(self, x):
        out = self.op(x)
        return out


class ResizeNearestNeighbor(nn.Cell):
    def __init__(self):
        super(ResizeNearestNeighbor, self).__init__()
        self.op = ops.ResizeNearestNeighbor((2, 2))

    def construct(self, x):
        out = self.op(x)
        return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm():
    """
    Feature: Test Dynamic batchnorm and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 64), (64,), (64,), (64,), (64,)]
    net = BatchNorm()
    comm_func(dynamic_range, input_shape, data_type, net, output_compare_idx=0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm2():
    """
    Feature: Test Dynamic batchnorm and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(64, None), (None,), (None,), (None,), (None,)]
    net = BatchNorm()
    comm_func(dynamic_range, input_shape, data_type, net, output_compare_idx=0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_maxpool1():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, 16, 32, None)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_maxpool2():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, 16, None, 8)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_maxpool3():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, None, 32, 8)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_maxpool4():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 16, 32, 8)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_sigmoid_cross_entropy_with_logits():
    """
    Feature: Test Dynamic SigmoidCrossEntropyWithLogits and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 16, 32, 8), (None, 16, 32, 8)]
    net = SigmoidCrossEntropyWithLogits()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_sigmoid_cross_entropy_with_logits_grad():
    """
    Feature: Test Dynamic SigmoidCrossEntropyWithLogitsGrad and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(4, 16, None, 8), (4, 16, None, 8), (4, 16, None, 8)]
    net = GradNetWrtX(SigmoidCrossEntropyWithLogits())
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_sigmoid_grad():
    """
    Feature: Test Dynamic SigmoidGrad and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(4, 16, None, 8), (4, 16, None, 8)]
    net = GradNetWrtX(Sigmoid())
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_resize_nearest_neighbor():
    """
    Feature: Test Dynamic ResizeNearestNeighbor and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(4, 16, None, 8)]
    net = ResizeNearestNeighbor()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_resize_nearest_neighbor_grad():
    """
    Feature: Test Dynamic ResizeNearestNeighborGrad and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(4, 16, None, 8), (4, 16, 2, 2)]
    net = GradNetWrtX(ResizeNearestNeighbor())
    comm_func(dynamic_range, input_shape, data_type, net)
