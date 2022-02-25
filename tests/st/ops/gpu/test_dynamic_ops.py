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

from mindspore import ops, nn, ParameterTuple, context, set_seed
from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
set_seed(2)


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
        data.append(np.random.rand(i, 16).astype(dtype))
        data.append(np.random.rand(i, 32).astype(dtype))
        if is_grad:
            data.append(np.random.rand(i, 48).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    dynamic_columns = {column_names[0]: [
        None, 16], column_names[1]: [None, 32]}
    if is_grad:
        dynamic_columns[column_names[-1]] = [None, 48]
    dataset.set_dynamic_columns(columns=dynamic_columns)
    net = ConcatNet(axis)
    if is_grad:
        net = GradNetWrtX(net)
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_concat_forward():
    """
    Feature: Test Concat.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_concat_run(False)


@pytest.mark.level0
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


@pytest.mark.level0
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
    dynamic_columns = {column_names[0]: [None, c], column_names[1]: [None, c]}
    dataset.set_dynamic_columns(columns=dynamic_columns)
    net = GradNetWrtX(BatchNormNet(c))
    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


class ReshapeNet(nn.Cell):
    def construct(self, x, y):
        shape_of_y = ops.DynamicShape()(y)
        return ops.Reshape()(x, shape_of_y)


@pytest.mark.level0
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
    dynamic_columns = {column_names[0]: [
        None, 64, 1], column_names[1]: [None, 64]}
    dataset.set_dynamic_columns(columns=dynamic_columns)
    net = ReshapeNet()
    output = dynamic_shape_sink_process(net, dataset)
    output_cmp = fixed_shape_process(net, dataset)
    assert compare(output, output_cmp)


class ReduceSumNet(nn.Cell):
    def __init__(self):
        super(ReduceSumNet, self).__init__()
        self.reduce = ops.ReduceSum()

    def construct(self, x, y):
        return self.reduce(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_reduce_sum():
    """
    Feature: Test ReduceSum.
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
    dynamic_columns = {column_names[0]: [None, 256]}
    dataset.set_dynamic_columns(columns=dynamic_columns)
    net = ReduceSumNet()
    output = dynamic_shape_sink_process(net, dataset)
    # Currently, the parameter axis of ReduceSum operator is dynamic(tensor) is
    # not supported under the fixed shape, so numpy is used for comparison
    inputs = data_list[0]
    output_cmp = np.sum(inputs[0], inputs[1][0])
    assert np.allclose(output.asnumpy(), output_cmp, rtol=1.0e-4, atol=1.0e-4)
