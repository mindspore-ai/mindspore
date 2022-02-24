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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dynamic_concat():
    """
    Feature: Test Concat and its backward.
    Description:  The shape of inputs is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    axis = 1
    dtype = np.float32
    data_list = []
    for i in [2, 64]:
        data = []
        data.append(np.random.rand(i, 16).astype(dtype))
        data.append(np.random.rand(i, 32).astype(dtype))
        data.append(np.random.rand(i, 48).astype(dtype))
        data_list.append(tuple(data))
    column_names = get_columns(len(data_list[0]))
    dataset = ds.GeneratorDataset(data_list, column_names, shuffle=False)
    dataset.set_dynamic_columns(
        columns={column_names[0]: [None, 16], column_names[1]: [None, 32], column_names[2]: [None, 48]})

    net = GradNetWrtX(ConcatNet(axis))

    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)
