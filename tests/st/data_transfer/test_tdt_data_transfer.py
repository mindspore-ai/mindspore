# Copyright 2021 Huawei Technologies Co., Ltd
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
import time
import numpy as np
import pytest
from mindspore import context, nn, Tensor
from mindspore import log as logger
from mindspore.common.api import _executor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
import mindspore.dataset as de
from mindspore.dataset.vision import c_transforms as c_vision
from mindspore.dataset.transforms import c_transforms as c_trans


DATA_DIR = "/home/workspace/mindspore_dataset/cifar-10-verify-bin"


def dataset_cifar(dataset_path=None, batch_size=32, repeat_num=1, num_rows=9600, distribution_num=None, shard_id=None,
                  drop_remainder=True, usage=None, shuffle=False, num_workers=8, resize_size=32, pad_info=None):
    if dataset_path is None:
        dataset_path = DATA_DIR

    ds = de.Cifar10Dataset(dataset_path, num_samples=num_rows, num_shards=distribution_num, shard_id=shard_id,
                           shuffle=shuffle, usage=usage, num_parallel_workers=num_workers)

    typecast_op = c_trans.TypeCast(mstype.int32)
    ds = ds.map(input_columns="label", operations=typecast_op, num_parallel_workers=num_workers)

    image_op_list = [c_vision.Resize(resize_size),
                     c_vision.Rescale(1.0 / 255.0, 0.0),
                     c_vision.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                     c_vision.HWC2CHW()]
    ds = ds.map(input_columns="image", operations=image_op_list, num_parallel_workers=num_workers)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder, num_parallel_workers=num_workers, pad_info=pad_info)
    ds = ds.repeat(repeat_num)

    return ds


def op_network_with_epoch(network, step_num):
    iter_num = 0
    network.set_train()
    for _ in range(step_num):
        op_return = network()
        op_return = op_return.asnumpy()
        logger.info("Op_return is : %s", op_return)
        iter_num += 1
        logger.info("Iter Num : %s", iter_num)

    return iter_num


def convert_type(shapes, types):
    ms_types = []
    for np_shape, np_type in zip(shapes, types):
        input_np = np.zeros(np_shape, np_type)
        tensor = Tensor(input_np)
        ms_types.append(tensor.dtype)
    return ms_types


def get_dataset_base_value(dataset):
    dataset_size = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()
    return dataset_size, batch_size


def dataset_send_tdt(dataset):
    time.sleep(1)
    dataset.send(1)


def get_dataset_shapes_and_types(dataset):
    dataset_shapes = dataset.output_shapes()
    np_types = dataset.output_types()
    dataset_types = convert_type(dataset_shapes, np_types)
    return dataset_shapes, dataset_types


class SingleOpNetwork(nn.Cell):
    def __init__(self, shapes):
        super(SingleOpNetwork, self).__init__()
        self.shapes = tuple(shapes[0])
        self.Op_Reshape_network = P.Reshape()

    def construct(self, network_input):
        return self.Op_Reshape_network(network_input, self.shapes)


class NetWithTDT(nn.Cell):
    def __init__(self, network, dataset_types, dataset_shapes, shared_name=''):
        super(NetWithTDT, self).__init__()
        self.get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_shapes), shared_name)
        self.Op_network = network

    def construct(self):
        next_input, _ = self.get_next()
        return self.Op_network(next_input)


def op_network_with_step_num(dataset, step_num):
    dataset_shapes, dataset_types = get_dataset_shapes_and_types(dataset)
    _, batch_size = get_dataset_base_value(dataset)
    dataset = dataset.device_que()
    queue_name = dataset.queue_name

    net = SingleOpNetwork(dataset_shapes)
    net_with_dataset = NetWithTDT(net, dataset_types, dataset_shapes, queue_name)
    # when device type is Davinci, net should has get_next operation before call init_dataset
    _executor.init_dataset(dataset.queue_name, 1, batch_size, dataset_types, dataset_shapes, (), "")
    dataset_send_tdt(dataset)
    return op_network_with_epoch(net_with_dataset, step_num)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tdt_consume_beyond_produce():
    context.set_context(mode=context.GRAPH_MODE)

    batch_size = 64
    repeat_num = 1
    num_rows = 640
    beyond_step_num = 1000
    ds = dataset_cifar(batch_size=batch_size, repeat_num=repeat_num, num_rows=num_rows)

    try:
        iter_num = op_network_with_step_num(ds, step_num=beyond_step_num)
        logger.info("out_iter_num：%s", iter_num)
        assert False
    except RuntimeError as e:
        logger.info("when dataset batch num is less than train loop, error msg is %s", e)
        assert True


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_tdt_produce_beyond_consume():
    context.set_context(mode=context.GRAPH_MODE)

    batch_size = 64
    repeat_num = 1
    num_rows = 6400
    beyond_step_num = 10
    ds = dataset_cifar(batch_size=batch_size, repeat_num=repeat_num, num_rows=num_rows)

    iter_num = op_network_with_step_num(ds, step_num=beyond_step_num)
    logger.info("out_iter_num：%s", iter_num)
    assert iter_num == 10
