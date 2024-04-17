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

import numpy as np
import pytest

from mindspore import Tensor, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


BATCH_SIZE = 32
NUM_BOXES = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
_images = Tensor(np.random.normal(size=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]).astype(np.float32))
_boxes = Tensor(np.random.uniform(size=[NUM_BOXES, 4]).astype(np.float32))
_box_index = Tensor(np.random.uniform(size=[NUM_BOXES], low=0, high=BATCH_SIZE).astype(np.int32))
_crop_size = (24, 24)


class Net(Cell):
    def __init__(self, crop_size, strategy=None):
        super(Net, self).__init__()
        self.crop_size = crop_size
        self.crop_and_resize = P.CropAndResize().shard(strategy)

    def construct(self, images, boxes, box_index):
        output = self.crop_and_resize(images, boxes, box_index, self.crop_size)
        return output


def compile_net(net: Cell, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()


def test_crop_and_resize_auto_parallel():
    """
    Feature: test CropAndResize auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net(_crop_size)
    compile_net(net, _images, _boxes, _box_index)


def test_crop_and_resize_data_parallel():
    """
    Feature: test CropAndResize data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 1, 1), (2, 1), (2,))
    net = Net(_crop_size, strategy)
    compile_net(net, _images, _boxes, _box_index)


def test_crop_and_resize_strategy_error():
    """
    Feature: test invalid strategy for CropAndResize
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 1, 1), (2, 1), (1,))
    net = Net(_crop_size, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, _images, _boxes, _box_index)
    context.reset_auto_parallel_context()


def test_crop_and_resize_dynamic_shape_constraint():
    """
    Feature: test CropAndResize dynamic shape
    Description: data parallel
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0, full_batch=False)
    strategy = ((4, 1, 1, 1), (2, 1), (2,))
    net = Net(_crop_size, strategy)
    dynamic_images = Tensor(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS], dtype=mstype.float32)
    with pytest.raises(RuntimeError):
        compile_net(net, dynamic_images, _boxes, _box_index)
