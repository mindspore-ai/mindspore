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

import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


_anchor_box = Tensor(np.ones([32, 4]), ms.float32)
_gt_boxes = Tensor(np.ones([32, 4]), ms.float32)


class Net(Cell):
    """
    Create the test net.
    """
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.bbox_encode = P.BoundingBoxEncode().shard(strategy)

    def construct(self, anchor_boxes, gt_boxes):
        x = self.bbox_encode(anchor_boxes, gt_boxes)
        return x


def compile_net(net: Cell, *inputs):
    net.set_train()
    _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()


def test_bounding_box_encode_data_parallel():
    """
    Feature: test BoundingBoxEncode data parallel strategy
    Description: only shard the batch dimension
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1), (8, 1))
    net = Net(strategy)
    compile_net(net, _anchor_box, _gt_boxes)


def test_bounding_box_encode_auto_parallel():
    """
    Feature: test BoundingBoxEncode auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, _anchor_box, _gt_boxes)


def test_bounding_box_encode_strategy_error():
    """
    Feature: test BoundingBoxEncode with illegal strategy
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((8, 1), (4, 1))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, _anchor_box, _gt_boxes)
    context.reset_auto_parallel_context()
