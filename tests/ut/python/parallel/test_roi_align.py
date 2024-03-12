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
from mindspore import Tensor, context
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


POOLED_HEIGHT = 2
POOLED_WIDTH = 2
SPATIAL_SCALE = 0.5
BATCH_SIZE = 32
FEATURES_HEIGHT = 256
FEATURES_WIDTH = 256
CHANNELS = 3
NUM_ROIS = 16
_features = Tensor(np.random.normal(size=[BATCH_SIZE, CHANNELS, FEATURES_HEIGHT, FEATURES_WIDTH]).astype(np.float32))
_rois = Tensor(
    np.hstack((np.random.randint(0, BATCH_SIZE, [NUM_ROIS, 1]).astype(np.float32),
               np.random.uniform(low=0, high=FEATURES_HEIGHT / SPATIAL_SCALE, size=[NUM_ROIS, 4]).astype(np.float32))))


class Net(Cell):
    def __init__(self, pooled_h, pooled_w, spatial_scale, strategy=None):
        super(Net, self).__init__()
        self.roi_align = P.ROIAlign(pooled_h, pooled_w, spatial_scale).shard(strategy)

    def construct(self, features, rois):
        output = self.roi_align(features, rois)
        return output


def compile_net(net: Cell, *inputs):
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


def test_roi_align_auto_parallel():
    """
    Feature: test ROIAlign auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net(POOLED_HEIGHT, POOLED_WIDTH, SPATIAL_SCALE)
    compile_net(net, _features, _rois)


def test_roi_align_data_parallel():
    """
    Feature: test ROIAlign data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 1, 1), (2, 1))
    net = Net(POOLED_HEIGHT, POOLED_WIDTH, SPATIAL_SCALE, strategy)
    compile_net(net, _features, _rois)


def test_roi_align_strategy_error():
    """
    Feature: test invalid strategy for ROIAlign
    Description: illegal strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 1, 2, 2), (1, 1))
    net = Net(POOLED_HEIGHT, POOLED_WIDTH, SPATIAL_SCALE, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, _features, _rois)
    context.reset_auto_parallel_context()


def test_roi_align_layout():
    """
    Features: ROIAlignInfo
    Description: validate layout and structure
    Expectation: No raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1, 1, 1), (2, 1))
    net = Net(POOLED_HEIGHT, POOLED_WIDTH, SPATIAL_SCALE, strategy)
    phase = compile_net(net, _features, _rois)

    validator = ParallelValidator(net, phase)
    # check layout
    features_expect_layout = ([8], [-1, -1, -1, -1], [32, 3, 256, 256], 0, True, '')
    assert validator.check_parameter_layout('features', features_expect_layout)

    # check attrs
    roi_expect_attrs = {'pooled_height': POOLED_HEIGHT, 'pooled_width': POOLED_WIDTH, 'spatial_scale': SPATIAL_SCALE}
    assert validator.check_node_attrs('ROIAlign-0', roi_expect_attrs)

    # check inputs
    roi_expect_inputs = ['StridedSlice-0', 'TensorScatterUpdate-0']
    assert validator.check_node_inputs('ROIAlign-0', roi_expect_inputs)

    # check sub_graph
    sub_graph = {
        'TensorScatterUpdate-0': ['StridedSlice-1', 'Stack-0', 'Minimum-0'],
        'Equal-0': ['Sub-0', 'Minimum-0'],
        'ROIAlign-0': ['StridedSlice-0', 'TensorScatterUpdate-0'],
        'Mul-0': ['ROIAlign-0', 'ExpandDims-2'],
        'AllReduce-0': ['Mul-0']
    }
    assert validator.check_graph_structure(sub_graph)


def test_roi_align_dynamic_shape_constraint():
    """
    Feature: test ROIAlign dynamic shape
    Description: data parallel
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0, full_batch=False)
    strategy = ((4, 1, 1, 1), (2, 1))
    net = Net(POOLED_HEIGHT, POOLED_WIDTH, SPATIAL_SCALE, strategy)

    input_features = Tensor(shape=[None, CHANNELS, FEATURES_HEIGHT, FEATURES_WIDTH], dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_net(net, input_features, _rois)
