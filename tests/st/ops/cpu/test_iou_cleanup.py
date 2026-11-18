# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Guard the IOU primitive and FasterRCNN consumer after removing ops.iou."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn
from mindspore.ops.operations import _inner_ops
from tests.mark_utils import arg_mark


@pytest.fixture(name='label_box_fn')
def load_label_box(monkeypatch):
    """Load real model modules without executing the standalone network entry point."""
    source = Path(__file__).resolve().parents[2] / 'networks/models/fasterrcnn/src'
    namespace = '_iou_cleanup_fasterrcnn'
    for suffix, path in (('', source), ('.utils', source / 'utils')):
        package = types.ModuleType(namespace + suffix)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, namespace + suffix, package)
    for suffix in ('utils.box_utils', 'label_assignment'):
        name = namespace + '.' + suffix
        spec = importlib.util.spec_from_file_location(name, source / (suffix.replace('.', '/') + '.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module.label_box


@pytest.fixture(autouse=True)
def restore_context():
    """Restore the caller's execution mode after each case."""
    original_mode = ms.get_context('mode')
    yield
    ms.set_context(mode=original_mode)


class IOUNet(nn.Cell):
    """Exercise the retained primitive in both execution modes."""

    def __init__(self, mode):
        super().__init__()
        self.overlap = _inner_ops.IOU(mode=mode)

    def construct(self, anchors, ground_truth):
        return self.overlap(anchors, ground_truth)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('execution_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('overlap_mode', ['iou', 'iof'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_retained_iou_values(execution_mode, overlap_mode, dtype):
    """
    Feature: Retained IOU primitive.
    Description: Compute unequal-size, disjoint and identical boxes in IOU/IOF modes.
    Expectation: Output shape, dtype and values match inclusive-coordinate NumPy arithmetic.
    """
    ms.set_context(mode=execution_mode)
    anchors = np.array([[0, 0, 9, 9], [20, 20, 29, 29], [0, 0, 4, 4]], dtype=dtype)
    ground_truth = np.array([[0, 0, 9, 9], [20, 20, 24, 24]], dtype=dtype)
    top_left = np.maximum(ground_truth[:, None, :2], anchors[None, :, :2])
    bottom_right = np.minimum(ground_truth[:, None, 2:], anchors[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left + 1, 0), axis=-1)
    gt_area = np.prod(ground_truth[:, 2:] - ground_truth[:, :2] + 1, axis=-1)[:, None]
    anchor_area = np.prod(anchors[:, 2:] - anchors[:, :2] + 1, axis=-1)[None, :]
    denominator = gt_area if overlap_mode == 'iof' else gt_area + anchor_area - intersection
    output = IOUNet(overlap_mode)(Tensor(anchors), Tensor(ground_truth)).asnumpy()
    assert output.shape == (2, 3)
    assert output.dtype == dtype
    np.testing.assert_allclose(output, intersection / denominator, rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('execution_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('allow_low_quality', [False, True])
def test_fasterrcnn_label_box_after_iou_cleanup(execution_mode, allow_low_quality, label_box_fn):
    """
    Feature: FasterRCNN IOU consumer migration.
    Description: Match exact, low-quality and invalid ground truth with the retained IOU primitive.
    Expectation: Matches and positive/background labels retain their original semantics.
    """
    ms.set_context(mode=execution_mode)
    anchors = Tensor(np.array([[0, 0, 9, 9], [20, 20, 29, 29], [40, 40, 49, 49]], np.float32))
    boxes = Tensor(np.array([[1, 0, 0, 9, 9], [2, 20, 20, 24, 24], [-1, 40, 40, 49, 49]], np.float32))
    match = ms.jit(label_box_fn) if execution_mode == ms.GRAPH_MODE else label_box_fn
    indices, labels = match(anchors, boxes, 0.7, 0.3, allow_low_quality)
    np.testing.assert_array_equal(indices.asnumpy(), [0, 1, 0])
    np.testing.assert_array_equal(labels.asnumpy(), [1, int(allow_low_quality), 0])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('execution_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fasterrcnn_invalid_ground_truth_after_iou_cleanup(execution_mode, label_box_fn):
    """
    Feature: FasterRCNN invalid ground-truth handling.
    Description: Supply only padded invalid ground-truth rows after migrating the IOU call.
    Expectation: Every anchor remains ignored rather than becoming a positive or background sample.
    """
    ms.set_context(mode=execution_mode)
    anchors = Tensor(np.array([[0, 0, 9, 9], [20, 20, 29, 29]], np.float32))
    boxes = Tensor(np.array([[-1, 0, 0, 9, 9]], np.float32))
    match = ms.jit(label_box_fn) if execution_mode == ms.GRAPH_MODE else label_box_fn
    _, labels = match(anchors, boxes, 0.7, 0.3, True)
    np.testing.assert_array_equal(labels.asnumpy(), [-1, -1])
