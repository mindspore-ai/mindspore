# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Regression guards for retained shape, index and image operations."""
import numpy as np
import pytest

from mindspore import Tensor, nn, ops
from mindspore.ops import deprecated, operations
from mindspore.ops.operations import _inner_ops, array_ops, image_ops, other_ops, sparse_ops
from mindspore.parallel._utils import _get_python_op
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("module,name,args", [
    (deprecated, "BatchToSpaceND", ([2, 2], [[0, 0], [0, 0]])),
    (array_ops, "SpaceToBatchND", ([2, 2], [[0, 0], [0, 0]])),
    (array_ops, "MatrixDiagPartV3", ()), (array_ops, "MatrixDiagV3", ()),
    (array_ops, "TensorShape", ()), (array_ops, "ScalarToTensor", ()), (array_ops, "GatherNd", ()),
    (other_ops, "CheckValid", ()), (other_ops, "IOU", ()), (image_ops, "CropAndResize", ()),
    (sparse_ops, "SparseTensorDenseMatmul", ()),
])
def test_private_shape_lookup(module, name, args):
    """
    Feature: Public shape and image primitive retirement.
    Description: Construct operators by name from both internal parallel lookup paths.
    Expectation: Public lookup is unavailable; internal lookup preserves class identity.
    """
    assert not hasattr(ops, name) and name not in ops.__all__
    assert not hasattr(operations, name) and name not in operations.__all__
    cls = getattr(module, name)
    assert getattr(_inner_ops, name) is cls
    for path in ("mindspore.ops.operations", "mindspore.ops.operations._inner_ops"):
        assert _get_python_op(name, path, 'shape_lookup', list(args)).__class__ is cls


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_retained_gather_nd_gradient():
    """
    Feature: Retained gather_nd functional API and its gradient.
    Description: Gather the same row twice and differentiate a sum of the returned values.
    Expectation: Both reads match and both gradient contributions accumulate in that row.
    """
    data = Tensor(np.arange(6, dtype=np.float32).reshape(3, 2))
    indices = Tensor(np.array([[1], [1]], np.int32))

    def gather_sum(values):
        return ops.gather_nd(values, indices).sum()

    np.testing.assert_array_equal(ops.gather_nd(data, indices).asnumpy(), [[2, 3], [2, 3]])
    np.testing.assert_array_equal(ops.grad(gather_sum)(data).asnumpy(), [[0, 0], [2, 2], [0, 0]])
    assert operations.GatherD is array_ops.GatherD


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_retained_shape_and_diagonal_functions():
    """
    Feature: Retained shape, scalar and matrix diagonal functions.
    Description: Inspect a rectangular tensor and build a matrix from a nonconstant diagonal.
    Expectation: Shapes, scalar values and diagonal elements agree with NumPy.
    """
    data = Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
    np.testing.assert_array_equal(_inner_ops.TensorShape()(data).asnumpy(), [2, 3])
    np.testing.assert_array_equal(ops.scalar_to_tensor(7).asnumpy(), 7)
    values = Tensor(np.array([1, 2, 3], np.float32))
    diagonal = _inner_ops.MatrixDiagV3()(values, Tensor(np.int32(0)), Tensor(np.int32(-1)),
                                         Tensor(np.int32(-1)), Tensor(np.float32(0)))
    np.testing.assert_array_equal(diagonal.asnumpy(), np.diag(values.asnumpy()))
    result = _inner_ops.MatrixDiagPartV3()(diagonal, Tensor(np.int32(0)), Tensor(np.float32(0)))
    np.testing.assert_array_equal(result.asnumpy(), values.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_retained_sparse_layer():
    """
    Feature: Retained nn sparse layer.
    Description: Multiply a sparse diagonal matrix by a dense column through the nn interface.
    Expectation: Removing the public Primitive does not change the layer's result.
    """
    indices = Tensor(np.array([[0, 0], [1, 1]], np.int32))
    values = Tensor(np.array([2, 3], np.float32))
    dense = Tensor(np.array([[4], [5]], np.float32))
    result = nn.SparseTensorDenseMatmul()(indices, values, (2, 2), dense)
    np.testing.assert_array_equal(result.asnumpy(), [[8], [15]])
