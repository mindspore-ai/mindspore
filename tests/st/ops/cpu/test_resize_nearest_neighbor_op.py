# Copyright 2020 Huawei Technologies Co., Ltd
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

from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore import nn

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetResizeNearestNeighbor(nn.Cell):
    def __init__(self, size=None, align_corners=False):
        super(NetResizeNearestNeighbor, self).__init__()
        self.op = P.ResizeNearestNeighbor(size=size, align_corners=align_corners)

    def construct(self, inputs):
        return self.op(inputs)


def resize_nn_grayscale_integer_ratio(datatype):
    input_tensor = Tensor(np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeNearestNeighbor((9, 9))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
                                         [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
                                         [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
                                         [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
                                         [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
                                         [0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
                                         [0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
                                         [0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
                                         [0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h and w
    resize_nn = NetResizeNearestNeighbor((1, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, larger w
    resize_nn = NetResizeNearestNeighbor((1, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.1, 0.2, 0.2, 0.3, 0.3]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, smaller w
    resize_nn = NetResizeNearestNeighbor((6, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1], [0.1], [0.4], [0.4], [0.7], [0.7]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, same w
    resize_nn = NetResizeNearestNeighbor((1, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, same w
    resize_nn = NetResizeNearestNeighbor((6, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3],
                                         [0.1, 0.2, 0.3],
                                         [0.4, 0.5, 0.6],
                                         [0.4, 0.5, 0.6],
                                         [0.7, 0.8, 0.9],
                                         [0.7, 0.8, 0.9]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, smaller w
    resize_nn = NetResizeNearestNeighbor((3, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1], [0.4], [0.7]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, larger w
    resize_nn = NetResizeNearestNeighbor((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
                                         [0.4, 0.4, 0.5, 0.5, 0.6, 0.6],
                                         [0.7, 0.7, 0.8, 0.8, 0.9, 0.9]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same w, same h (identity)
    resize_nn = NetResizeNearestNeighbor((3, 3))
    output = resize_nn(input_tensor)
    np.testing.assert_array_equal(output.asnumpy(), input_tensor.asnumpy())

def resize_nn_grayscale_not_integer_ratio(datatype):
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 0.0, 0.1, 0.2]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeNearestNeighbor((7, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
                                         [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
                                         [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
                                         [0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8],
                                         [0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8],
                                         [0.9, 0.9, 0.0, 0.0, 0.1, 0.1, 0.2],
                                         [0.9, 0.9, 0.0, 0.0, 0.1, 0.1, 0.2]]]]).astype(datatype))

    # smaller h and w
    resize_nn = NetResizeNearestNeighbor((2, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, larger w
    resize_nn = NetResizeNearestNeighbor((2, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
                                         [0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, smaller w
    resize_nn = NetResizeNearestNeighbor((5, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3],
                                         [0.1, 0.2, 0.3],
                                         [0.5, 0.6, 0.7],
                                         [0.5, 0.6, 0.7],
                                         [0.9, 0.0, 0.1]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, same w
    resize_nn = NetResizeNearestNeighbor((2, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.5, 0.6, 0.7, 0.8]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, same w
    resize_nn = NetResizeNearestNeighbor((8, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.1, 0.2, 0.3, 0.4],
                                         [0.1, 0.2, 0.3, 0.4],
                                         [0.5, 0.6, 0.7, 0.8],
                                         [0.5, 0.6, 0.7, 0.8],
                                         [0.5, 0.6, 0.7, 0.8],
                                         [0.9, 0.0, 0.1, 0.2],
                                         [0.9, 0.0, 0.1, 0.2]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, smaller w
    resize_nn = NetResizeNearestNeighbor((3, 2))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.3],
                                         [0.5, 0.7],
                                         [0.9, 0.1]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, larger w
    resize_nn = NetResizeNearestNeighbor((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1, 0.2, 0.3, 0.3, 0.4],
                                         [0.5, 0.5, 0.6, 0.7, 0.7, 0.8],
                                         [0.9, 0.9, 0.0, 0.1, 0.1, 0.2]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same w, same h (identity)
    resize_nn = NetResizeNearestNeighbor((3, 4))
    output = resize_nn(input_tensor)
    np.testing.assert_array_equal(output.asnumpy(), input_tensor.asnumpy())

def resize_nn_rgb_integer_ratio(datatype):
    input_tensor = Tensor(np.array(
        [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
          [[111, 112, 113], [114, 115, 116], [117, 118, 119]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeNearestNeighbor((9, 9))
    output = resize_nn(input_tensor)
    expected_output_array = np.array([[[[1, 1, 1, 2, 2, 2, 3, 3, 3],
                                        [1, 1, 1, 2, 2, 2, 3, 3, 3],
                                        [1, 1, 1, 2, 2, 2, 3, 3, 3],
                                        [4, 4, 4, 5, 5, 5, 6, 6, 6],
                                        [4, 4, 4, 5, 5, 5, 6, 6, 6],
                                        [4, 4, 4, 5, 5, 5, 6, 6, 6],
                                        [7, 7, 7, 8, 8, 8, 9, 9, 9],
                                        [7, 7, 7, 8, 8, 8, 9, 9, 9],
                                        [7, 7, 7, 8, 8, 8, 9, 9, 9]],
                                       [[11, 11, 11, 12, 12, 12, 13, 13, 13],
                                        [11, 11, 11, 12, 12, 12, 13, 13, 13],
                                        [11, 11, 11, 12, 12, 12, 13, 13, 13],
                                        [14, 14, 14, 15, 15, 15, 16, 16, 16],
                                        [14, 14, 14, 15, 15, 15, 16, 16, 16],
                                        [14, 14, 14, 15, 15, 15, 16, 16, 16],
                                        [17, 17, 17, 18, 18, 18, 19, 19, 19],
                                        [17, 17, 17, 18, 18, 18, 19, 19, 19],
                                        [17, 17, 17, 18, 18, 18, 19, 19, 19]],
                                       [[111, 111, 111, 112, 112, 112, 113, 113, 113],
                                        [111, 111, 111, 112, 112, 112, 113, 113, 113],
                                        [111, 111, 111, 112, 112, 112, 113, 113, 113],
                                        [114, 114, 114, 115, 115, 115, 116, 116, 116],
                                        [114, 114, 114, 115, 115, 115, 116, 116, 116],
                                        [114, 114, 114, 115, 115, 115, 116, 116, 116],
                                        [117, 117, 117, 118, 118, 118, 119, 119, 119],
                                        [117, 117, 117, 118, 118, 118, 119, 119, 119],
                                        [117, 117, 117, 118, 118, 118, 119, 119, 119]]]])
    expected_output = Tensor(np.array(expected_output_array).astype(datatype))

    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h and w
    resize_nn = NetResizeNearestNeighbor((1, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[1]], [[11]], [[111]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, larger w
    resize_nn = NetResizeNearestNeighbor((1, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 1, 2, 2, 3, 3]],
                                        [[11, 11, 12, 12, 13, 13]],
                                        [[111, 111, 112, 112, 113, 113]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, smaller w
    resize_nn = NetResizeNearestNeighbor((6, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1], [1], [4], [4], [7], [7]],
                                        [[11], [11], [14], [14], [17], [17]],
                                        [[111], [111], [114], [114], [117], [117]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, same w
    resize_nn = NetResizeNearestNeighbor((1, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 2, 3]],
                                        [[11, 12, 13]],
                                        [[111, 112, 113]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, same w
    resize_nn = NetResizeNearestNeighbor((6, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 2, 3],
                                         [1, 2, 3],
                                         [4, 5, 6],
                                         [4, 5, 6],
                                         [7, 8, 9],
                                         [7, 8, 9]],
                                        [[11, 12, 13],
                                         [11, 12, 13],
                                         [14, 15, 16],
                                         [14, 15, 16],
                                         [17, 18, 19],
                                         [17, 18, 19]],
                                        [[111, 112, 113],
                                         [111, 112, 113],
                                         [114, 115, 116],
                                         [114, 115, 116],
                                         [117, 118, 119],
                                         [117, 118, 119]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, smaller w
    resize_nn = NetResizeNearestNeighbor((3, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1], [4], [7]],
                                        [[11], [14], [17]],
                                        [[111], [114], [117]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, larger w
    resize_nn = NetResizeNearestNeighbor((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 1, 2, 2, 3, 3],
                                         [4, 4, 5, 5, 6, 6],
                                         [7, 7, 8, 8, 9, 9]],
                                        [[11, 11, 12, 12, 13, 13],
                                         [14, 14, 15, 15, 16, 16],
                                         [17, 17, 18, 18, 19, 19]],
                                        [[111, 111, 112, 112, 113, 113],
                                         [114, 114, 115, 115, 116, 116],
                                         [117, 117, 118, 118, 119, 119]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same w, same h (identity)
    resize_nn = NetResizeNearestNeighbor((3, 3))
    output = resize_nn(input_tensor)
    np.testing.assert_array_equal(output.asnumpy(), input_tensor.asnumpy())

def resize_nn_rgb_not_integer_ratio(datatype):
    input_tensor = Tensor(np.array([[[[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 0, 1, 2]],
                                     [[11, 12, 13, 14],
                                      [15, 16, 17, 18],
                                      [19, 10, 11, 12]],
                                     [[111, 112, 113, 114],
                                      [115, 116, 117, 118],
                                      [119, 110, 111, 112]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeNearestNeighbor((7, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 1, 2, 2, 3, 3, 4],
                                         [1, 1, 2, 2, 3, 3, 4],
                                         [1, 1, 2, 2, 3, 3, 4],
                                         [5, 5, 6, 6, 7, 7, 8],
                                         [5, 5, 6, 6, 7, 7, 8],
                                         [9, 9, 0, 0, 1, 1, 2],
                                         [9, 9, 0, 0, 1, 1, 2]],
                                        [[11, 11, 12, 12, 13, 13, 14],
                                         [11, 11, 12, 12, 13, 13, 14],
                                         [11, 11, 12, 12, 13, 13, 14],
                                         [15, 15, 16, 16, 17, 17, 18],
                                         [15, 15, 16, 16, 17, 17, 18],
                                         [19, 19, 10, 10, 11, 11, 12],
                                         [19, 19, 10, 10, 11, 11, 12]],
                                        [[111, 111, 112, 112, 113, 113, 114],
                                         [111, 111, 112, 112, 113, 113, 114],
                                         [111, 111, 112, 112, 113, 113, 114],
                                         [115, 115, 116, 116, 117, 117, 118],
                                         [115, 115, 116, 116, 117, 117, 118],
                                         [119, 119, 110, 110, 111, 111, 112],
                                         [119, 119, 110, 110, 111, 111, 112]]]]).astype(datatype))

    # smaller h and w
    resize_nn = NetResizeNearestNeighbor((2, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 2, 3], [5, 6, 7]],
                                        [[11, 12, 13], [15, 16, 17]],
                                        [[111, 112, 113], [115, 116, 117]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, larger w
    resize_nn = NetResizeNearestNeighbor((2, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 1, 2, 2, 3, 3, 4],
                                         [5, 5, 6, 6, 7, 7, 8]],
                                        [[11, 11, 12, 12, 13, 13, 14],
                                         [15, 15, 16, 16, 17, 17, 18]],
                                        [[111, 111, 112, 112, 113, 113, 114],
                                         [115, 115, 116, 116, 117, 117, 118]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, smaller w
    resize_nn = NetResizeNearestNeighbor((5, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 2, 3],
                                         [1, 2, 3],
                                         [5, 6, 7],
                                         [5, 6, 7],
                                         [9, 0, 1]],
                                        [[11, 12, 13],
                                         [11, 12, 13],
                                         [15, 16, 17],
                                         [15, 16, 17],
                                         [19, 10, 11]],
                                        [[111, 112, 113],
                                         [111, 112, 113],
                                         [115, 116, 117],
                                         [115, 116, 117],
                                         [119, 110, 111]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # smaller h, same w
    resize_nn = NetResizeNearestNeighbor((2, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 2, 3, 4],
                                         [5, 6, 7, 8]],
                                        [[11, 12, 13, 14],
                                         [15, 16, 17, 18]],
                                        [[111, 112, 113, 114],
                                         [115, 116, 117, 118]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # larger h, same w
    resize_nn = NetResizeNearestNeighbor((8, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 2, 3, 4],
                                         [1, 2, 3, 4],
                                         [1, 2, 3, 4],
                                         [5, 6, 7, 8],
                                         [5, 6, 7, 8],
                                         [5, 6, 7, 8],
                                         [9, 0, 1, 2],
                                         [9, 0, 1, 2]],
                                        [[11, 12, 13, 14],
                                         [11, 12, 13, 14],
                                         [11, 12, 13, 14],
                                         [15, 16, 17, 18],
                                         [15, 16, 17, 18],
                                         [15, 16, 17, 18],
                                         [19, 10, 11, 12],
                                         [19, 10, 11, 12]],
                                        [[111, 112, 113, 114],
                                         [111, 112, 113, 114],
                                         [111, 112, 113, 114],
                                         [115, 116, 117, 118],
                                         [115, 116, 117, 118],
                                         [115, 116, 117, 118],
                                         [119, 110, 111, 112],
                                         [119, 110, 111, 112]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, smaller w
    resize_nn = NetResizeNearestNeighbor((3, 2))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 3], [5, 7], [9, 1]],
                                        [[11, 13], [15, 17], [19, 11]],
                                        [[111, 113], [115, 117], [119, 111]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same h, larger w
    resize_nn = NetResizeNearestNeighbor((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[1, 1, 2, 3, 3, 4],
                                         [5, 5, 6, 7, 7, 8],
                                         [9, 9, 0, 1, 1, 2]],
                                        [[11, 11, 12, 13, 13, 14],
                                         [15, 15, 16, 17, 17, 18],
                                         [19, 19, 10, 11, 11, 12]],
                                        [[111, 111, 112, 113, 113, 114],
                                         [115, 115, 116, 117, 117, 118],
                                         [119, 119, 110, 111, 111, 112]]]]).astype(datatype))
    np.testing.assert_array_equal(expected_output.asnumpy(), output.asnumpy())

    # same w, same h (identity)
    resize_nn = NetResizeNearestNeighbor((3, 4))
    output = resize_nn(input_tensor)
    np.testing.assert_array_equal(output.asnumpy(), input_tensor.asnumpy())

def resize_nn_rgb_multiple(datatype):
    input_tensor = Tensor(np.array([[[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                     [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                                     [[111, 112, 113, 114, 115], [116, 117, 118, 119, 120]]],
                                    [[[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
                                     [[111, 112, 113, 114, 115], [116, 117, 118, 119, 120]],
                                     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]],
                                    [[[111, 112, 113, 114, 115], [116, 117, 118, 119, 120]],
                                     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                     [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]]]).astype(datatype))

    resize_nn = NetResizeNearestNeighbor((5, 2))
    output = resize_nn(input_tensor)

    expected_output = Tensor(np.array([[[[1, 3], [1, 3], [1, 3], [6, 8], [6, 8]],
                                        [[11, 13], [11, 13], [11, 13], [16, 18], [16, 18]],
                                        [[111, 113], [111, 113], [111, 113], [116, 118], [116, 118]]],
                                       [[[11, 13], [11, 13], [11, 13], [16, 18], [16, 18]],
                                        [[111, 113], [111, 113], [111, 113], [116, 118], [116, 118]],
                                        [[1, 3], [1, 3], [1, 3], [6, 8], [6, 8]]],
                                       [[[111, 113], [111, 113], [111, 113], [116, 118], [116, 118]],
                                        [[1, 3], [1, 3], [1, 3], [6, 8], [6, 8]],
                                        [[11, 13], [11, 13], [11, 13], [16, 18], [16, 18]]]]).astype(datatype))

    np.testing.assert_array_equal(output.asnumpy(), expected_output.asnumpy())

def resize_nn_rgb_align_corners(datatype):
    input_tensor = Tensor(np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]],
                                     [[11, 12, 13, 14], [15, 16, 17, 18]],
                                     [[21, 22, 23, 24], [25, 26, 27, 28]]]]).astype(datatype))

    resize_nn_corners_aligned = NetResizeNearestNeighbor(
        (5, 2), align_corners=True)
    output_corners_aligned = resize_nn_corners_aligned(input_tensor)

    resize_nn = NetResizeNearestNeighbor((5, 2))
    output = resize_nn(input_tensor)

    expected_output = Tensor(np.array([[[[1, 4], [1, 4], [5, 8], [5, 8], [5, 8]],
                                        [[11, 14], [11, 14], [15, 18],
                                         [15, 18], [15, 18]],
                                        [[21, 24], [21, 24], [25, 28], [25, 28], [25, 28]]]]).astype(datatype))

    np.testing.assert_array_equal(
        output_corners_aligned.asnumpy(), expected_output.asnumpy())
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             output.asnumpy(), expected_output.asnumpy())

def resize_nn_grayscale_multiple_images(datatype):
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
                                    [[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]],
                                    [[[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]]).astype(datatype))

    resize_nn = NetResizeNearestNeighbor((2, 6))
    output = resize_nn(input_tensor)

    expected_output = Tensor(np.array([[[[0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
                                         [0.4, 0.4, 0.5, 0.5, 0.6, 0.6]]],
                                       [[[0.4, 0.4, 0.5, 0.5, 0.6, 0.6],
                                         [0.7, 0.7, 0.8, 0.8, 0.9, 0.9]]],
                                       [[[0.7, 0.7, 0.8, 0.8, 0.9, 0.9],
                                         [0.1, 0.1, 0.2, 0.2, 0.3, 0.3]]]]).astype(datatype))

    np.testing.assert_array_equal(output.asnumpy(), expected_output.asnumpy())

def resize_nn_grayscale_align_corners(datatype):
    input_tensor = Tensor(
        np.array([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]]).astype(datatype))

    resize_nn_corners_aligned = NetResizeNearestNeighbor(
        (3, 7), align_corners=True)
    output_corners_aligned = resize_nn_corners_aligned(input_tensor)

    resize_nn = NetResizeNearestNeighbor((3, 7))
    output = resize_nn(input_tensor)

    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4],
                                         [0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8],
                                         [0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8]]]]).astype(datatype))

    np.testing.assert_array_equal(
        output_corners_aligned.asnumpy(), expected_output.asnumpy())
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal,
                             output.asnumpy(), expected_output.asnumpy())

def test_resize_nn_grayscale_integer_ratio_half():
    resize_nn_grayscale_integer_ratio(np.float16)

def test_resize_nn_grayscale_integer_ratio_float():
    resize_nn_grayscale_integer_ratio(np.float32)

def test_resize_nn_grayscale_integer_ratio_double():
    resize_nn_grayscale_integer_ratio(np.float64)

def test_resize_nn_grayscale_not_integer_ratio_half():
    resize_nn_grayscale_not_integer_ratio(np.float16)

def test_resize_nn_grayscale_not_integer_ratio_float():
    resize_nn_grayscale_not_integer_ratio(np.float32)

def test_resize_nn_grayscale_not_integer_ratio_double():
    resize_nn_grayscale_not_integer_ratio(np.float64)

def test_resize_nn_grayscale_multiple_half():
    resize_nn_grayscale_multiple_images(np.float16)

def test_resize_nn_grayscale_multiple_float():
    resize_nn_grayscale_multiple_images(np.float32)

def test_resize_nn_grayscale_multiple_double():
    resize_nn_grayscale_multiple_images(np.float64)

def test_resize_nn_grayscale_align_corners_half():
    resize_nn_grayscale_align_corners(np.float16)

def test_resize_nn_grayscale_align_corners_float():
    resize_nn_grayscale_align_corners(np.float32)

def test_resize_nn_grayscale_align_corners_double():
    resize_nn_grayscale_align_corners(np.float64)

def test_resize_nn_rgb_integer_ratio_int32():
    resize_nn_rgb_integer_ratio(np.int32)

def test_resize_nn_rgb_integer_ratio_int64():
    resize_nn_rgb_integer_ratio(np.int64)

def test_resize_nn_rgb_not_integer_ratio_int32():
    resize_nn_rgb_not_integer_ratio(np.int32)

def test_resize_nn_rgb_not_integer_ratio_int64():
    resize_nn_rgb_not_integer_ratio(np.int64)

def test_resize_nn_rgb_multiple_int32():
    resize_nn_rgb_multiple(np.int32)

def test_resize_nn_rgb_multiple_int64():
    resize_nn_rgb_multiple(np.int64)

def test_resize_nn_rgb_align_corners_int32():
    resize_nn_rgb_align_corners(np.int32)

def test_resize_nn_rgb_align_corners_int64():
    resize_nn_rgb_align_corners(np.int64)

if __name__ == "__main__":
    test_resize_nn_grayscale_integer_ratio_half()
    test_resize_nn_grayscale_integer_ratio_float()
    test_resize_nn_grayscale_integer_ratio_double()
    test_resize_nn_grayscale_not_integer_ratio_half()
    test_resize_nn_grayscale_not_integer_ratio_float()
    test_resize_nn_grayscale_not_integer_ratio_double()
    test_resize_nn_grayscale_multiple_half()
    test_resize_nn_grayscale_multiple_float()
    test_resize_nn_grayscale_multiple_double()
    test_resize_nn_grayscale_align_corners_half()
    test_resize_nn_grayscale_align_corners_float()
    test_resize_nn_grayscale_align_corners_double()
    test_resize_nn_rgb_integer_ratio_int32()
    test_resize_nn_rgb_integer_ratio_int64()
    test_resize_nn_rgb_not_integer_ratio_int32()
    test_resize_nn_rgb_not_integer_ratio_int64()
    test_resize_nn_rgb_multiple_int32()
    test_resize_nn_rgb_multiple_int64()
    test_resize_nn_rgb_align_corners_int32()
    test_resize_nn_rgb_align_corners_int64()
