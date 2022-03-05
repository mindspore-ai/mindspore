# Copyright 2020-2021 Huawei Technologies Co., Ltd
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


class NetResizeBilinear(nn.Cell):
    def __init__(self, size=None, align_corner=False, half_pixel_centers=False):
        super(NetResizeBilinear, self).__init__()
        self.op = P.ResizeBilinear(size=size, align_corners=align_corner, half_pixel_centers=half_pixel_centers)

    def construct(self, inputs):
        return self.op(inputs)


def test_resize_nn_grayscale_integer_ratio_half(datatype=np.float16):
    input_tensor = Tensor(np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeBilinear((9, 9))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1333, 0.1666, 0.2, 0.2333, 0.2666, 0.3, 0.3, 0.3],
                                         [0.2, 0.2333, 0.2666, 0.2998, 0.3333, 0.3667, 0.4, 0.4, 0.4],
                                         [0.2998, 0.3333, 0.3665, 0.4, 0.433, 0.4668, 0.5, 0.5, 0.5],
                                         [0.4, 0.4333, 0.4666, 0.5, 0.533, 0.567, 0.6, 0.6, 0.6],
                                         [0.5, 0.533, 0.5664, 0.6, 0.6333, 0.667, 0.7, 0.7, 0.7],
                                         [0.6, 0.6333, 0.6665, 0.6997, 0.733, 0.7666, 0.8, 0.8, 0.8],
                                         [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.8667, 0.9, 0.9, 0.9],
                                         [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.8667, 0.9, 0.9, 0.9],
                                         [0.7, 0.7334, 0.7666, 0.8, 0.833, 0.8667, 0.9, 0.9, 0.9]]]]
                                      ).astype(np.float16))
    error = np.ones(shape=[9, 9]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h and w
    resize_nn = NetResizeBilinear((1, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1]]]]).astype(np.float16))
    error = np.ones(shape=[1, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, larger w
    resize_nn = NetResizeBilinear((1, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.1499, 0.2, 0.25, 0.3, 0.3]]]]).astype(np.float16))
    error = np.ones(shape=[1, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, smaller w
    resize_nn = NetResizeBilinear((6, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1],
                    [0.2499],
                    [0.4],
                    [0.55],
                    [0.7],
                    [0.7]]]]).astype(np.float16))
    error = np.ones(shape=[6, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, same w
    resize_nn = NetResizeBilinear((1, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.2, 0.3]]]]).astype(np.float16))
    error = np.ones(shape=[1, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, same w
    resize_nn = NetResizeBilinear((6, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3],
                                         [0.2499, 0.35, 0.4502],
                                         [0.4, 0.5, 0.6],
                                         [0.55, 0.65, 0.75],
                                         [0.7, 0.8, 0.9],
                                         [0.7, 0.8, 0.9]]]]).astype(np.float16))
    error = np.ones(shape=[6, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, smaller w
    resize_nn = NetResizeBilinear((3, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1],
                    [0.4],
                    [0.7]]]]).astype(np.float16))
    error = np.ones(shape=[3, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, larger w
    resize_nn = NetResizeBilinear((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1499, 0.2, 0.25, 0.3, 0.3],
                                         [0.4, 0.45, 0.5, 0.55, 0.6, 0.6],
                                         [0.7, 0.75, 0.8, 0.8496, 0.9, 0.9]]]]).astype(np.float16))
    error = np.ones(shape=[3, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same w, same h (identity)
    resize_nn = NetResizeBilinear((3, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array(
        [[[[0.1, 0.2, 0.3],
           [0.4, 0.5, 0.6],
           [0.7, 0.8, 0.9]]]]).astype(np.float16))
    error = np.ones(shape=[3, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)


def test_resize_nn_grayscale_integer_ratio_float(datatype=np.float32):
    input_tensor = Tensor(np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeBilinear((9, 9))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.13333334, 0.16666667, 0.2, 0.23333335, 0.26666668, 0.3, 0.3, 0.3],
                                         [0.20000002, 0.23333335, 0.26666668, 0.3, 0.33333337, 0.3666667, 0.40000004,
                                          0.40000004, 0.40000004],
                                         [0.3, 0.33333337, 0.36666667, 0.40000004, 0.43333337, 0.4666667, 0.5, 0.5,
                                          0.5],
                                         [0.4, 0.43333334, 0.46666667, 0.5, 0.53333336, 0.5666667, 0.6, 0.6, 0.6],
                                         [0.5, 0.53333336, 0.56666666, 0.6, 0.6333333, 0.66666675, 0.70000005,
                                          0.70000005, 0.70000005],
                                         [0.6, 0.6333334, 0.6666667, 0.70000005, 0.73333335, 0.7666667, 0.8, 0.8, 0.8],
                                         [0.7, 0.73333335, 0.76666665, 0.8, 0.8333333, 0.8666667, 0.9, 0.9, 0.9],
                                         [0.7, 0.73333335, 0.76666665, 0.8, 0.8333333, 0.8666667, 0.9, 0.9, 0.9],
                                         [0.7, 0.73333335, 0.76666665, 0.8, 0.8333333, 0.8666667, 0.9, 0.9,
                                          0.9]]]]).astype(np.float32))
    error = np.ones(shape=[9, 9]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h and w
    resize_nn = NetResizeBilinear((1, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1]]]]).astype(np.float32))
    error = np.ones(shape=[1, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, larger w
    resize_nn = NetResizeBilinear((1, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.15, 0.2, 0.25, 0.3, 0.3]]]]).astype(np.float32))
    error = np.ones(shape=[1, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, smaller w
    resize_nn = NetResizeBilinear((6, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1], [0.25], [0.4], [0.55], [0.7], [0.7]]]]).astype(np.float32))
    error = np.ones(shape=[6, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, same w
    resize_nn = NetResizeBilinear((1, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.2, 0.3]]]]).astype(np.float32))
    error = np.ones(shape=[1, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, same w
    resize_nn = NetResizeBilinear((6, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3],
                                         [0.25, 0.35000002, 0.45000002],
                                         [0.4, 0.5, 0.6],
                                         [0.55, 0.65, 0.75],
                                         [0.7, 0.8, 0.9],
                                         [0.7, 0.8, 0.9]]]]).astype(np.float32))
    error = np.ones(shape=[6, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, smaller w
    resize_nn = NetResizeBilinear((3, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1], [0.4], [0.7]]]]).astype(np.float32))
    error = np.ones(shape=[3, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, larger w
    resize_nn = NetResizeBilinear((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.15, 0.2, 0.25, 0.3, 0.3],
                                         [0.4, 0.45, 0.5, 0.55, 0.6, 0.6],
                                         [0.7, 0.75, 0.8, 0.85, 0.9, 0.9]]]]).astype(np.float32))
    error = np.ones(shape=[3, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same w, same h (identity)
    resize_nn = NetResizeBilinear((3, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(np.float32))
    error = np.ones(shape=[3, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)


def test_resize_nn_grayscale_not_integer_ratio_half(datatype=np.float16):
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 0.0, 0.1, 0.2]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeBilinear((7, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1571, 0.2142, 0.2715, 0.3286, 0.3857, 0.4],
                                         [0.2715, 0.3286, 0.3857, 0.4429, 0.5, 0.557, 0.5713],
                                         [0.4429, 0.5, 0.557, 0.6143, 0.6714, 0.7285, 0.7427],
                                         [0.6143, 0.5083, 0.4429, 0.5005, 0.557, 0.6143, 0.6284],
                                         [0.7856, 0.4346, 0.1855, 0.2429, 0.2998, 0.357, 0.3716],
                                         [0.9, 0.3857, 0.01428, 0.0714, 0.1285, 0.1857, 0.2],
                                         [0.9, 0.3857, 0.01428, 0.0714, 0.1285, 0.1857, 0.2]]]]).astype(np.float16))
    error = np.ones(shape=[7, 7]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h and w
    resize_nn = NetResizeBilinear((2, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.2333, 0.3667],
                    [0.7, 0.3333, 0.4666]]]]).astype(np.float16))
    error = np.ones(shape=[2, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, larger w
    resize_nn = NetResizeBilinear((2, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1571, 0.2142, 0.2715, 0.3286, 0.3857, 0.4],
                                         [0.7, 0.4714, 0.3142, 0.3716, 0.4285, 0.4856, 0.5]]]]).astype(np.float16))
    error = np.ones(shape=[2, 7]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, smaller w
    resize_nn = NetResizeBilinear((5, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2333, 0.3667],
                                         [0.3398, 0.4731, 0.6064],
                                         [0.58, 0.513, 0.6465],
                                         [0.82, 0.1533, 0.2866],
                                         [0.9, 0.03333, 0.1666]]]]).astype(np.float16))
    error = np.ones(shape=[5, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, same w
    resize_nn = NetResizeBilinear((2, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.7, 0.3, 0.4001, 0.5]]]]).astype(np.float16))
    error = np.ones(shape=[2, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, same w
    resize_nn = NetResizeBilinear((8, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.2499, 0.35, 0.4502, 0.55],
                                         [0.4, 0.5, 0.6, 0.6997],
                                         [0.55, 0.525, 0.625, 0.7246],
                                         [0.7, 0.3, 0.4001, 0.5],
                                         [0.8496, 0.0752, 0.1753, 0.2754],
                                         [0.9, 0., 0.1, 0.2],
                                         [0.9, 0., 0.1, 0.2]]]]).astype(np.float16))
    error = np.ones(shape=[8, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, smaller w
    resize_nn = NetResizeBilinear((3, 2))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.3],
                                         [0.5, 0.7],
                                         [0.9, 0.1]]]]).astype(np.float16))
    error = np.ones(shape=[3, 2]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, larger w
    resize_nn = NetResizeBilinear((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1666, 0.2333, 0.3, 0.3667, 0.4],
                                         [0.5, 0.567, 0.6333, 0.7, 0.7666, 0.8],
                                         [0.9, 0.3003, 0.03333, 0.1, 0.1666, 0.2]]]]).astype(np.float16))
    error = np.ones(shape=[3, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same w, same h (identity)
    resize_nn = NetResizeBilinear((3, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.5, 0.6, 0.7, 0.8],
                                         [0.9, 0., 0.1, 0.2]]]]).astype(np.float16))
    error = np.ones(shape=[3, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)


def test_resize_nn_grayscale_not_integer_ratio_float(datatype=np.float32):
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                      [0.5, 0.6, 0.7, 0.8],
                                      [0.9, 0.0, 0.1, 0.2]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeBilinear((7, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.15714286, 0.21428573, 0.27142859, 0.32857144, 0.3857143, 0.4],
                                         [0.27142859, 0.32857144, 0.38571432, 0.44285715, 0.5, 0.55714285, 0.5714286],
                                         [0.44285715, 0.5, 0.5571429, 0.6142857, 0.67142856, 0.7285714, 0.74285716],
                                         [0.6142857, 0.5081633, 0.4428572, 0.5, 0.55714285, 0.6142857, 0.62857145],
                                         [0.78571427, 0.43469384, 0.1857143, 0.24285716, 0.3, 0.35714287, 0.37142855],
                                         [0.9, 0.38571423, 0.01428572, 0.07142859, 0.12857144, 0.1857143, 0.2],
                                         [0.9, 0.38571423, 0.01428572, 0.07142859, 0.12857144, 0.1857143,
                                          0.2]]]]).astype(np.float32))
    error = np.ones(shape=[7, 7]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h and w
    resize_nn = NetResizeBilinear((2, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.1, 0.23333335, 0.36666667],
                    [0.7, 0.33333334, 0.46666667]]]]).astype(np.float32))
    error = np.ones(shape=[2, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, larger w
    resize_nn = NetResizeBilinear((2, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.15714286, 0.21428573, 0.27142859, 0.32857144,
                                          0.3857143, 0.4],
                                         [0.7, 0.47142854, 0.31428576, 0.37142858, 0.42857143,
                                          0.4857143, 0.5]]]]).astype(np.float32))
    error = np.ones(shape=[2, 7]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, smaller w
    resize_nn = NetResizeBilinear((5, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.23333335, 0.36666667],
                                         [0.34, 0.47333336, 0.6066667],
                                         [0.58000004, 0.5133333, 0.64666665],
                                         [0.82000005, 0.1533333, 0.28666663],
                                         [0.9, 0.03333334, 0.16666669]]]]).astype(np.float32))
    error = np.ones(shape=[5, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, same w
    resize_nn = NetResizeBilinear((2, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.7, 0.3, 0.4, 0.5]]]]).astype(np.float32))
    error = np.ones(shape=[2, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, same w
    resize_nn = NetResizeBilinear((8, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.25, 0.35000002, 0.45, 0.55],
                                         [0.4, 0.5, 0.6, 0.70000005],
                                         [0.55, 0.52500004, 0.625, 0.725],
                                         [0.7, 0.3, 0.4, 0.5],
                                         [0.84999996, 0.07499999,
                                          0.17500001, 0.27499998],
                                         [0.9, 0., 0.1, 0.2],
                                         [0.9, 0., 0.1, 0.2]]]]).astype(np.float32))
    error = np.ones(shape=[8, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, smaller w
    resize_nn = NetResizeBilinear((3, 2))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.3],
                                         [0.5, 0.7],
                                         [0.9, 0.1]]]]).astype(np.float32))
    error = np.ones(shape=[3, 2]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, larger w
    resize_nn = NetResizeBilinear((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.16666667, 0.23333335, 0.3, 0.36666667, 0.4],
                                         [0.5, 0.56666666, 0.6333333, 0.7, 0.76666665, 0.8],
                                         [0.9, 0.29999995, 0.03333334, 0.1, 0.16666669, 0.2]]]]).astype(np.float32))
    error = np.ones(shape=[3, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same w, same h (identity)
    resize_nn = NetResizeBilinear((3, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.2, 0.3, 0.4],
                                         [0.5, 0.6, 0.7, 0.8],
                                         [0.9, 0., 0.1, 0.2]]]]).astype(np.float32))
    error = np.ones(shape=[3, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)


def test_resize_nn_grayscale_multiple_images_half(datatype=np.float16):
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
                                    [[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]],
                                    [[[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]]).astype(datatype))

    resize_nn = NetResizeBilinear((2, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.1, 0.1499, 0.2, 0.25, 0.3, 0.3],
                                         [0.55, 0.6, 0.65, 0.6997, 0.75, 0.75]]],
                                       [[[0.4, 0.45, 0.5, 0.55, 0.6, 0.6],
                                         [0.4001, 0.45, 0.5, 0.55, 0.6, 0.6]]],
                                       [[[0.7, 0.75, 0.8, 0.8496, 0.9, 0.9],
                                         [0.2499, 0.2998, 0.35, 0.4, 0.4502, 0.4502]]]]).astype(np.float16))

    error = np.ones(shape=[3, 3, 2, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)


def test_resize_nn_grayscale_multiple_images_float(datatype=np.float32):
    input_tensor = Tensor(np.array([[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
                                    [[[0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]],
                                    [[[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]]).astype(datatype))

    resize_nn = NetResizeBilinear((2, 6))
    output = resize_nn(input_tensor)

    expected_output = Tensor(np.array([[[[0.1, 0.15, 0.2, 0.25, 0.3, 0.3],
                                         [0.55, 0.6, 0.65, 0.70000005, 0.75, 0.75]]],
                                       [[[0.4, 0.45, 0.5, 0.55, 0.6, 0.6],
                                         [0.4, 0.45, 0.5, 0.55, 0.6, 0.6]]],
                                       [[[0.7, 0.75, 0.8, 0.85, 0.9, 0.9],
                                         [0.25, 0.3, 0.35000002, 0.4, 0.45000002, 0.45000002]]]]).astype(np.float32))

    error = np.ones(shape=[3, 3, 2, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)


def test_resize_nn_grayscale_align_corners_half(datatype=np.float16):
    input_tensor = Tensor(
        np.array([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]]).astype(datatype))

    resize_nn_corners_aligned = NetResizeBilinear(
        size=(3, 7), align_corner=True)
    output_corners_aligned = resize_nn_corners_aligned(input_tensor)

    resize_nn = NetResizeBilinear((3, 7))
    output = resize_nn(input_tensor)

    expected_output_align = Tensor(np.array([[[[0.1, 0.1499, 0.2, 0.25, 0.3, 0.35, 0.4],
                                               [0.2998, 0.3499, 0.4, 0.4502, 0.5, 0.55, 0.5996],
                                               [0.5, 0.55, 0.6, 0.6504, 0.7, 0.75, 0.8]]]]).astype(np.float16))
    expected_output = Tensor(np.array([[[[0.1, 0.1571, 0.2142, 0.2715, 0.3286, 0.3857, 0.4],
                                         [0.3667, 0.4238, 0.481, 0.538, 0.595, 0.6523, 0.6665],
                                         [0.5, 0.557, 0.6143, 0.672, 0.7285, 0.7856, 0.8]]]]).astype(np.float16))

    error = np.ones(shape=[3, 7]) * 1.0e-6
    diff_align = output_corners_aligned.asnumpy() - expected_output_align.asnumpy()
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)
    assert np.all(abs(diff_align) < error)


def test_resize_nn_grayscale_align_corners_float(datatype=np.float32):
    input_tensor = Tensor(
        np.array([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]]]).astype(datatype))

    resize_nn_corners_aligned = NetResizeBilinear(
        size=(3, 7), align_corner=True)
    output_corners_aligned = resize_nn_corners_aligned(input_tensor)

    resize_nn = NetResizeBilinear((3, 7))
    output = resize_nn(input_tensor)

    expected_output_align = Tensor(np.array([[[[0.1, 0.15, 0.2, 0.25, 0.3,
                                                0.35000002, 0.4],
                                               [0.3, 0.35000002, 0.40000004, 0.45, 0.5,
                                                0.55, 0.6],
                                               [0.5, 0.55, 0.6, 0.65, 0.7,
                                                0.75, 0.8]]]]).astype(datatype))
    expected_output = Tensor(np.array([[[[0.1, 0.15714286, 0.21428573, 0.27142859, 0.32857144,
                                          0.3857143, 0.4],
                                         [0.36666667, 0.42380953, 0.48095244, 0.53809524, 0.5952381,
                                          0.65238094, 0.6666667],
                                         [0.5, 0.55714285, 0.61428577, 0.67142856, 0.7285714,
                                          0.78571427, 0.8]]]]).astype(datatype))

    error = np.ones(shape=[3, 7]) * 1.0e-6
    diff_align = output_corners_aligned.asnumpy() - expected_output_align.asnumpy()
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)
    assert np.all(abs(diff_align) < error)
