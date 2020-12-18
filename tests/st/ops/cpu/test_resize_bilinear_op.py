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


class NetResizeBilinear(nn.Cell):
    def __init__(self, size=None, align_corner=False):
        super(NetResizeBilinear, self).__init__()
        self.op = P.ResizeBilinear(size=size, align_corners=align_corner)

    def construct(self, inputs):
        return self.op(inputs)


def test_resize_nn_grayscale_integer_ratio_half(datatype=np.float16):
    input_tensor = Tensor(np.array(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]]).astype(datatype))

    # larger h and w
    resize_nn = NetResizeBilinear((9, 9))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.13330078, 0.16662598, 0.19995117, 0.23331706,
                                          0.26668295, 0.30004883, 0.30004883, 0.30004883],
                                         [0.19995117, 0.23328993, 0.26662868, 0.29996747, 0.33333334,
                                          0.36669925, 0.40006512, 0.40006512, 0.40006512],
                                         [0.29992676, 0.33327907, 0.36663142, 0.39998373, 0.4333496,
                                          0.4667155, 0.5000814, 0.5000814, 0.5000814],
                                         [0.39990234, 0.43326822, 0.46663412, 0.5, 0.5333659,
                                          0.5667318, 0.60009766, 0.60009766, 0.60009766],
                                         [0.5, 0.5333116, 0.5666233, 0.59993494, 0.6333008,
                                          0.66666675, 0.7000326, 0.7000326, 0.7000326],
                                         [0.60009766, 0.633355, 0.66661245, 0.6998698, 0.7332357,
                                          0.7666016, 0.79996747, 0.79996747, 0.79996747],
                                         [0.7001953, 0.73339844, 0.76660156, 0.7998047, 0.8331706,
                                          0.8665365, 0.89990234, 0.89990234, 0.89990234],
                                         [0.7001953, 0.73339844, 0.76660156, 0.7998047, 0.8331706,
                                          0.8665365, 0.89990234, 0.89990234, 0.89990234],
                                         [0.7001953, 0.73339844, 0.76660156, 0.7998047, 0.8331706,
                                          0.8665365, 0.89990234, 0.89990234, 0.89990234]]]]).astype(np.float32))
    error = np.ones(shape=[9, 9]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h and w
    resize_nn = NetResizeBilinear((1, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559]]]]).astype(np.float32))
    error = np.ones(shape=[1, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, larger w
    resize_nn = NetResizeBilinear((1, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.09997559, 0.14996338, 0.19995117, 0.25, 0.30004883, 0.30004883]]]]).astype(np.float32))
    error = np.ones(shape=[1, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, smaller w
    resize_nn = NetResizeBilinear((6, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.09997559], [0.24993896], [0.39990234], [0.5500488], [0.7001953], [0.7001953]]]]).astype(
            np.float32))
    error = np.ones(shape=[6, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, same w
    resize_nn = NetResizeBilinear((1, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.09997559, 0.19995117, 0.30004883]]]]).astype(np.float32))
    error = np.ones(shape=[1, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, same w
    resize_nn = NetResizeBilinear((6, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.19995117, 0.30004883],
                                         [0.24993896, 0.3499756, 0.45007324],
                                         [0.39990234, 0.5, 0.60009766],
                                         [0.5500488, 0.64990234, 0.75],
                                         [0.7001953, 0.7998047, 0.89990234],
                                         [0.7001953, 0.7998047, 0.89990234]]]]).astype(np.float32))
    error = np.ones(shape=[6, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, smaller w
    resize_nn = NetResizeBilinear((3, 1))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.09997559], [0.39990234], [0.7001953]]]]).astype(np.float32))
    error = np.ones(shape=[3, 1]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, larger w
    resize_nn = NetResizeBilinear((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.14996338, 0.19995117, 0.25, 0.30004883,
                                          0.30004883],
                                         [0.39990234, 0.44995117, 0.5, 0.5500488, 0.60009766,
                                          0.60009766],
                                         [0.7001953, 0.75, 0.7998047, 0.8498535, 0.89990234,
                                          0.89990234]]]]).astype(np.float32))
    error = np.ones(shape=[3, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same w, same h (identity)
    resize_nn = NetResizeBilinear((3, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array(
        [[[[0.09997559, 0.19995117, 0.30004883],
           [0.39990234, 0.5, 0.60009766],
           [0.7001953, 0.7998047, 0.89990234]]]]).astype(np.float32))
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
    expected_output = Tensor(np.array([[[[0.09997559, 0.15710449, 0.21425085, 0.2714495, 0.3285784,
                                          0.38563755, 0.39990234],
                                         [0.27141464, 0.3285734, 0.3857422, 0.44294086, 0.5000399,
                                          0.55703926, 0.57128906],
                                         [0.44285366, 0.5000423, 0.5572336, 0.6144322, 0.67150134,
                                          0.7284409, 0.7426758],
                                         [0.6142578, 0.50819117, 0.44293588, 0.5001146, 0.5571937,
                                          0.6141731, 0.62841797],
                                         [0.78564453, 0.4346799, 0.18574369, 0.2428925, 0.3000015,
                                          0.3570706, 0.3713379],
                                         [0.89990234, 0.3856724, 0.01428223, 0.07141115, 0.12854005,
                                          0.18566895, 0.19995117],
                                         [0.89990234, 0.3856724, 0.01428223, 0.07141115, 0.12854005,
                                          0.18566895, 0.19995117]]]]).astype(np.float32))
    error = np.ones(shape=[7, 7]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h and w
    resize_nn = NetResizeBilinear((2, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(
        np.array([[[[0.09997559, 0.23331706, 0.36661786],
                    [0.6999512, 0.33339438, 0.46661377]]]]).astype(np.float32))
    error = np.ones(shape=[2, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, larger w
    resize_nn = NetResizeBilinear((2, 7))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.15710449, 0.21425085, 0.2714495, 0.3285784,
                                          0.38563755, 0.39990234],
                                         [0.6999512, 0.47143552, 0.3143398, 0.37150356, 0.4285976,
                                          0.48562187, 0.49987793]]]]).astype(np.float32))
    error = np.ones(shape=[2, 7]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, smaller w
    resize_nn = NetResizeBilinear((5, 3))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.23331706, 0.36661786],
                                         [0.33999026, 0.47340494, 0.6066081],
                                         [0.5799805, 0.51343584, 0.64660645],
                                         [0.8199219, 0.15335283, 0.28662106],
                                         [0.89990234, 0.0333252, 0.16662598]]]]).astype(np.float32))
    error = np.ones(shape=[5, 3]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # smaller h, same w
    resize_nn = NetResizeBilinear((2, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.19995117, 0.30004883, 0.39990234],
                                         [0.6999512, 0.30004883, 0.40008545, 0.49987793]]]]).astype(np.float32))
    error = np.ones(shape=[2, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # larger h, same w
    resize_nn = NetResizeBilinear((8, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.19995117, 0.30004883, 0.39990234],
                                         [0.24998474, 0.3500061, 0.45010376, 0.5498657],
                                         [0.3999939, 0.50006104, 0.6001587, 0.6998291],
                                         [0.5499878, 0.52508545, 0.62516785, 0.724823],
                                         [0.6999512, 0.30004883, 0.40008545, 0.49987793],
                                         [0.84991455, 0.07501221, 0.17500305, 0.27493286],
                                         [0.89990234, 0., 0.09997559, 0.19995117],
                                         [0.89990234, 0., 0.09997559, 0.19995117]]]]).astype(np.float32))
    error = np.ones(shape=[8, 4]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, smaller w
    resize_nn = NetResizeBilinear((3, 2))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.30004883],
                                         [0.5, 0.7001953],
                                         [0.89990234, 0.09997559]]]]).astype(np.float32))
    error = np.ones(shape=[3, 2]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same h, larger w
    resize_nn = NetResizeBilinear((3, 6))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.16662598, 0.23331706, 0.30004883, 0.36661786,
                                          0.39990234],
                                         [0.5, 0.56673175, 0.63346356, 0.7001953, 0.76660156,
                                          0.7998047],
                                         [0.89990234, 0.2999674, 0.0333252, 0.09997559, 0.16662598,
                                          0.19995117]]]]).astype(np.float32))
    error = np.ones(shape=[3, 6]) * 1.0e-6
    diff = output.asnumpy() - expected_output.asnumpy()
    assert np.all(abs(diff) < error)

    # same w, same h (identity)
    resize_nn = NetResizeBilinear((3, 4))
    output = resize_nn(input_tensor)
    expected_output = Tensor(np.array([[[[0.09997559, 0.19995117, 0.30004883, 0.39990234],
                                         [0.5, 0.60009766, 0.7001953, 0.7998047],
                                         [0.89990234, 0., 0.09997559, 0.19995117]]]]).astype(np.float32))
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
    expected_output = Tensor(np.array([[[[0.09997559, 0.14996338, 0.19995117, 0.25, 0.30004883, 0.30004883],
                                         [0.5500488, 0.5999756, 0.64990234, 0.6999512, 0.75, 0.75]]],
                                       [[[0.39990234, 0.44995117, 0.5, 0.5500488, 0.60009766, 0.60009766],
                                         [0.40008545, 0.4499817, 0.49987793, 0.54992676, 0.5999756, 0.5999756]]],
                                       [[[0.7001953, 0.75, 0.7998047, 0.8498535, 0.89990234, 0.89990234],
                                         [0.24993896, 0.29995728, 0.3499756, 0.4000244, 0.45007324,
                                          0.45007324]]]]).astype(np.float32))

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

    expected_output_align = Tensor(np.array([[[[0.09997559, 0.14996338, 0.19995117, 0.25, 0.30004883,
                                                0.3499756, 0.39990234],
                                               [0.2999878, 0.3500061, 0.4000244, 0.45007324, 0.5001221,
                                                0.5499878, 0.5998535],
                                               [0.5, 0.5500488, 0.60009766, 0.6501465, 0.7001953,
                                                0.75, 0.7998047]]]]).astype(np.float32))
    expected_output = Tensor(np.array([[[[0.09997559, 0.15710449, 0.21425085, 0.2714495, 0.3285784,
                                          0.38563755, 0.39990234],
                                         [0.36665854, 0.42383394, 0.4810152, 0.53821385, 0.59529626,
                                          0.6522624, 0.6665039],
                                         [0.5, 0.55719864, 0.61439735, 0.671596, 0.72865516,
                                          0.7855748, 0.7998047]]]]).astype(np.float32))

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
