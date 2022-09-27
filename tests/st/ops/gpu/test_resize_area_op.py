import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops.operations.image_ops as ops


class NetResizeArea(nn.Cell):

    def __init__(self, align_corners=False):
        super(NetResizeArea, self).__init__()
        self.resize_area = ops.ResizeArea(align_corners=align_corners)

    def construct(self, images, size):
        return self.resize_area(images, size)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_area_float16():
    """
    Feature: Input type of float16
    Description: Input type of [float16, int32].
    Expectation: success.
    """
    for mode in [context.PYNATIVE_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        data_type = np.half
        images_array = np.array([[[[1.2, 120], [140, 2.1], [40, 0.12]],
                                  [[1.35, 1.2], [0.04, 5], [10, 4]]],
                                 [[[34, 10.2], [1.05, 12.1], [3, 0.06]],
                                  [[65, 23], [14, 4.3], [2.2, 4]]]]).astype(data_type)
        size_array = np.array([2, 2]).astype(np.int32)
        align_corners = True

        images_ms = Tensor(images_array)
        size_ms = Tensor(size_array)
        resize_area = NetResizeArea(align_corners=align_corners)
        output_ms = resize_area(images_ms, size_ms)
        expect = np.array([[[[7.0600098e+01, 6.1049805e+01], [4.0000000e+01, 1.1999512e-01]],
                            [[6.9480896e-01, 3.1000977e+00], [1.0000000e+01, 4.0000000e+00]]],
                           [[[1.7524902e+01, 1.1152344e+01], [3.0000000e+00, 5.9997559e-02]],
                            [[3.9500000e+01, 1.3650391e+01], [2.1992188e+00, 4.0000000e+00]]]]).astype(np.float32)
        assert np.allclose(output_ms.asnumpy(),
                           expect,
                           rtol=1e-4,
                           atol=1e-4,
                           equal_nan=False)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resize_area_float32():
    """
    Feature: Input type of float32
    Description: Input type of [float32, int32].
    Expectation: success.
    """
    for mode in [context.PYNATIVE_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        data_type = np.float32
        images_array = np.array([[[[1.2, 0.1, 120], [0.140, 2.5, 21]], [[135, 0.02, 2], [0.00102, 4.1, 3]]],
                                 [[[4, 0.01, 1.02], [21, 0.15, 11]], [[65, 2.1, 23], [22, 1.2, 4]]]]).astype(data_type)
        size_array = np.array([3, 3]).astype(np.int32)
        align_corners = False

        images_ms = Tensor(images_array)
        size_ms = Tensor(size_array)
        resize_area = NetResizeArea(align_corners=align_corners)
        output_ms = resize_area(images_ms, size_ms)
        expect = np.array([[[[1.2000000e+00, 1.0000000e-01, 1.1999999e+02],
                             [6.6999996e-01, 1.3000001e+00, 7.0499992e+01],
                             [1.3999999e-01, 2.4999995e+00, 2.0999996e+01]],
                            [[6.8099998e+01, 5.9999995e-02, 6.0999989e+01],
                             [3.4085251e+01, 1.6800001e+00, 3.6499989e+01],
                             [7.0509978e-02, 3.2999995e+00, 1.1999997e+01]],
                            [[1.3499998e+02, 1.9999996e-02, 1.9999996e+00],
                             [6.7500488e+01, 2.0599999e+00, 2.4999995e+00],
                             [1.0199997e-03, 4.0999990e+00, 2.9999993e+00]]],
                           [[[3.9999998e+00, 9.9999988e-03, 1.0200000e+00],
                             [1.2500000e+01, 8.0000013e-02, 6.0100002e+00],
                             [2.0999996e+01, 1.4999999e-01, 1.0999999e+01]],
                            [[3.4500004e+01, 1.0549999e+00, 1.2010000e+01],
                             [2.8000000e+01, 8.6499995e-01, 9.7550001e+00],
                             [2.1499998e+01, 6.7500001e-01, 7.4999986e+00]],
                            [[6.4999992e+01, 2.0999997e+00, 2.2999998e+01],
                             [4.3499992e+01, 1.6499997e+00, 1.3499997e+01],
                             [2.1999996e+01, 1.1999998e+00, 3.9999990e+00]]]]).astype(np.float32)
        assert np.allclose(output_ms.asnumpy(),
                           expect,
                           rtol=1e-4,
                           atol=1e-4,
                           equal_nan=False)
