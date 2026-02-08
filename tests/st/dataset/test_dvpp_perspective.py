# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Testing DVPP Perspective operation
"""
import os
import numpy as np
import pytest
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as v_trans
from mindspore.dataset.vision import Inter as v_Inter
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=112.0)
def test_dvpp_perspective_operation_01():
    """
    Feature: Perspective operation on device
    Description: Testing the normal functionality of the Perspective operator on device
    Expectation: The Output is equal to the expected output
    """
    # Using Perspective operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    start_points = [[1, 1], [2, 2], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device("Ascend")(
            img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST)(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # pipeline mode, no interpolation parameter, Perspective interface call successful
    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds2 = ds.ImageFolderDataset(data_dir, 1)
    transforms1 = [
        v_trans.Decode(to_pil=False),
        v_trans.Perspective(start_points, end_points).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for _ in ds2.create_dict_iterator(output_numpy=True):
        pass

    # eager mode, when input is 3-dimensional uint8 data, Perspective interface call successful
    image = np.random.randint(0, 128, (128, 128, 3)).astype(np.uint8)
    start_points = [[1, 1], [2, 2], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    perspective_op = v_trans.Perspective(start_points, end_points).device(device_target="Ascend")
    perspective_op_cpu = v_trans.Perspective(start_points, end_points)(image)
    out = perspective_op(image)
    assert (out == perspective_op_cpu).all()

    # eager mode, when input is 2-dimensional uint8 data, Perspective interface call successful
    image = np.random.randint(0, 128, (128, 128)).astype(np.uint8)
    start_points = [[10, 1], [2, 20], [30, 3], [40, 4]]
    end_points = [[20, 2], [10, 1], [3, 30], [4, 40]]
    perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
    perspective_op_cpu = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST)(image)
    out = perspective_op(image)
    assert (out == perspective_op_cpu).all()

    # When input is 3-dimensional float32 data, Perspective interface call successful
    image = np.random.randn(255, 255, 3).astype(np.float32)
    start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
    perspective_op_cpu = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST)(image)
    out = perspective_op(image)
    assert (out == perspective_op_cpu).all()

    image = np.random.randn(1, 128, 128, 3).astype(np.uint8)
    start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
    out = perspective_op(image)
    assert out.shape == (128, 128, 3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=34.0)
def test_dvpp_perspective_operation_02():
    """
    Feature: Perspective operation on device
    Description: Testing the normal functionality of the Perspective operator on device
    Expectation: The Output is equal to the expected output
    """
    # When interpolation parameter is Inter.BILINEAR or Inter.LINEAR, Perspective interface call successful
    image = Image.open(image_gif)
    img_array = np.array(image)
    start_points = [(-218, -21474), (-2147, -21474), (21474, 214748), (218, 21474)]
    end_points = [(-218, -214), (-2147, -2147), (2178, 21783), (218, 214)]
    perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.BILINEAR).device(device_target="Ascend")
    perspective_op_cpu = v_trans.Perspective(start_points, end_points, v_Inter.BILINEAR)(img_array)
    perspective_op1 = v_trans.Perspective(start_points, end_points, v_Inter.LINEAR).device(device_target="Ascend")
    perspective_op1_cpu = v_trans.Perspective(start_points, end_points, v_Inter.LINEAR)(img_array)
    out = perspective_op(img_array)
    out1 = perspective_op1(img_array)
    assert (out == perspective_op_cpu).all()
    assert (out1 == perspective_op1_cpu).all()
    image.close()

    # pipeline mode, when input is numpy data, Perspective interface call successful
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds1 = ds.ImageFolderDataset(data_dir, 1)
    start_points = [[-28, -2474], [-214, -2144], [2147, 21448], [28, 2144]]
    end_points = [[-21, -24], [-217, -214], [218, 213], [21, 24]]
    transforms1 = [
        v_trans.Decode(to_pil=False),
        v_trans.Perspective(start_points, end_points).device(device_target="Ascend"),
    ]
    transform1 = t_trans.Compose(transforms1)
    ds1 = ds1.map(input_columns=["image"], operations=transform1)
    for _ in ds1.create_dict_iterator(output_numpy=True):
        pass

    # When input image format is png, Perspective interface call successful
    image = cv2.imread(image_png)
    start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
    perspective_op_cpu = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST)(image)
    out = perspective_op(image)
    assert (out == perspective_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_perspective_exception_01():
    """
    Feature: Perspective operation on device
    Description: Testing the Perspective Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # eager mode, when input is 3-dimensional shape(128, 128, 4) data, Perspective interface call failed
    image = np.random.randint(0, 128, (128, 128, 4)).astype(np.uint8)
    start_points = [[1, 1], [2, 2], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[H,W,C\] is not 1, 3, but got: 4"):
        perspective_op = v_trans.Perspective(start_points, end_points).device(device_target="Ascend")
        _ = perspective_op(image)

    # eager mode, when input is 2-dimensional uint16 data, Perspective interface call failed
    image = np.random.randint(0, 128, (128, 128)).astype(np.uint16)
    start_points = [[10, 1], [2, 20], [30, 3], [40, 4]]
    end_points = [[20, 2], [10, 1], [3, 30], [4, 40]]
    with pytest.raises(RuntimeError) as e_log:
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
        perspective_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    image = np.random.randint(0, 128, (128, 128)).astype(np.float64)
    start_points = [[10, 1], [2, 20], [30, 3], [40, 4]]
    end_points = [[20, 2], [10, 1], [3, 30], [4, 40]]
    with pytest.raises(RuntimeError) as e_log:
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
        perspective_op(image)
        assert "The input data is not uint8 or float32" in str(e_log.value)

    # When input is 3-dimensional float32 data, unsupported interpolation method
    image = np.random.randint(0, 128, (128, 128, 4)).astype(np.uint8)
    start_points = [[1089, 1], [2, 202], [30, 3], [40, 402]]
    end_points = [[20, 2], [10, 1], [3, 303], [4, 4018]]
    with pytest.raises(RuntimeError) as e_log:
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.PILCUBIC).device(
            device_target="Ascend")
        perspective_op(image)
        assert "The current InterpolationMode is not supported by DVPP" in str(e_log.value)

    with pytest.raises(RuntimeError) as e_log:
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.AREA).device(device_target="Ascend")
        perspective_op(image)
        assert "The current InterpolationMode is not supported by DVPP" in str(e_log.value)

    # When input is 3-dimensional float32 data, unsupported interpolation method
    image = cv2.imread(image_png)
    start_points = [[1089, 1], [2, 202], [30, 3], [40, 402]]
    end_points = [[20, 2], [10, 1], [3, 303], [4, 4018]]
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR and NEAREST"):
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.ANTIALIAS).device(device_target="Ascend")
        perspective_op(image)

    # When input is 3-dimensional float32 data, unsupported interpolation method
    image = Image.open(image_png)
    start_points = [[1089, 1], [2, 202], [30, 3], [40, 402]]
    end_points = [[20, 2], [10, 1], [3, 303], [4, 4018]]
    with pytest.raises(TypeError,
                       match="The input PIL Image cannot be executed on Ascend, you can convert "
                             "the input to the numpy ndarray type."):
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.BILINEAR).device(device_target="Ascend")
        perspective_op(image)
        image.close()

    # Perspective interface unsupported interpolation methods BICUBIC, CUBIC
    image = np.random.randn(255, 255, 3).astype(np.float32)
    start_points = [
        [-100, 100], [200, -200], [300, 300], [-400, 400]
    ]
    end_points = [
        [-2147483648, -2147483648], [-21474836, -21474836], [2147483647, 2147483647],
        [2147483647, 2147483647]
    ]
    with pytest.raises(RuntimeError) as e_log:
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.BICUBIC).device(device_target="Ascend")
        perspective_op(image)
        assert "Only support bilinear(0) and nearest(1), current interpolation is 2." in str(e_log.value)

    image = np.random.randn(255, 255, 3).astype(np.float32)
    start_points = [
        [-100, 100], [200, -200], [300, 300], [-400, 400]
    ]
    end_points = [
        [-2147483648, -2147483648], [-21474836, -21474836], [2147483647, 2147483647],
        [2147483647, 2147483647]
    ]
    with pytest.raises(RuntimeError) as e_log:
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.CUBIC).device(device_target="Ascend")
        perspective_op(image)
        assert "Only support bilinear(0) and nearest(1), current interpolation is 2." in str(e_log.value)

    # When input is PIL, parameter interpolation is Inter.AREA, Perspective interface call failed
    with pytest.raises(RuntimeError) as e_log:
        image = cv2.imread(image_png)
        start_points = [[-28, -2474], [-214, -2144], [2147, 21448], [28, 2144]]
        end_points = [[-21, -24], [-217, -214], [218, 213], [21, 24]]
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.AREA).device(device_target="Ascend")
        perspective_op(image)
        assert "The current InterpolationMode is not supported by DVPP" in str(e_log.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_perspective_exception_02():
    """
    Feature: Perspective operation on device
    Description: Testing the Perspective Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When input is PIL, parameter interpolation is Inter.PILCUBIC, Perspective interface call failed
    with pytest.raises(RuntimeError) as e_log:
        image = cv2.imread(image_bmp)
        start_points = [[-28, -2474], [-214, -2144], [2147, 21448], [28, 2144]]
        end_points = [[-21, -24], [-217, -214], [218, 213], [21, 24]]
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.PILCUBIC).device(
            device_target="Ascend")
        perspective_op(image)
        assert "The current InterpolationMode is not supported by DVPP" in str(e_log.value)

    # When input is 1-dimensional data, Perspective interface call failed
    with pytest.raises(RuntimeError, match="invalid input shape, only support NHWC input, got rank: 1"):
        image = np.random.randn(255, ).astype(np.float64)
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = v_trans.Perspective(start_points, end_points, v_Inter.NEAREST).device(device_target="Ascend")
        perspective_op(image)

    # When input is 4-dimensional data, Perspective interface call failed.
    with pytest.raises(RuntimeError,
                       match=r"The channel of the input tensor of shape \[N,H,W,C\] is not 1, 3, but got: 4"):
        image = np.random.randn(128, 128, 128, 4)
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = v_trans.Perspective(start_points, end_points).device(device_target="Ascend")
        perspective_op(image)
    # When input image is 4-dimensional NHWC, N must be 1.
    with pytest.raises(RuntimeError,
                       match="The input tensor NHWC should be 1HWC or HWC."):
        image = np.random.randn(2, 128, 128, 3).astype(np.float32)
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = v_trans.Perspective(start_points, end_points).device(device_target="Ascend")
        perspective_op(image)

    # When input is tuple data, Perspective interface call failed
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        image = tuple(np.random.randn(128, 128, 3))
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = v_trans.Perspective(start_points, end_points).device(device_target="Ascend")
        perspective_op(image)

    # When input is list data, Perspective interface call failed
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8).tolist()
        start_points = ((1, 1), (2, 2), (3, 3), (4, 4))
        end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
        perspective_op = v_trans.Perspective(start_points, end_points).device(device_target="Ascend")
        perspective_op(image)

    # When no start_points parameter, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    interpolation = v_Inter.BILINEAR
    with pytest.raises(TypeError, match="missing a required argument: 'start_points'"):
        perspective_op = v_trans.Perspective(end_points=end_points, interpolation=interpolation).device(
            device_target="Ascend")
        perspective_op(image)

    # When no end_points parameter, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="missing a required argument: 'end_points'"):
        perspective_op = v_trans.Perspective(start_points=start_points).device(device_target="Ascend")
        perspective_op(image)

    # When parameter start_points list length is 3, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="start_points should be a list or tuple of length 4."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter start_points is list, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [2, 2]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter start_points is 1-tuple, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 2])
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_perspective_exception_03():
    """
    Feature: Perspective operation on device
    Description: Testing the Perspective Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When parameter start_points value is str, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], ['1', 1], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[1\\]\\[0\\] with value 1 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter start_points value is float, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3.0], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points\\[2\\]\\[1\\] with value 3.0 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter start_points value is bool, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = True
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument start_points with value True is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'bool'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When sub-list length in parameter start_points is not equal to 2, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3], (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="start_points\\[2\\] should be a list or tuple of length 2."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter end_points list length is 3, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3]]
    with pytest.raises(TypeError, match="end_points should be a list or tuple of length 4."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter end_points is list, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [2, 2]
    with pytest.raises(TypeError, match="Argument end_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter end_points is 1-tuple, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = ([2, 2])
    with pytest.raises(TypeError, match="Argument end_points\\[0\\] with value 2 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter end_points value is str, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [[2, 2], ['1', 1], [3, 3], [4, 4]]
    with pytest.raises(TypeError, match="Argument end_points\\[1\\]\\[0\\] with value 1 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter end_points value is float, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3.0], [4, 4]]
    with pytest.raises(TypeError, match="Argument end_points\\[2\\]\\[1\\] with value 3.0 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter end_points value is bool, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = True
    with pytest.raises(TypeError, match="Argument end_points with value True is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'bool'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_perspective_exception_04():
    """
    Feature: Perspective operation on device
    Description: Testing the Perspective Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When sub-list length in parameter end_points is not equal to 2, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    end_points = ([2, 0], [1, 1], [3], (4, 4))
    with pytest.raises(TypeError, match="end_points\\[2\\] should be a list or tuple of length 2."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When value in parameter start_points is less than -2147483649, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [1, 1], [3, 3], [4, -2147483649]]
    end_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    with pytest.raises(ValueError, match="Input start_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When value in parameter start_points is greater than 2147483647, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = [[2, 2], [-21474836, 1], [3, 3], [4, 2147483648]]
    end_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    with pytest.raises(ValueError, match="Input start_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When value in parameter end_points is less than -2147483649, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [1, 1], [3, 3], [4, -2147483649]]
    with pytest.raises(ValueError, match="Input end_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When value in parameter end_points is greater than 2147483647, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21474836, 1], [3, 3], [4, 2147483648]]
    with pytest.raises(ValueError, match="Input end_points\\[3\\]\\[1\\] is not within the required interval of "
                                         "\\[-2147483648, 2147483647\\]."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points).device(
            device_target="Ascend")
        perspective_op(image)

    # When parameter interpolationMode is bool, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = True
    with pytest.raises(TypeError, match="Argument interpolation with value True is not of type \\[<enum 'Inter'>\\],"
                                        " but got <class 'bool'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points,
                                             interpolation=interpolation).device(device_target="Ascend")
        perspective_op(image)

    # When parameter interpolationMode is int, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = 1
    with pytest.raises(TypeError, match="Argument interpolation with value 1 is not of type \\[<enum 'Inter'>\\],"
                                        " but got <class 'int'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points,
                                             interpolation=interpolation).device(device_target="Ascend")
        perspective_op(image)

    # When parameter interpolationMode is str, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = 'v_Inter.AREA'
    with pytest.raises(TypeError, match="Argument interpolation with value v_Inter.AREA is not of type "
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points,
                                             interpolation=interpolation).device(device_target="Ascend")
        perspective_op(image)

    # When parameter interpolationMode is list, Perspective interface call failed
    image = np.random.randn(3, 4, 3)
    start_points = ([2, 0], [1, 1], [3, -2], (4, 4))
    end_points = [[2, 2], [-21478, 1], [3, 3], [4, 214788]]
    interpolation = [v_Inter.BICUBIC]
    with pytest.raises(TypeError, match="Argument interpolation with value \\[<Inter.BICUBIC: 3>\\] is not of type "
                                        "\\[<enum 'Inter'>\\], but got <class 'list'>."):
        perspective_op = v_trans.Perspective(start_points=start_points, end_points=end_points,
                                             interpolation=interpolation).device(device_target="Ascend")
        perspective_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_dvpp_perspective_exception_05():
    """
    Feature: Perspective operation on device
    Description: Testing the Perspective Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # eager mode, device_target error
    image = np.random.randint(0, 128, (128, 128, 3)).astype(np.uint8)
    start_points = [[1, 1], [2, 2], [3, 3], [4, 4]]
    end_points = [[2, 2], [1, 1], [3, 3], [4, 4]]
    with pytest.raises(ValueError, match=r"Input device_target is not within the valid set of \['CPU', 'Ascend'\]."):
        perspective_op = v_trans.Perspective(start_points, end_points, ).device(device_target="GPU")
        _ = perspective_op(image)

    image = np.random.randint(0, 128, (8193, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="the input shape should be from \\[6, 10\\] to \\[8192, 4096\\]"):
        perspective_op = v_trans.Perspective(start_points, end_points, ).device(device_target="Ascend")
        _ = perspective_op(image)


if __name__ == '__main__':
    test_dvpp_perspective_operation_01()
    test_dvpp_perspective_operation_02()
    test_dvpp_perspective_exception_01()
    test_dvpp_perspective_exception_02()
    test_dvpp_perspective_exception_03()
    test_dvpp_perspective_exception_04()
    test_dvpp_perspective_exception_05()
