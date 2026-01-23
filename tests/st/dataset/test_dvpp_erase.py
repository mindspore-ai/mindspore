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
Testing DVPP Erase operation
"""
import os
import numpy as np
import pytest
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    data_dir3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    data_dir4 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    data_dir5 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    data_dir6 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    data_list.append(data_dir3)
    data_list.append(data_dir4)
    data_list.append(data_dir5)
    data_list.append(data_dir6)
    return data_list


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=194.0)
def test_dvpp_erase_operation_01():
    """
    Feature: Erase operation on device
    Description: Testing the normal functionality of the Erase operator on device
    Expectation: The Output is equal to the expected output
    """
    # Use Erase operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    v = (30, 50, 100)
    def pyfunc1(img_bytes):
        img_decode = vision.Decode().device("Ascend")(img_bytes)
        img_ops = vision.Erase(1, 4, 20, 30, v).device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)
        img_ops = vision.Erase(1, 4, 20, 30, v)(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Erase operator: input HWC Numpy uint8 format
    image = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    v = (30, 5, 100)
    erase = vision.Erase(1, 4, 20, 30, v)(image)
    erase_op = vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)
    assert (erase_op == erase).all()

    # Erase operator: input HWC Numpy float32 format
    image = np.random.randn(30, 60, 3).astype(np.float32)
    v = (0.1, 0.5, 1)
    erase = vision.Erase(1, 4, 20, 30, v)(image)
    erase_op = vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)
    assert np.allclose(erase, erase_op, rtol=1, atol=5)

    # Erase operator: no argument value
    image = np.random.randint(0, 255, (4, 6, 1)).astype(np.uint8)
    vision.Erase(top=1, left=1, height=2, width=2, inplace=True).device(device_target="Ascend")(image)

    # Erase operator: no argument inplace
    image = np.random.randn(8192, 4096, 3).astype(np.uint8)
    v = (30, 5, 100)
    erase = vision.Erase(top=1, left=2, height=1, width=3, value=v)(image)
    erase_op = vision.Erase(top=1, left=2, height=1, width=3, value=v).device(device_target="Ascend")(image)
    assert (erase_op == erase).all()

    # Erase operator: value contains float
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 3.5, 100)
    erase_op = vision.Erase(10, 10, 10, 10, v, False).device(device_target="Ascend")(image)
    erase = vision.Erase(10, 10, 10, 10, v, False)(image)
    assert np.allclose(erase, erase_op, rtol=1, atol=1)

    # Erase operator: input HWC Numpy float32 format
    image = np.random.randn(30, 60, 3).astype(np.float32)
    v = (0.1, 0.5, 1)
    erase = vision.Erase(1, 4, 20, 30, v)(image)
    erase_op = vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)
    assert np.allclose(erase, erase_op, rtol=1, atol=5)

    # Erase operator: Test input dimension contains 1
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    v = (30, 5, 100)
    erase_op = vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)
    erase = vision.Erase(1, 4, 20, 30, v)(new_arr)
    assert (erase_op == erase).all()
    assert erase_op.shape == (128, 128, 3)

    # Erase operator: Test 4-dimensional image processing normal
    image = np.random.randint(0, 255, (1, 256, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (256, 128, 3))
    v = (30, 5, 100)
    erase_op = vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)
    erase = vision.Erase(1, 4, 20, 30, v)(new_arr)
    assert (erase_op == erase).all()
    assert erase_op.shape == (256, 128, 3)

    # Erase operator: When the channel is 1, the value parameter length is 1
    image = np.random.randint(0, 255, (30, 30, 1), dtype=np.uint8)
    v = 5
    vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_erase_exception_01():
    """
    Feature: Erase operation on device
    Description: Testing the Erase Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Erase operator: input CHW format
    image = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    with Image.open(image) as image:
        image = vision.ToNumpy()(image)
        image = vision.HWC2CHW()(image)
        v = (30, 50, 100)
        with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 718"):
            vision.Erase(1, 1, 2, 30, v).device(device_target="Ascend")(image)

    # Erase operator: DVPP does not support PIL format images
    image = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    with Image.open(image) as image:
        v = (30, 50, 100)
        with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend"):
            vision.Erase(1, 4, 3, 10, v).device(device_target="Ascend")(image)

    # Erase operator: input HWC Numpy uint8 format
    image = np.random.randn(3, 20, 30).astype(np.uint8)
    v = (30, 50, 100)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 30"):
        vision.Erase(1, 4, 3, 10, v).device(device_target="Ascend")(image)

    # Erase operator: no argument top
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'top'"):
        vision.Erase(left=3, height=20, width=20, value=v, inplace=False).device(device_target="Ascend")(image)

    # Erase operator: no argument left
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'left'"):
        vision.Erase(top=1, height=20, width=20, value=v, inplace=False).device(device_target="Ascend")(image)

    # Erase operator: no argument height
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'height'"):
        vision.Erase(top=1, left=20, width=20, value=v, inplace=False).device(device_target="Ascend")(image)

    # Erase operator: no argument width
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    v = (30, 5, 100)

    with pytest.raises(TypeError, match="missing a required argument: 'width'"):
        vision.Erase(top=1, left=20, height=20, value=v, inplace=False).device(device_target="Ascend")(image)

    # Erase operator: invalid top parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input top is not within the required interval of \\[0, 2147483647\\]."):
        vision.Erase(2147483648, 10, 10, 10, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid top parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument top with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10.5, 10, 10, 10, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid left parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input left is not within the required interval of \\[0, 2147483647\\]."):
        vision.Erase(10, 2147483648, 10, 10, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid left parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument left with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10, 10.5, 10, 10, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid height parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input height is not within the required interval of \\[1, 2147483647\\]."):
        vision.Erase(10, 10, 2147483648, 10, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid height parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument height with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10, 10, 10.5, 10, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid width parameter Value
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input width is not within the required interval of \\[1, 2147483647\\]."):
        vision.Erase(10, 10, 10, 2147483648, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid width parameter type
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument width with value 10.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        vision.Erase(10, 10, 10, 10.5, 0, False).device(device_target="Ascend")(image)

    # Erase operator: invalid value over 255
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(ValueError, match="Input value\\[2\\] is not within the required interval of \\[0, 255\\]."):
        vision.Erase(10, 10, 10, 10, (30, 5, 256), False).device(device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=194.0)
def test_dvpp_erase_exception_02():
    """
    Feature: Erase operation on device
    Description: Testing the Erase Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Erase operator: invalid value is (2, 3)
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="value should be a single integer/float or a 3-tuple"):
        vision.Erase(10, 10, 10, 10, (2, 3), False).device(device_target="Ascend")(image)

    # Erase operator: invalid value is [2, 3]
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument value with value \\[2, 3\\] is not of type .*, but got"):
        vision.Erase(10, 10, 10, 10, [2, 3], False).device(device_target="Ascend")(image)

    # Erase operator: invalid inplace is int
    image = np.random.randn(30, 60, 3).astype(np.uint8)
    with pytest.raises(TypeError, match="Argument inplace with value 0 is not of type \\[<class 'bool'>\\], "
                                        "but got <class 'int'>."):
        vision.Erase(10, 10, 10, 10, 0, 0).device(device_target="Ascend")(image)

    # Erase operator: input 2-D Numpy
    image = np.random.randn(30, 8).astype(np.float32)
    v = (0.1, 0.5)
    with pytest.raises(TypeError, match="value should be a single integer/float or a 3-tuple"):
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: input 1-D Numpy
    image = np.random.randn(30).astype(np.float32)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="DvppErase: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: input 4 channel Numpy
    image = np.random.randn(30, 60, 4).astype(np.float32)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 4"):
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: Error when input image array is not uint8 or float32
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float64)
    with pytest.raises(RuntimeError, match="Error in dvpp"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: Error when input image channel is not 1 or 3
    image = np.random.randint(0, 255, (128, 128, 2)).astype(np.float32)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 2"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: Error when device is not Ascend or CPU
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="Input device_target is not within the valid set of"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="test")(image)

    # Erase operator: Error when 4-dimensional image N is not 1
    image = np.random.randint(0, 255, (2, 256, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: Error when data is 1-dimensional
    image = np.random.randint(0, 255, (256,)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="DvppErase: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: Error when data is 5-dimensional
    image = np.random.randint(0, 255, (1, 256, 128, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor is not of shape"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: Error when value is not aclFloatArray type
    image = np.random.randint(0, 255, (1, 256, 128, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor is not of shape"):
        v = (30, 5, 100)
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)

    # Erase operator: First parameter of image exceeds range
    image = np.random.randn(8193, 4096, 3).astype(np.uint8)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="DvppEraseOp: the input shape should be from .*, but got"):
        vision.Erase(top=1, left=2, height=1, width=3, value=v).device(device_target="Ascend")(image)

    # Erase operator: Second parameter of image exceeds range
    image = np.random.randn(8192, 4097, 3).astype(np.uint8)
    v = (30, 5, 100)
    with pytest.raises(RuntimeError, match="DvppEraseOp: the input shape should be from .*, but got"):
        vision.Erase(top=1, left=2, height=1, width=3, value=v).device(device_target="Ascend")(image)

    # Erase operator: Error when value length does not match channel value
    image = np.random.randn(30, 8, 1).astype(np.float32)
    v = (0.1, 0.5, 1)
    with pytest.raises(RuntimeError, match="The length of value should be the same as the value of channel"):
        vision.Erase(1, 4, 20, 30, v).device(device_target="Ascend")(image)


if __name__ == '__main__':
    test_dvpp_erase_operation_01()
    test_dvpp_erase_exception_01()
    test_dvpp_erase_exception_02()
