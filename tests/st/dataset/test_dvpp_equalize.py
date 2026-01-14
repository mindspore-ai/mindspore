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
Testing DVPP Equalize operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.transforms.transforms as t_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=104.0)
def test_dvpp_equalize_operation_01():
    """
    Feature: Equalize operation on device
    Description: Testing the normal functionality of the Equalize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Equalize operator:Test normal
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    dataset1 = dataset1.map(input_columns=["image"], operations=equalize_op)
    dataset2 = dataset2.map(input_columns=["image"], operations=equalize_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # Using the equalize dvpp operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.Equalize().device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.Equalize()(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # Equalize operator:Test image is jpg
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_jpg)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(image), rtol=2, atol=1)

    # Equalize operator:Test image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        img_array = np.array(image)
        equalize_op = v_trans.Equalize().device(device_target="Ascend")
        equalize_op_cpu = v_trans.Equalize()
        assert np.allclose(equalize_op(img_array), equalize_op_cpu(img_array), rtol=5, atol=1)

    # Equalize operator:Test image is bmp
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_bmp)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(image), rtol=15, atol=1)

    # Equalize operator:Test input is image opened using the cv2 method
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_1.jpg")
    image = cv2.imread(image_file)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(image), rtol=1, atol=1)

    # Equalize operator:Test input is 3-d image 01
    image = np.random.randn(468, 368, 3).astype(np.uint8)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(image), rtol=1, atol=1)

    # Equalize operator:The test input has a dimension of 1.
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(new_arr), rtol=1, atol=1)

    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128))
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(new_arr), rtol=1, atol=1)

    # Equalize operator:Test input is 2-d image
    image = np.random.randint(-255, 255, (256, 128)).astype(np.uint8)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(image), rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_equalize_operation_02():
    """
    Feature: Equalize operation on device
    Description: Testing the normal functionality of the Equalize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Equalize operator:Test image is png
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_png)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize()
    assert np.allclose(equalize_op(image), equalize_op_cpu(image), rtol=1, atol=1)

    # Equalize operator:Testing input image channel count of 1 is normal.
    image = np.random.randint(-255, 255, (256, 128, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (256, 128))
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu = v_trans.Equalize().device(device_target="CPU")
    assert np.allclose(equalize_op(image), equalize_op_cpu(new_arr), rtol=1, atol=1)

    # Equalize operator:Testing 4D image processing is functioning normally.
    image1 = np.random.randint(0, 255, (1, 256, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image1, (256, 128, 3))
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    equalize_op_cpu1 = v_trans.Equalize()
    assert np.allclose(equalize_op(image1), equalize_op_cpu1(new_arr), rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_equalize_exception_01():
    """
    Feature: Equalize operation on device
    Description: Testing the Equalize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Equalize operator:Test PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    ds2 = ds.ImageFolderDataset(data_dir)
    transforms1 = [
        v_trans.Decode(to_pil=True),
        v_trans.Equalize().device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)
    with pytest.raises(RuntimeError, match="The input PIL Image cannot be executed on Ascend"):
        for _ in zip(ds2.create_dict_iterator(output_numpy=True)):
            pass

    # Equalize operator:Test input is 3d list
    image = list(np.random.randn(128, 128, 3).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        equalize_op = v_trans.Equalize().device(device_target="Ascend")
        equalize_op(image)

    # Equalize operator:Test input is 2d tuple
    image = tuple(np.random.randint(0, 255, (20, 10)).astype(np.uint8))
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        equalize_op = v_trans.Equalize().device(device_target="Ascend")
        equalize_op(image)

    # Equalize operator:Test input is 4-d image
    image = np.random.randn(10, 468, 368, 3).astype(np.uint8)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC"):
        equalize_op(image)

    # Equalize operator:Test input is 1-d numpy data
    image = np.random.randn(200, ).astype(np.uint8)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="DvppEqualize: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        equalize_op(image)

    # Equalize operator:Test input is 4 channel numpy array
    image = np.random.randn(128, 128, 4).astype(np.uint8)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape .* is not 1, 3, but got: 4"):
        equalize_op(image)

    # Equalize operator:Test input is 3d numpy list
    image = np.random.randn(128, 128, 3).astype(np.uint8).tolist()
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        equalize_op(image)

    # Equalize operator:Test input is int
    image = 10
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>"):
        equalize_op(image)

    # Equalize operator:Test input is tuple
    image = (10,)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>"):
        equalize_op(image)

    # Equalize operator:Test no image is transferred
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="Input Tensor is not valid."):
        equalize_op()

    # Equalize operator:Test one more parameter
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        v_trans.Equalize(1).device(device_target="Ascend")

    # Equalize operator:The test throws an error when the input image array is not of type uint8 or float32.
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float64)
    with pytest.raises(RuntimeError, match="DvppEqualize: Error in dvpp processing: 2027"):
        equalize_op = v_trans.Equalize().device(device_target="Ascend")
        equalize_op(image)

    # Equalize operator:Error occurs when the number of channels in the input image is not 1 or 3.
    image = np.random.randint(0, 255, (128, 128, 2)).astype(np.float32)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 2"):
        equalize_op = v_trans.Equalize().device(device_target="Ascend")
        equalize_op(image)

    # Equalize operator:Error occurs when the test device is not an Ascend device or does not have a CPU.
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="Input device_target is not within the valid set of .*"):
        equalize_op = v_trans.Equalize().device(device_target="test")
        equalize_op(image)

    # Equalize operator:Testing 4D images: Error occurs when N is not equal to 1
    image1 = np.random.randint(0, 255, (2, 256, 128, 3)).astype(np.uint8)
    equalize_op = v_trans.Equalize().device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC"):
        equalize_op(image1)


if __name__ == '__main__':
    test_dvpp_equalize_operation_01()
    test_dvpp_equalize_operation_02()
    test_dvpp_equalize_exception_01()
