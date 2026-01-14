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
Testing DVPP Invert operation
"""
import os
import pytest
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=114.0)
def test_dvpp_invert_operation_01():
    """
    Feature: Invert operation on device
    Description: Testing the normal functionality of the Invert operator on device
    Expectation: The Output is equal to the expected output
    """
    # Normal test of Invert operator, pipeline mode, numpy image
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    invert_op = vision.Invert()
    invert_op_dvpp = vision.Invert().device(device_target="Ascend")
    dataset2 = dataset2.map(input_columns=["image"], operations=invert_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=invert_op_dvpp)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image == image_aug).all()

    # Using Invert operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = vision.Decode().device("Ascend")(img_bytes)
        img_ops = vision.Invert().device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)
        img_ops = vision.Invert()(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Normal test of Invert operator, eager mode, jpg image
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_1.jpg")
    image = cv2.imread(image_file)
    invert_op = vision.Invert()(image)
    invert_op_dvpp = vision.Invert().device(device_target="Ascend")(image)
    assert (invert_op == invert_op_dvpp).all()

    # Normal test of Invert operator, eager mode, bmp image
    image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_bmp)
    invert_op = vision.Invert()(image)
    invert_op_dvpp = vision.Invert().device(device_target="Ascend")(image)
    assert (invert_op == invert_op_dvpp).all()

    # Normal test of Invert operator, eager mode, png image
    image_path = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_path)
    invert_op = vision.Invert()(image)
    invert_op_dvpp = vision.Invert().device(device_target="Ascend")(image)
    assert (invert_op == invert_op_dvpp).all()

    # Normal test of Invert operator, eager mode, numpy random number image
    image = np.random.randn(366, 255, 3).astype(np.uint8)
    invert_op = vision.Invert()(image)
    invert_op_dvpp = vision.Invert().device(device_target="Ascend")(image)
    assert (invert_op == invert_op_dvpp).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=98.0)
def test_dvpp_invert_exception_01():
    """
    Feature: Invert operation on device
    Description: Testing the Invert Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Invert operator does not support PIL images
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1",
                              "1_1.jpg")
    with Image.open(image_file) as image:
        with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend, "
                                            "you can convert the input to the numpy ndarray type"):
            _ = vision.Invert().device(device_target="Ascend")(image)

    # Test of Invert operator, eager mode, four-dimensional data
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    invert_op = vision.Invert()(new_arr)
    invert_op_dvpp = vision.Invert().device(device_target="Ascend")(image)
    assert (invert_op == invert_op_dvpp).all()

    # Exception test of Invert operator, eager mode, one-dimensional data
    image = np.random.randn(200, ).astype(np.uint8)
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="invalid input shape, only support NHWC input, got rank: 1"):
        invert_op(image)

    # Exception test of Invert operator, eager mode, four-channel data
    image = np.random.randn(128, 128, 4).astype(np.uint8)
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 4"):
        invert_op(image)

    # Exception test of Invert operator, eager mode, input data as list data
    image = np.random.randn(128, 128, 3).astype(np.uint8).tolist()
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        invert_op(image)

    # Exception test of Invert operator, eager mode, input data as int data
    image = 10
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'int'>."):
        invert_op(image)

    # Exception test of Invert operator, eager mode, input data as tuple data
    image = (10,)
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        invert_op(image)

    # Exception test of Invert operator, eager mode, input data as tuple data
    image = np.random.randn(128, 128, 3).astype(np.float32)
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(RuntimeError) as e:
        invert_op(image)
        assert "The input data is not uint8" in str(e.value)

    # Exception test of Invert operator, eager mode, input data as tuple data
    image = np.random.randn(3, 3, 3).astype(np.uint8)
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="DvppInvertOp: the input shape should be from \\[4, 6\\] to "
                             "\\[8192, 4096\\], but got \\[3, 3\\]"):
        invert_op(image)

    image = np.random.randn(8193, 4097, 3).astype(np.uint8)
    invert_op = vision.Invert().device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="DvppInvertOp: the input shape should be from \\[4, 6\\] to \\[8192, 4096\\],"
                             " but got \\[8193, 4097\\]"):
        invert_op(image)


if __name__ == '__main__':
    test_dvpp_invert_operation_01()
    test_dvpp_invert_exception_01()
