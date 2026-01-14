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
Testing DVPP Crop operation
"""
import os
import pytest
import cv2
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")


def dir_data():
    """Obtain the dataset"""
    data_list = []
    data_dir1 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train/")
    data_dir2 = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_1.jpg")
    data_list.append(data_dir1)
    data_list.append(data_dir2)
    return data_list


class VideoDataset:
    def __init__(self):
        self._label = np.zeros((100, 1))

    def __getitem__(self, index):
        image = np.fromfile(dir_data()[1], dtype=np.uint8)
        image = v_trans.Decode().device("Ascend")(image)
        image = v_trans.Crop((0, 0), 224).device("Ascend")(image)
        return image, self._label[index]

    @staticmethod
    def __len__():
        return 10


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_crop_operation_01():
    """
    Feature: Crop operation on device
    Description: Testing the normal functionality of the Crop operator on device
    Expectation: The Output is equal to the expected output
    """
    # Using crop dvpp operator in pyfunc
    ms.set_context(device_target="Ascend")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=dir_data()[0], shuffle=False)
    coordinates = [50, 10]
    size = (5, 6)
    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.Crop(coordinates=coordinates, size=size).device("Ascend")(img_decode)
        return img_ops
    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.Crop(coordinates=coordinates, size=size)(img_decode)
        return img_ops
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # input is bmp image
    image = cv2.imread(image_bmp)
    coordinates = [50, 10]
    size = (5, 6)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()

    # input is png image
    image = cv2.imread(image_png)
    coordinates = [50, 10]
    size = (150, 250)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()

    # input is jpg image
    image = cv2.imread(image_jpg)
    coordinates = [10, 0]
    size = (150, 250)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()

    # input is gif image
    image = Image.open(image_gif)
    img_array = np.array(image)
    coordinates = [10, 0]
    size = (150, 100)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(img_array)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(img_array)
    assert (crop_op == crop_op_cpu).all()
    image.close()

    # Crop operator: Test size is 10
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    coordinates = (0, 0)
    size = 10
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=crop_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=crop_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert (data1["image"] == data2["image"]).all()

    # Crop operator: Test size is (100, 150)
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False, decode=True)
    coordinates = (10, 20)
    size = (100, 150)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)
    dataset2 = dataset2.map(input_columns=["image"], operations=crop_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=crop_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert (data1["image"] == data2["image"]).all()

    # Crop operator: Test coordinates = [300, 100], size = (584, 618)
    image = np.random.randn(900, 900).astype(np.uint8)
    coordinates = [300, 100]
    size = (584, 618)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()

    # Crop operator: Test input is 2d
    image = np.random.randn(560, 560).astype(np.float32)
    coordinates = [0, 0]
    size = (560, 560)
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=26.0)
def test_dvpp_crop_operation_02():
    """
    Feature: Crop operation on device
    Description: Testing the normal functionality of the Crop operator on device
    Expectation: The Output is equal to the expected output
    """
    # Crop operator: Test input.shape is (300, 300, 3)
    image = np.random.randint(0, 255, (300, 300, 3)).astype(np.float32)
    coordinates = (200, 0)
    size = [4, 6]
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()

    # Crop operator: Test input.shape is (658, 714, 3)
    image = np.random.randint(0, 255, (658, 714, 3)).astype(np.float32)
    coordinates = (100, 50)
    size = [200, 200]
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    assert (image[100:300, 50:250, 0:3] == crop_op).all()
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()

    # Crop operator: dimension have one
    image = np.random.randn(1024, 2048, 1).astype(np.float32)
    image_cpu = np.reshape(image, (1024, 2048))
    coordinates = (1, 1)
    size = 380
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    assert crop_op.shape == (380, 380)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image_cpu)
    assert (crop_op == crop_op_cpu).all()

    image = np.random.randn(1, 1024, 2048, 3).astype(np.float32)
    image_cpu = np.reshape(image, (1024, 2048, 3))
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    assert crop_op.shape == (380, 380, 3)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image_cpu)
    assert (crop_op == crop_op_cpu).all()

    # Crop operator: Test size is less than image.shape
    image = np.random.randint(0, 255, (200, 200, 3)).astype(np.float32)
    coordinates = (100, 100)
    size = [50, 50]
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
    crop_op_cpu = v_trans.Crop(coordinates=coordinates, size=size)(image)
    assert (crop_op == crop_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=8.0)
def test_dvpp_crop_exception_01():
    """
    Feature: Crop operation on device
    Description: Testing the Crop Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Crop operator: input is PIL
    with Image.open(dir_data()[1]) as image:
        coordinates = [300, 100]
        size = (584, 618)
        crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")
        with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend, "
                                            "you can convert the input to the numpy ndarray type"):
            crop_op(image)

    # Crop operator: Crop exceeds original dimensions
    image = np.random.randn(600, 600)
    coordinates = [300, 100]
    size = (584, 618)
    with pytest.raises(RuntimeError, match="Crop height dimension: 884 exceeds image height: 600"):
        _ = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)

    # Crop operator: Test input.shape is (1, 1, 3)
    image = np.random.randint(0, 255, (1, 1, 3)).astype(np.uint8)
    coordinates = (0, 0)
    size = 1
    with pytest.raises(RuntimeError,
                       match="DvppCropOp: the input shape should be "
                             "from \\[4, 6\\] to \\[32768, 32768\\], but got \\[1, 1\\]"):
        _ = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)

    image = np.random.randint(0, 255, (200, 200, 3)).astype(np.float32)
    size = 1
    with pytest.raises(RuntimeError,
                       match="DvppCropOp: the output shape should be "
                             "from \\[4, 6\\] to \\[32768, 32768\\], but got \\[1, 1\\]"):
        _ = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)

    # Crop operator: Test input.shape is (800, 1024, 10)
    image = np.random.randint(0, 255, (800, 1024, 10)).astype(np.uint8)
    coordinates = (200, 0)
    size = 500
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 10"):
        _ = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)

    # Crop operator, Testing for invalid input types
    image = np.random.randint(0, 255, (600, 100, 3)).astype(np.float64)
    coordinates = (200, 0)
    size = 500
    with pytest.raises(RuntimeError) as error_log:
        _ = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
        assert "The input data is not uint8 or float32" in str(error_log.value)

    image = np.random.randint(0, 255, (600, 100, 3)).astype(np.int64)
    coordinates = (200, 0)
    size = 500
    with pytest.raises(RuntimeError) as error_log:
        _ = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")(image)
        assert "The input data is not uint8 or float32" in str(error_log.value)

    # Crop operator: Test input.shape is  (658, 714, 3, 3)
    image = np.random.randint(0, 255, (658, 714, 3, 3)).astype(np.uint8)
    coordinates = (1, 1)
    size = 380
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match=r"The input tensor NHWC should be 1HWC or HWC."):
        crop_op(image)

    # Crop operator: Test input.shape is  (658,)
    image = np.random.randint(0, 255, (658,)).astype(np.uint8)
    coordinates = (10, 10)
    size = 380
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="DvppCrop: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        crop_op(image)

    # Crop operator: Test coordinates is greater than image.shape
    image = np.random.randint(0, 255, (200, 200, 3)).astype(np.uint8)
    coordinates = (201, 1)
    size = 10
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match=r"Crop: Crop height dimension: 211 exceeds image height: 200"):
        crop_op(image)

    # Crop operator: Test coordinates plus size is greater than image.shape
    image = np.random.randint(0, 255, (200, 200, 3)).astype(np.uint8)
    coordinates = (100, 100)
    size = [101, 50]
    crop_op = v_trans.Crop(coordinates=coordinates, size=size).device(device_target="Ascend")
    with pytest.raises(RuntimeError,
                       match="Crop: Crop height dimension: 201 exceeds image height: 200"):
        crop_op(image)

    # Crop operator: Test coordinates is less than 0
    with pytest.raises(ValueError, match=("Input coordinates\\[0\\] is not within the "
                                          "required interval of \\[0, 2147483647\\].")):
        v_trans.Crop(coordinates=[-1, 100], size=100).device(device_target="Ascend")

    # Crop operator: Test coordinates is greater than 2147483647
    with pytest.raises(ValueError, match=("Input coordinates\\[1\\] is not within the "
                                          "required interval of \\[0, 2147483647\\].")):
        v_trans.Crop(coordinates=[10, 2147483648], size=100).device(device_target="Ascend")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_crop_exception_02():
    """
    Feature: Crop operation on device
    Description: Testing the Crop Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Crop operator: Test coordinates is float
    with pytest.raises(TypeError, match="Argument coordinates\\[1\\] with value 10.0 is not of "
                                         "type \\[<class 'int'>\\], but got <class 'float'>."):
        v_trans.Crop(coordinates=[10, 10.0], size=100).device(device_target="Ascend")

    # Crop operator: Test coordinates is True
    with pytest.raises(TypeError, match="Argument coordinates\\[1\\] with value True is not of "
                                         "type \\(<class 'int'>,\\), but got <class 'bool'>."):
        v_trans.Crop(coordinates=(10, True), size=100).device(device_target="Ascend")

    # Crop operator: Test coordinates is np
    with pytest.raises(TypeError, match="Argument coordinates with value \\[10 10\\] is not of type \\[<class "
                                         "'list'>, <class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        v_trans.Crop(coordinates=np.array([10, 10]), size=100).device(device_target="Ascend")

    # Crop operator: Test coordinates is int
    with pytest.raises(TypeError, match="Argument coordinates with value 20 is not of type \\[<class "
                                         "'list'>, <class 'tuple'>\\], but got <class 'int'>."):
        v_trans.Crop(coordinates=20, size=100).device(device_target="Ascend")

    # Crop operator: Test coordinates is 3-tuple
    with pytest.raises(TypeError, match="Coordinates should be a list/tuple \\(y, x\\) of length 2."):
        v_trans.Crop(coordinates=(10, 10, 10), size=100).device(device_target="Ascend")

    # Crop operator: Test coordinates is 1-list
    with pytest.raises(TypeError, match="Coordinates should be a list/tuple \\(y, x\\) of length 2."):
        v_trans.Crop(coordinates=[10], size=100).device(device_target="Ascend")

    # Crop operator: Test no coordinates
    with pytest.raises(TypeError, match="missing a required argument: 'coordinates'"):
        v_trans.Crop(size=100).device(device_target="Ascend")

    # Crop operator: Test size is float
    with pytest.raises(TypeError, match="Argument size with value 100.0 is not of type \\[<class "
                                         "'int'>, <class 'list'>, <class 'tuple'>\\], but got <class 'float'>."):
        v_trans.Crop(coordinates=[10, 10], size=100.0).device(device_target="Ascend")

    # Crop operator: Test size is 0
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        v_trans.Crop(coordinates=[10, 10], size=0).device(device_target="Ascend")

    # Crop operator: Test size is greater than 2147483647
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        v_trans.Crop(coordinates=[10, 10], size=[2147483648, 10]).device(device_target="Ascend")

    # Crop operator: Test size is True
    with pytest.raises(TypeError, match="Argument size\\[0\\] with value True is not of "
                                         "type \\(<class 'int'>,\\), but got <class 'bool'>."):
        v_trans.Crop(coordinates=[10, 10], size=[True, 10]).device(device_target="Ascend")

    # Crop operator: Test size is np
    with pytest.raises(TypeError, match="Argument size with value \\[20 20\\] is not of type \\[<class 'int'>, "
                                         "<class 'list'>, <class 'tuple'>\\], but got <class 'numpy.ndarray'>."):
        v_trans.Crop(coordinates=[10, 10], size=np.array([20, 20])).device(device_target="Ascend")

    # Crop operator: Test size is 3-list
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        v_trans.Crop(coordinates=[10, 10], size=[10, 10, 10]).device(device_target="Ascend")

    # Crop operator: Test size is 1-tuple
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple \\(h, w\\) of length 2."):
        v_trans.Crop(coordinates=[10, 10], size=(50,)).device(device_target="Ascend")

    # Crop operator: Test no size
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        v_trans.Crop(coordinates=[10, 10]).device(device_target="Ascend")

    # Crop operator: Test more Parameters
    with pytest.raises(TypeError, match="too many positional arguments"):
        v_trans.Crop([10, 10], (50, 50), 100).device(device_target="Ascend")


if __name__ == '__main__':
    test_dvpp_crop_operation_01()
    test_dvpp_crop_operation_02()
    test_dvpp_crop_exception_01()
    test_dvpp_crop_exception_02()
