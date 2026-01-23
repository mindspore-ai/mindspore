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
Testing DVPP Affine operation
"""
import os
import numpy as np
import cv2
import pytest
from PIL import Image
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms.transforms as t_trans
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision import Inter
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"

DATA_DIR = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "xiaoji.gif")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=58.0)
def test_dvpp_affine_operation_01():
    """
    Feature: Affine operation on device
    Description: Testing the normal functionality of the Affine operator on device
    Expectation: The Output is equal to the expected output
    """
    # When the degrees parameter value is 20.0, the Affine interface is successfully called
    degrees = 20.0
    translate = (-1, 1)
    scale = 0.9
    shear = (50.0, 100.0)
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, shear=shear, scale=scale).device(
        device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, shear=shear, scale=scale)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert (data1["image"] == data2["image"]).all()

    # Using the affinedvpp operator in pyfunc
    ms.set_context(device_target="Ascend")

    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=DATA_DIR, shuffle=False)
    degrees = 20.0
    translate = (-1, 1)
    scale = 0.9
    shear = (50.0, 100.0)

    def pyfunc1(img_bytes):
        img_decode = vision.Decode().device("Ascend")(img_bytes)
        img_ops = vision.Affine(degrees=degrees, translate=translate, shear=shear, scale=scale).device("Ascend")(
            img_decode)

        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)
        img_ops = vision.Affine(degrees=degrees, translate=translate, shear=shear, scale=scale)(img_decode)

        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec)(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # When the input is uint8 and the pipeline is 3, the Affine interface is successfully called
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    degrees = 10
    translate = (1, 1)
    scale = 2.2
    shear = 1.8
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
        device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)
    assert (affine_op == affine_op_cpu).all()

    # When the degrees parameter value is 0 and Inter.BILINEAR, the Affine interface is successfully called
    degrees = 20
    translate = [-0, 0]
    scale = 11.11
    shear = [10.12, 22.09]
    resample = Inter.BILINEAR
    fill_value = (1, 2, 3)
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value).device(device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                  resample=resample, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)

    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=34.0)
def test_dvpp_affine_operation_02():
    """
    Feature: Affine operation on device
    Description: Testing the normal functionality of the Affine operator on device
    Expectation: The Output is equal to the expected output
    """
    # When the degrees parameter value is 0, the Affine interface is successfully called
    degrees = 0
    translate = [-0, 0]
    scale = 0.9
    shear = (10.12, 22.09)
    resample = Inter.NEAREST
    fill_value = (1, 2, 3)
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value).device(device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                  resample=resample, fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)

    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1)

    # The translate parameter is a normal array and resample is BILINEAR.
    degrees = 15.1
    translate = [-1, 0.9]
    scale = 100.10
    shear = (10.0, 10.0)
    resample = Inter.BILINEAR
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample).device(
        device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)

    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1)

    # When the translate parameter is -1, the Affine interface is successfully called
    degrees = 123.321
    translate = (-1, -1)
    scale = 0.01
    shear = (5.1, 0.10)
    resample = Inter.LINEAR
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample).device(
        device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)

    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert (data1["image"] == data2["image"]).all()

    # When the translate parameter value is 0,1, the Affine interface is successfully called
    degrees = 150.5
    translate = [0, 1]
    scale = 1.1112
    shear = [1.0, 4.0]
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
        device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1)

    # When the shear parameter has negative numbers, the Affine interface is successfully called
    degrees = 1.005
    translate = (-0.45, 0)
    scale = 256.0
    shear = [-180.0, 180.0]
    resample = Inter.LINEAR
    fill_value = 255
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                              fill_value=fill_value).device(device_target="Ascend")
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                                  fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)

    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_dvpp_affine_operation_03():
    """
    Feature: Affine operation on device
    Description: Testing the normal functionality of the Affine operator on device
    Expectation: The Output is equal to the expected output
    """
    # When the input image is jpg, the Affine interface is successfully called
    image = cv2.imread(image_jpg)
    degrees = 0
    translate = (-0.15648, 1)
    scale = 100.2
    shear = 0.001
    resample = Inter.BILINEAR
    fill_value = 1
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              resample=resample, fill_value=fill_value).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                  resample=resample, fill_value=fill_value)(image)
    assert (affine_op == affine_op_cpu).all()

    # When the input image is png, the Affine interface is successfully called
    image = cv2.imread(image_png)
    affine_op = vision.Affine(degrees=11.56, translate=[0, -0.100], scale=1.0, shear=180,
                              resample=Inter.NEAREST, fill_value=255).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=11.56, translate=[0, -0.100], scale=1.0, shear=180,
                                  resample=Inter.NEAREST, fill_value=255)(image)
    assert np.allclose(affine_op, affine_op_cpu, rtol=1)

    # When the input image is gif and the input is 2-dimensional, the Affine interface is successfully called
    image = Image.open(image_gif)
    img_array = np.array(image)
    affine_op = vision.Affine(degrees=180, translate=[-0.256, 0.255], scale=2.1, shear=10).device(
        device_target="Ascend")(img_array)
    affine_op_cpu = vision.Affine(degrees=180, translate=[-0.256, 0.255], scale=2.1, shear=10)(img_array)
    assert (affine_op == affine_op_cpu).all()
    image.close()

    # When the input image is bmp, the Affine interface is successfully called
    image = cv2.imread(image_bmp)
    affine_op = vision.Affine(degrees=100, translate=[0, 0], scale=16777216, shear=[2.2, 8.8],
                              resample=Inter.BILINEAR, fill_value=(180, 26, 109)).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=100, translate=[0, 0], scale=16777216, shear=[2.2, 8.8],
                                  resample=Inter.BILINEAR, fill_value=(180, 26, 109))(image)
    assert (affine_op == affine_op_cpu).all()

    # When the input shape is (346, 489, 3) and shear is 0, the Affine interface is successfully called
    image = np.random.randn(346, 489, 3).astype(np.float32)
    affine_op = vision.Affine(degrees=180.0, translate=(0, 0), scale=0.03, shear=0,
                              resample=Inter.BILINEAR, fill_value=(64, 64, 64)).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=180.0, translate=(0, 0), scale=0.03, shear=0,
                                  resample=Inter.BILINEAR, fill_value=(64, 64, 64))(image)
    assert np.allclose(affine_op, affine_op_cpu, rtol=1)

    # When the input shape is (128, 128, 1), the Affine interface is successfully called
    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    affine_op = vision.Affine(degrees=4.2, translate=[-1, 1], scale=0.2, shear=(3.0, 3.0),
                              resample=Inter.NEAREST, fill_value=(180, 180, 200)).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=4.2, translate=[-1, 1], scale=0.2, shear=(3.0, 3.0),
                                  resample=Inter.NEAREST, fill_value=(180, 180, 200))(image)
    assert (affine_op == affine_op_cpu).all()

    # When the input shape has one dimension, the Affine interface is successfully called
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    affine_op = vision.Affine(degrees=0.6, translate=(-1, -1), scale=16777215.1, shear=[0.8, 4.1],
                              resample=Inter.BILINEAR, fill_value=(0, 255, 230)).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=0.6, translate=(-1, -1), scale=16777215.1, shear=[0.8, 4.1],
                                  resample=Inter.BILINEAR, fill_value=(0, 255, 230))(new_arr)
    assert (affine_op == affine_op_cpu).all()
    assert affine_op.shape == (128, 128, 3)

    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128))
    affine_op = vision.Affine(degrees=0.6, translate=(-1, -1), scale=16777215.1, shear=[0.8, 4.1],
                              resample=Inter.BILINEAR, fill_value=(0, 255, 230)).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=0.6, translate=(-1, -1), scale=16777215.1, shear=[0.8, 4.1],
                                  resample=Inter.BILINEAR, fill_value=(0, 255, 230))(new_arr)
    assert (affine_op == affine_op_cpu).all()
    assert affine_op.shape == (128, 128)

    # Pipeline, combined enhancement, the Affine interface is successfully called
    dataset = ds.ImageFolderDataset(DATA_DIR, 1)
    transforms1 = [
        vision.Decode(),
        vision.Affine(10, [0, 1], 20.1, (1.0, 5.0), Inter.NEAREST, 0).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    dataset = dataset.map(input_columns=["image"], operations=transform1)

    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=16.0)
def test_dvpp_affine_operation_04():
    """
    Feature: Affine operation on device
    Description: Testing the normal functionality of the Affine operator on device
    Expectation: The Output is equal to the expected output
    """
    # When the degrees parameter value is negative, the Affine interface is successfully called
    image = cv2.imread(image_bmp)
    affine_op_cpu = vision.Affine(degrees=-50, translate=(-1, 1), scale=3.01, shear=[0.8, 4.1])(image)
    affine_op = vision.Affine(degrees=-50, translate=(-1, 1), scale=3.01, shear=[0.8, 4.1]).device(
        device_target="Ascend")(image)
    assert (affine_op == affine_op_cpu).all()

    # When the degrees parameter is int, the Affine interface is successfully called
    image = cv2.imread(image_jpg)
    affine_op = vision.Affine(degrees=10, translate=(0, 1), scale=0.9, shear=(0.8, 4.1)).device(
        device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=10, translate=(0, 1), scale=0.9, shear=(0.8, 4.1))(image)
    assert (affine_op == affine_op_cpu).all()

    # When tx_min in the translate parameter is 0.8 and tx_max value is -1, the Affine interface is successfully called
    degrees = 15.1
    translate = [0.8, -1]
    scale = 0.9
    shear = (0.9, 1.1)
    dataset1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=True)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
        device_target="Ascend")
    dataset2 = dataset2.map(input_columns=["image"], operations=affine_op)
    dataset1 = dataset1.map(input_columns=["image"], operations=affine_op_cpu)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert (data1["image"] == data2["image"]).all()

    # When the scale parameter is int, the Affine interface is successfully called
    image = cv2.imread(image_jpg)
    degrees = 100
    translate = [-1, 1]
    scale = 180
    shear = (11.1, 0.11)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
        device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)
    assert np.allclose(affine_op, affine_op_cpu, rtol=1)

    # When the shear parameter is less than -180, the Affine interface is successfully called
    image = cv2.imread(image_jpg)
    degrees = 100.20
    translate = (-1, 0)
    scale = 11.10
    shear = -180
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
        device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)
    assert np.allclose(affine_op, affine_op_cpu, rtol=1)

    # When the shear parameter is greater than 180 or less than -180, the Affine interface call fails
    image = np.random.randn(192, 263, 3).astype(np.float32)
    degrees = 180
    translate = (-1, 0)
    scale = 11.10
    shear = 180.123
    with pytest.raises(ValueError, match="Input shear is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    shear = -180.123
    with pytest.raises(ValueError, match="Input shear is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the shear parameter is a tuple, the Affine interface is successfully called
    image = cv2.imread(image_jpg)
    degrees = 10
    translate = (-1, 1)
    scale = 1.1
    shear = (-1, 1)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
        device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)(image)
    assert (affine_op == affine_op_cpu).all()

    # When the input is a PIL image and fill_value is a tuple, the Affine interface is successfully called
    image = np.random.randn(192, 263, 3).astype(np.uint8)
    degrees = 128.5
    translate = (0, 1)
    scale = 0.9
    shear = [10.0, 100.0]
    fill_value = (100, 200, 220)
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                              fill_value=fill_value).device(device_target="Ascend")(image)
    affine_op_cpu = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                  fill_value=fill_value)(image)
    assert (affine_op_cpu == affine_op).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_affine_exception_01():
    """
    Feature: Affine operation on device
    Description: Testing the Affine Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # affine dvpp operator:Test input.shape is (50, 50, 8)
    image = np.random.randint(0, 255, (128, 128, 8)).astype(np.uint8)
    degrees = 10
    translate = (1, 1)
    scale = 2.2
    shear = 1.8
    with pytest.raises(RuntimeError, match=" The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3"):
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)
    image = np.random.randint(0, 255, (128, 128, 2)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match=" The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 2"):
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # test_func_dvpp_affine_error_dtype
    image = np.random.randint(0, 255, (128, 128, 8)).astype(np.float64)
    degrees = 10
    translate = (1, 1)
    scale = 2.2
    shear = 1.8
    with pytest.raises(RuntimeError) as error_log:
        _ = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)
        assert "The input data is not uint8 or float32" in str(error_log.value)

    # When the input image is Pillow, the Affine interface call fails
    with Image.open(image_jpg) as image:
        degrees = 0
        translate = (-0.15648, 1)
        scale = 100.2
        shear = 0.001
        resample = Inter.BILINEAR
        fill_value = 1
        affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear,
                                  resample=resample, fill_value=fill_value).device(device_target="Ascend")
        with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend, "
                                            "you can convert the input to the numpy ndarray type"):
            affine_op(image)

    # Unsupported inter method, the Affine interface call fails
    image = cv2.imread(image_jpg)
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR and NEAREST."):
        _ = vision.Affine(degrees=0, translate=(-0.15648, 1), scale=100.2, shear=0.001,
                          resample=Inter.ANTIALIAS, fill_value=1).device(device_target="Ascend")(image)
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR and NEAREST."):
        _ = vision.Affine(degrees=0, translate=(-0.15648, 1), scale=100.2, shear=0.001,
                          resample=Inter.CUBIC, fill_value=1).device(device_target="Ascend")(image)
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR and NEAREST."):
        _ = vision.Affine(degrees=0, translate=(-0.15648, 1), scale=100.2, shear=0.001,
                          resample=Inter.BICUBIC, fill_value=1).device(device_target="Ascend")(image)
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR and NEAREST."):
        _ = vision.Affine(degrees=0, translate=(-0.15648, 1), scale=100.2, shear=0.001,
                          resample=Inter.AREA, fill_value=1).device(device_target="Ascend")(image)
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR and NEAREST."):
        _ = vision.Affine(degrees=0, translate=(-0.15648, 1), scale=100.2, shear=0.001,
                          resample=Inter.PILCUBIC, fill_value=1).device(device_target="Ascend")(image)

    # test_func_dvpp_affine_error_device_type
    image = cv2.imread(image_jpg)
    with pytest.raises(ValueError, match="Input device_target is not within the valid set of \\['CPU', 'Ascend'\\]."):
        _ = vision.Affine(degrees=0, translate=(-0.15648, 1), scale=100.2, shear=0.001,
                          resample=Inter.BILINEAR, fill_value=1).device(device_target="GPU")(image)

    # When the degrees parameter value is string, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = "10"
    translate = (10, 1)
    scale = 1.1
    shear = 0.8
    with pytest.raises(TypeError, match=("Argument degrees with value 10 is not of type \\[<class 'int'>,"
                                         " <class 'float'>\\], but got <class 'str'>.")):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    image = cv2.imread(image_bmp)
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=-250, translate=(-1, 1), scale=3.01, shear=[0.8, 4.1]).device(
            device_target="Ascend")(image)

    image = cv2.imread(image_bmp)
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        _ = vision.Affine(degrees=250, translate=(-1, 1), scale=3.01, shear=[0.8, 4.1]).device(
            device_target="Ascend")(image)

    # When the degrees parameter is a tuple, the Affine interface call fails
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    degrees = (10, 20)
    translate = (1, 19)
    scale = 2.2
    shear = 1.8
    with pytest.raises(TypeError, match="Argument degrees with value \\(10, 20\\) is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'tuple'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_affine_exception_02():
    """
    Feature: Affine operation on device
    Description: Testing the Affine Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When the degrees parameter value is an array, the Affine interface call fails
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    degrees = [20]
    translate = (0, 1)
    scale = 1.8
    shear = (0.9, 1.1)
    with pytest.raises(TypeError, match="Argument degrees with value \\[20\\] is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'list'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the degrees parameter is not set, the Affine interface call fails
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.uint8)
    translate = (0, 1)
    scale = 0.9
    shear = (0.9, 1.1)
    with pytest.raises(TypeError, match="missing a required argument: 'degrees'"):
        vision.Affine(translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(image)

    # When the translate parameter value is greater than 1, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.0
    translate = (0, 240.12)
    scale = 0.9
    shear = (0.9, 1.1)
    with pytest.raises(ValueError, match="Input translate\\[1\\] is not within the required interval of"
                                         " \\[-1.0, 1.0\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the translate parameter value has only one, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.2
    translate = [0]
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="The length of translate should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the translate parameter value count is 4, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.2
    translate = [0, -1, -1, 1]
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="The length of translate should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the translate parameter value is string, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 100.0
    translate = ("0", "1")
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument translate\\[0\\] with value 0 is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the translate parameter value is bool, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 100.2
    translate = False
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument translate with value False is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'bool'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the translate parameter is not set, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 100.2
    scale = 10.9
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="missing a required argument: 'translate'"):
        vision.Affine(degrees=degrees, scale=scale, shear=shear).device(device_target="Ascend")(image)

    # When the scale parameter is a tuple, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 4.2
    translate = (0, 1)
    scale = (0.6, 0.9, 1.1)
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument scale with value \\(0.6, 0.9, 1.1\\) is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'tuple'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_affine_exception_03():
    """
    Feature: Affine operation on device
    Description: Testing the Affine Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When the scale parameter is string, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.32547
    translate = (-1, 0)
    scale = "0.9"
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument scale with value 0.9 is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the scale parameter is negative, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.15
    translate = [0, 1]
    scale = -0.1
    shear = (11.1, 11.11)
    with pytest.raises(ValueError, match="Input scale must be greater than 0"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the scale parameter is 0, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.15
    translate = [0, 1]
    scale = 0
    shear = (11.1, 11.11)
    with pytest.raises(ValueError, match="Input scale must be greater than 0"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the scale parameter is a list, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 15.32547
    translate = (-1, 0)
    scale = [11.1, 0.11]
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="Argument scale with value \\[11.1, 0.11\\] is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'list'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the scale parameter is not set, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 100.2
    translate = (-1, 0)
    shear = (11.1, 0.11)
    with pytest.raises(TypeError, match="missing a required argument: 'scale'"):
        vision.Affine(degrees=degrees, translate=translate, shear=shear).device(device_target="Ascend")(image)

    # When the shear parameter is not set, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 100.2
    translate = (-1, 0)
    scale = 11.1
    with pytest.raises(TypeError, match="missing a required argument: 'shear'"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale).device(device_target="Ascend")(image)

    # When the shear parameter is a 4-tuple, the Affine interface call fails
    image = np.random.randn(192, 263)
    degrees = 0
    translate = (1, 1)
    scale = 360
    shear = (1.0, 2.0, 4.0, 3.0)
    with pytest.raises(TypeError, match="The length of shear should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the shear parameter is a string type, the Affine interface call fails
    image = np.random.randn(192, 263, 1)
    degrees = 0
    translate = (0, 1)
    scale = 720
    shear = "10"
    with pytest.raises(TypeError, match=("Argument shear with value 10 is not of type \\[<class "
                                         "'numbers.Number'>, <class 'tuple'>, <class 'list'>\\]")):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the shear parameter is a list and the value count equals 1, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 10
    translate = (-1, 1)
    scale = 1.1
    shear = [5]
    with pytest.raises(TypeError, match="The length of shear should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the shear parameter value count equals 3, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 0
    translate = (0, 1)
    scale = 100
    shear = (1.0, 2.0, 3.0)
    with pytest.raises(TypeError, match="The length of shear should be 2."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_affine_exception_04():
    """
    Feature: Affine operation on device
    Description: Testing the Affine Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When the shear parameter value is 2 string types, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 180
    translate = [-1, 1]
    scale = 100
    shear = ("5", "10")
    with pytest.raises(TypeError, match="Argument shear\\[0\\] with value 5 is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the resample parameter is a string type, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 180
    translate = [-1, 1]
    scale = 100
    shear = [5.0, 10.1]
    resample = "Inter.NEAREST"
    with pytest.raises(TypeError,
                       match="Argument resample with value Inter.NEAREST is not of type \\[<enum 'Inter'>\\]"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample).device(
            device_target="Ascend")(image)

    # test_func_dvpp_affine_error_range，Affine_dvpp interface call fails
    image = np.random.randn(4, 3, 3).astype(np.float32)
    degrees = 100
    translate = [-1, 1]
    scale = 180
    shear = [5.0, 10.1]
    with pytest.raises(RuntimeError,
                       match="the input shape should be from \\[4, 6\\] to \\[32768, 32768\\], but got \\[4, 3\\]"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(
            device_target="Ascend")(image)

    # When the resample parameter is a list, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 100
    translate = [-1, 1]
    scale = 100
    shear = [5.0, 10.1]
    resample = [Inter.BILINEAR]
    with pytest.raises(TypeError, match="Argument resample with value \\[\\<Inter.BILINEAR: 2\\>\\] is not of type"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample).device(
            device_target="Ascend")(image)

    # When the resample parameter value is Inter, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 123
    translate = [0, 1]
    scale = 456
    shear = [15.0, 128.1]
    resample = Inter
    with pytest.raises(TypeError,
                       match="Argument resample with value <enum 'Inter'> is not of type \\[<enum 'Inter'>\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample).device(
            device_target="Ascend")(image)

    # When the resample parameter is int, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 123.1
    translate = [0, 1]
    scale = 456
    shear = [15.0, 128.1]
    resample = 10
    with pytest.raises(TypeError, match="Argument resample with value 10 is not of type"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample).device(
            device_target="Ascend")(image)

    # When the fill_value parameter value is negative, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 123.1
    translate = [0, 1]
    scale = 456.9
    shear = (15.0, 128.1)
    resample = Inter.NEAREST
    fill_value = -1
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                      fill_value=fill_value).device(device_target="Ascend")(image)

    # When the fill_value parameter value is empty, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 10
    translate = (0, 1)
    scale = 1.1
    shear = [-10.0, 10.0]
    fill_value = ()
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_affine_exception_05():
    """
    Feature: Affine operation on device
    Description: Testing the Affine Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When the fill_value parameter is a 2-tuple, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 10
    translate = (0, 1)
    scale = 1.1
    shear = [-10.0, 10.0]
    fill_value = (1, 2)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter is float, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 1
    translate = [0, 1]
    scale = 1.10
    shear = [10.0, -10.0]
    fill_value = 1.0
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter is string, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 0
    translate = (0, 1)
    scale = 0.9
    shear = [-10.0, 100.1]
    fill_value = "1"
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter is a 3-tuple string, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 0.123
    translate = (1, 0)
    scale = 101.1
    shear = [1.1, 12.12]
    fill_value = ("1", "2", 3)
    with pytest.raises(TypeError, match=r"Argument fill_value\[0\] with value 1 is not of type \[<class 'int'>\], "
                                        r"but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter value is greater than 255, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 120.123
    translate = (-1, 0)
    scale = 11.1
    shear = [-1.0, 10.1]
    fill_value = 256
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter is a 4-tuple, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 120.123
    translate = (-1, 0)
    scale = 101.1
    shear = [-1.12, 10.1]
    fill_value = (1, 2, 3, 4)
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter value is a tuple and greater than 255, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 120.123
    translate = (-1, 0)
    scale = 11.1
    shear = [-10.14, 10.1]
    fill_value = (1, 2, 256)
    with pytest.raises(ValueError, match=r"Input fill_value\[2\] is not within the required interval of \[0, 255\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter is numpy, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 12.12
    translate = (-1, 0)
    scale = 101.1
    shear = [-1.12, 10.1]
    fill_value = np.array([10, 20, 30])
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)

    # When the fill_value parameter is a list, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 0.1
    translate = (0, 1)
    scale = 0.9
    shear = [0.0, 100.0]
    fill_value = [1, 2, 3]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple"):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, fill_value=fill_value).device(
            device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_affine_exception_06():
    """
    Feature: Affine operation on device
    Description: Testing the Affine Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # When the input is a list, the Affine interface call fails
    image = np.random.randn(600, 450, 3).tolist()
    degrees = 0.1
    translate = (0, 1)
    scale = 0.9
    shear = [0.0, 100.0]
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        affine_op.device(device_target="Ascend")(image)

    # When the input is a tuple, the Affine interface call fails
    image = tuple(np.random.randn(600, 450, 3))
    degrees = 10.1
    translate = (0, 1)
    scale = 10.9
    shear = [1.0, 100.0]
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'tuple'>."):
        affine_op.device(device_target="Ascend")(image)

    # When the input image shape is 1-dimensional, the Affine interface call fails
    image = np.fromfile(image_jpg, dtype=np.uint8)
    degrees = 0.1
    translate = (0, 1)
    scale = 0.9
    shear = [10.0, 100.0]
    affine_op = vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear)
    with pytest.raises(RuntimeError, match="Exception thrown from dataset pipeline. "
                                           "Refer to 'Dataset Pipeline Error Message'."):
        affine_op(image).device(device_target="Ascend")(image)

    # When the degrees parameter is numpy, the Affine interface call fails
    degrees = np.array([4])
    translate = (0, 1)
    scale = 0.9
    shear = [10.0, 100.0]
    with pytest.raises(TypeError, match="Argument degrees with value \\[4\\] is not of type \\[<class 'int'>,"
                                        " <class 'float'>\\], but got <class 'numpy.ndarray'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")

    # When the translate parameter is None, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 0
    translate = None
    scale = 1
    shear = None
    resample = Inter.BILINEAR
    fill_value = 1
    with pytest.raises(TypeError, match="Argument translate with value None is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'NoneType'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                      fill_value=fill_value).device(device_target="Ascend")(image)

    # When the translate parameter is int, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 10.01
    translate = 1
    scale = 1.1
    shear = 50.0
    with pytest.raises(TypeError, match="Argument translate with value 1 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'int'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the translate parameter is string, the Affine interface call fails
    image = np.random.randn(192, 263, 3)
    degrees = 10.01
    translate = "1"
    scale = 1.1
    shear = 50.0
    with pytest.raises(TypeError, match="Argument translate with value 1 is not of type \\[<class 'list'>,"
                                        " <class 'tuple'>\\], but got <class 'str'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the degrees parameter values are all negative, the Affine interface call fails
    image = cv2.imread(image_bmp)
    degrees = -1000.1
    translate = (-1, 1)
    scale = 3.01
    shear = [0.8, 4.1]
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the degrees parameter value is greater than 16777216, the Affine interface call fails
    image = cv2.imread(image_bmp)
    degrees = 16777217
    translate = (-1, 1)
    scale = 3.01
    shear = [0.8, 4.1]
    with pytest.raises(ValueError, match="Input degrees is not within the required interval of \\[-180, 180\\]."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear).device(device_target="Ascend")(
            image)

    # When the shear parameter is None, the Affine interface call fails
    image = cv2.imread(image_jpg)
    degrees = 0
    translate = (-1, 0)
    scale = 1
    shear = None
    resample = Inter.BILINEAR
    fill_value = 1
    with pytest.raises(TypeError, match="Argument shear with value None is not of type \\[<class 'numbers.Number'>,"
                                        " <class 'tuple'>, <class 'list'>\\], but got <class 'NoneType'>."):
        vision.Affine(degrees=degrees, translate=translate, scale=scale, shear=shear, resample=resample,
                      fill_value=fill_value).device(device_target="Ascend").device(device_target="Ascend")(image)


if __name__ == '__main__':
    test_dvpp_affine_operation_01()
    test_dvpp_affine_operation_02()
    test_dvpp_affine_operation_03()
    test_dvpp_affine_operation_04()
    test_dvpp_affine_exception_01()
    test_dvpp_affine_exception_02()
    test_dvpp_affine_exception_03()
    test_dvpp_affine_exception_04()
    test_dvpp_affine_exception_05()
    test_dvpp_affine_exception_06()
