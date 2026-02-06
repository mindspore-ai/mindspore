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
Testing DVPP Pad operation
"""
import os
import numpy as np
import pytest
import cv2
from PIL import Image
import mindspore as ms
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.transforms as trans
import mindspore.dataset as ds
from mindspore.dataset.vision import Border as v_Border
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


DATA_DIR = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
image_bmp = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=1020.0)
def test_dvpp_pad_operation_01():
    """
    Feature: Pad operation on device
    Description: Testing the normal functionality of the Pad operator on device
    Expectation: The Output is equal to the expected output
    """
    # Pad operator, normal test, padding=100, Border.CONSTANT, numpy image
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 100
    fill_value = 2
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(to_pil=False),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(to_pil=False),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)
    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 200 == py_image.shape[0]
        assert c_image.shape[1] + 200 == py_image.shape[1]
        break

    # Using Pad operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    padding = 1000
    fill_value = 2
    padding_mode = v_Border.SYMMETRIC

    def pyfunc1(img_bytes):
        img_decode = vision.Decode().device("Ascend")(img_bytes)
        img_ops = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device("Ascend")(
            img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = vision.Decode()(img_bytes)
        img_ops = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Pad operator, normal test, padding=1000, Border.SYMMETRIC, numpy image
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 1000
    fill_value = 2
    padding_mode = v_Border.SYMMETRIC
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 2000 == py_image.shape[0]
        assert c_image.shape[1] + 2000 == py_image.shape[1]
        break


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_pad_operation_02():
    """
    Feature: Pad operation on device
    Description: Testing the normal functionality of the Pad operator on device
    Expectation: The Output is equal to the expected output
    """
    # Pad operator, normal test, padding=0, Border.REFLECT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 0
    fill_value = 2
    padding_mode = v_Border.REFLECT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] == py_image.shape[0]
        assert c_image.shape[1] == py_image.shape[1]
        break

    # Pad operator, normal test, padding=(50, 0), Border.REFLECT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = (50, 0)
    fill_value = 2
    padding_mode = v_Border.REFLECT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] == py_image.shape[0]
        assert c_image.shape[1] + 50 * 2 == py_image.shape[1]
        break

    # Pad operator, normal test, padding=(0, 10), Border.CONSTANT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = (0, 10)
    fill_value = 2
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 10 * 2 == py_image.shape[0]
        assert c_image.shape[1] == py_image.shape[1]
        break


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_pad_operation_03():
    """
    Feature: Pad operation on device
    Description: Testing the normal functionality of the Pad operator on device
    Expectation: The Output is equal to the expected output
    """
    # Pad operator, normal test, padding=[0,10], Border.EDGE
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = [0, 10]
    fill_value = 2
    padding_mode = v_Border.EDGE
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 10 * 2 == py_image.shape[0]
        assert c_image.shape[1] == py_image.shape[1]
        break

    # Pad operator, normal test, padding=(0,10,20,30), Border.CONSTANT
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = (0, 10, 20, 30)
    fill_value = 2
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 40 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        break

    # Pad operator, normal test, padding=[50,10,20,30], Border.EDGE
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = [50, 10, 20, 30]
    fill_value = 2
    padding_mode = v_Border.EDGE
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 40 == py_image.shape[0]
        assert c_image.shape[1] + 70 == py_image.shape[1]
        break


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_pad_operation_04():
    """
    Feature: Pad operation on device
    Description: Testing the normal functionality of the Pad operator on device
    Expectation: The Output is equal to the expected output
    """
    # Pad operator, normal test, fill_value=100
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = 100
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (100, 100, 100)).all()
        break

    # Pad operator, normal test, fill_value=0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = 0
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (0, 0, 0)).all()
        break

    # Pad operator, normal test, fill_value=(5,10,20)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = (5, 10, 20)
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (5, 10, 20)).all()
        break


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=30.0)
def test_dvpp_pad_operation_05():
    """
    Feature: Pad operation on device
    Description: Testing the normal functionality of the Pad operator on device
    Expectation: The Output is equal to the expected output
    """
    # Pad operator, normal test, fill_value=(0,100,0)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = (0, 100, 0)
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (0, 100, 0)).all()
        break

    # Pad operator, normal test, fill_value=(0,0,200)
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    padding = 10
    fill_value = (0, 0, 200)
    padding_mode = v_Border.CONSTANT
    ### First dataset
    ds1 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms = [
        vision.Decode(),
        vision.ToTensor()
    ]
    transform = trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ### Second dataset
    ds2 = ds.ImageFolderDataset(data_dir, shuffle=False)
    transforms1 = [
        vision.Decode(),
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend"),
        vision.ToTensor()
    ]
    transform1 = trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        # with the seed value, we can only guarantee the first number generated
        c_image = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        py_image = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert c_image.shape[0] + 20 == py_image.shape[0]
        assert c_image.shape[1] + 20 == py_image.shape[1]
        assert (py_image[-1][-1][:] == (0, 0, 200)).all()
        break

    # Pad operator, normal test, eager mode, jpg cv image
    image = cv2.imread(image_jpg)
    padding = 100
    fill_value = 100
    padding_mode = v_Border.CONSTANT
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()

    # Pad operator, normal test, eager mode, gif cv image
    image = Image.open(image_gif)
    img_array = np.array(image)
    padding = [100, 200]
    fill_value = 0
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(img_array)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(img_array)
    assert (pad_op == pad_op_cpu).all()
    image.close()

    # Pad operator, normal test, eager mode, bmp PIL image
    image = cv2.imread(image_bmp)
    padding = (100, 200, 300, 400)
    fill_value = (120, 255, 120)
    padding_mode = v_Border.SYMMETRIC
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=34.0)
def test_dvpp_pad_operation_06():
    """
    Feature: Pad operation on device
    Description: Testing the normal functionality of the Pad operator on device
    Expectation: The Output is equal to the expected output
    """
    # Pad operator, normal test, eager mode, png PIL image
    image = cv2.imread(image_png)
    padding = [185, 100, 250, 440]
    fill_value = 250
    padding_mode = v_Border.REFLECT
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()
    assert np.array(pad_op).shape[0] == np.array(image).shape[0] + padding[1] + padding[3]
    assert np.array(pad_op).shape[1] == np.array(image).shape[1] + padding[0] + padding[2]

    # Pad operator, normal test, eager mode, padding = (1000, 2000)
    image = cv2.imread(image_jpg)
    padding = (1000, 2000)
    fill_value = 250
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()
    assert np.array(pad_op).shape[0] == np.array(image).shape[0] + 4000
    assert np.array(pad_op).shape[1] == np.array(image).shape[1] + 2000

    # Pad operator, normal test, eager mode, padding = (0, 1)
    image = cv2.imread(image_bmp)
    padding = (0, 1)
    fill_value = 250
    padding_mode = v_Border.REFLECT
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()
    assert np.array(pad_op).shape[0] == np.array(image).shape[0] + 2
    assert np.array(pad_op).shape[1] == np.array(image).shape[1]

    # Pad operator, normal test, eager mode, padding = [100, 200]
    image = cv2.imread(image_png)
    padding = [100, 200]
    fill_value = 250
    padding_mode = v_Border.SYMMETRIC
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()
    assert np.array(pad_op).shape[0] == np.array(image).shape[0] + 400
    assert np.array(pad_op).shape[1] == np.array(image).shape[1] + 200

    # Pad operator, normal test, eager mode, padding = 863
    image = cv2.imread(image_png)
    padding = 863
    fill_value = 250
    padding_mode = v_Border.SYMMETRIC
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()
    assert np.array(pad_op).shape[0] == np.array(image).shape[0] + 863 * 2
    assert np.array(pad_op).shape[1] == np.array(image).shape[1] + 863 * 2

    # Pad operator, normal test, eager mode, padding = (100, 128, 150, 100)
    image = np.random.randint(0, 255, (128, 256, 3)).astype(np.uint8)
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()
    assert np.array(pad_op).shape[0] == np.array(image).shape[0] + padding[1] + padding[3]
    assert np.array(pad_op).shape[1] == np.array(image).shape[1] + padding[0] + padding[2]

    # When input shape has one dimension, pad interface call successful
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(new_arr)
    assert (pad_op == pad_op_cpu).all()
    assert pad_op.shape == (356, 378, 3)

    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128))
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(new_arr)
    assert (pad_op == pad_op_cpu).all()
    assert pad_op.shape == (356, 378)

    # When input shape is two dimensions, pad interface call successful
    image = np.random.randint(0, 255, (128, 128)).astype(np.uint8)
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    pad_op = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
        device_target="Ascend")(image)
    pad_op_cpu = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode)(image)
    assert (pad_op == pad_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=3074.0)
def test_dvpp_pad_exception_01():
    """
    Feature: Pad operation on device
    Description: Testing the Pad Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Pad operator, exception test, fill_value=1000
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        vision.Pad(padding=10, fill_value=1000).device(device_target="Ascend")

    # Pad operator, test_func_dvpp_pad_error_pic_size
    image = np.random.randint(0, 255, (128, 256, 4)).astype(np.uint8)
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    with pytest.raises(RuntimeError,
                       match="The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 4"):
        _ = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
            device_target="Ascend")(image)

    image = np.random.randint(0, 255, (2, 128, 256, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="The input tensor NHWC should be 1HWC or HWC."):
        _ = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
            device_target="Ascend")(image)

    # Pad operator, test_func_dvpp_pad_small_pic_size
    image = np.random.randint(0, 255, (4, 4, 3)).astype(np.uint8)
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    with pytest.raises(RuntimeError,
                       match="DvppPadOp: the input shape should be from \\[4, 6\\] "
                             "to \\[32768, 32768\\], but got \\[4, 4\\]"):
        _ = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
            device_target="Ascend")(image)

    image = np.random.randint(0, 255, (32768, 32769, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError,
                       match="DvppPadOp: the input shape should be from \\[4, 6\\] "
                             "to \\[32768, 32768\\], but got \\[32768, 32769\\]"):
        _ = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
            device_target="Ascend")(image)

    # Pad operator, test_func_dvpp_pad_error_dtype
    image = np.random.randint(0, 255, (4, 4, 3)).astype(np.float64)
    padding = (100, 128, 150, 100)
    fill_value = 250
    padding_mode = v_Border.EDGE
    with pytest.raises(RuntimeError) as error_log:
        _ = vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(
            device_target="Ascend")(image)
        assert "The input data is not uint8 or float32" in str(error_log.value)

    # Pad operator, exception test, eager mode, numpy 4-channel image
    with Image.open(image_png) as image:
        image = np.array(image)
        padding = 100
        fill_value = 25
        pad_op = vision.Pad(padding=padding, fill_value=fill_value).device(device_target="Ascend")
        with pytest.raises(RuntimeError,
                           match="The channel of the input tensor of shape \\[H,W,C\\] is not 1, 3, but got: 4"):
            pad_op(image)

    # Pad operator, exception test, eager mode, input data is list
    with Image.open(image_png) as image:
        image = np.array(image).tolist()
        pad_op = vision.Pad(padding=10).device(device_target="Ascend")
        with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
            pad_op(image)

    # Pad operator, exception test, padding length is 1
    padding = [100]
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.Pad(padding=padding).device(device_target="Ascend")

    # Pad operator, exception test, padding type error numpy
    padding = np.array([10, 20])
    with pytest.raises(TypeError, match="Argument padding with value \\[10 20\\] is not of "
                                        "type \\[<class 'tuple'>, <class 'list'>, <class 'numbers.Number'>\\]."):
        vision.Pad(padding=padding).device(device_target="Ascend")

    # Pad operator, exception test, padding length is 5
    padding = (10, 20, 30, 40, 50)
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        vision.Pad(padding=padding).device(device_target="Ascend")

    # Pad operator, exception test, element value in padding is error -1
    padding = (-1, 20)
    with pytest.raises(ValueError,
                       match="Input pad_value is not within the required interval of \\[0, 2147483647\\]."):
        vision.Pad(padding=padding).device(device_target="Ascend")

    # Pad operator, exception test, element value in padding is error 2147483648
    padding = [10, 10, 20, 2147483648]
    with pytest.raises(ValueError,
                       match="Input pad_value is not within the required interval of \\[0, 2147483647\\]."):
        vision.Pad(padding=padding).device(device_target="Ascend")

    # Pad operator, exception test, padding type error float
    image = cv2.imread(image_jpg)
    padding = 100.0
    with pytest.raises(TypeError, match="Argument padding with value 100.0 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>"):
        pad_op = vision.Pad(padding=padding).device(device_target="Ascend")
        pad_op(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_pad_exception_02():
    """
    Feature: Pad operation on device
    Description: Testing the Pad Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Pad operator, exception test, no padding passed
    with pytest.raises(TypeError, match="missing a required argument: 'padding'"):
        vision.Pad().device(device_target="Ascend")

    # Pad operator, exception test, fill_value=(250,) length incorrect
    padding = 100
    fill_value = (250,)
    padding_mode = v_Border.SYMMETRIC
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, fill_value = (250, 120, 130, 2) length incorrect
    padding = 100
    fill_value = (250, 120, 130, 2)
    padding_mode = v_Border.CONSTANT
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, element value in fill_value is error 256
    padding = 100
    fill_value = (256, 120, 130)
    with pytest.raises(ValueError, match=r"Input fill_value\[0\] is not within the required interval of \[0, 255\]."):
        vision.Pad(padding=padding, fill_value=fill_value).device(device_target="Ascend")

    # Pad operator, exception test, element value in fill_value is error -1
    padding = 100
    fill_value = (25, -1, 130)
    padding_mode = v_Border.REFLECT
    with pytest.raises(ValueError, match=r"Input fill_value\[1\] is not within the required interval of \[0, 255\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, element type in fill_value is error float
    padding = 100
    fill_value = (25, 10.0, 130)
    padding_mode = v_Border.EDGE
    with pytest.raises(TypeError, match=r"Argument fill_value\[1\] with value 10.0 is not of type \[<class 'int'>\],"
                                        r" but got <class 'float'>."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, fill_value type error list
    padding = 100
    fill_value = [25, 10, 130]
    padding_mode = v_Border.SYMMETRIC
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, fill_value type error numpy
    padding = 100
    fill_value = np.array([25, 10, 130])
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        vision.Pad(padding=padding, fill_value=fill_value).device(device_target="Ascend")

    # Pad operator, exception test, padding_mode type error str
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = "v_Border.EDGE"
    with pytest.raises(TypeError,
                       match="Argument padding_mode with value v_Border.EDGE is not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, padding_mode type error Border
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = v_Border
    with pytest.raises(TypeError, match="Argument padding_mode with value <enum 'Border'> is "
                                        "not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, padding_mode type error int
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = 1
    with pytest.raises(TypeError, match="Argument padding_mode with value 1 is not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")

    # Pad operator, exception test, padding_mode type error list
    padding = 100
    fill_value = (25, 10, 130)
    padding_mode = [v_Border.EDGE]
    with pytest.raises(TypeError, match="Argument padding_mode with value \\[<Border.EDGE: 'edge'>\\] is "
                                        "not of type \\[<enum 'Border'>\\]."):
        vision.Pad(padding=padding, fill_value=fill_value, padding_mode=padding_mode).device(device_target="Ascend")


if __name__ == '__main__':
    test_dvpp_pad_operation_01()
    test_dvpp_pad_operation_02()
    test_dvpp_pad_operation_03()
    test_dvpp_pad_operation_04()
    test_dvpp_pad_operation_05()
    test_dvpp_pad_operation_06()
    test_dvpp_pad_exception_01()
    test_dvpp_pad_exception_02()
