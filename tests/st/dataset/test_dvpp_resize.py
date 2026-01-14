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
Testing DVPP Resize operation
"""
import os
import numpy as np
import pytest
from PIL import Image
import cv2
import mindspore as ms
from mindspore.common.tensor import Tensor
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
from mindspore.dataset.vision import Inter as v_Inter
from tests.mark_utils import arg_mark


PWD = os.path.dirname(__file__)
TEST_DATA_DATASET_FUNC = PWD + "/data"


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=40.0)
def test_dvpp_resize_operation_01():
    """
    Feature: Resize operation on device
    Description: Testing the normal functionality of the Resize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Using the Resize operator in pyfunc
    ms.set_context(device_target="Ascend")
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.Resize(size=(10, 500)).device("Ascend")(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.Resize(size=(10, 500))(img_decode)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize
    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"])

    # Resize operator:Test size is 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")

    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 6  # [6, 32768]
    resize_op_cpu = v_trans.Resize(size=size)
    resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
    dataset1 = dataset1.map(input_columns=["image"], operations=resize_op_cpu)
    dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image_aug == image).all()

    # Resize operator:Test size is a list sequence of length 2
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 520]
    resize_op_cpu = v_trans.Resize(size=size)
    resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
    dataset1 = dataset1.map(input_columns=["image"], operations=resize_op_cpu)
    dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image_aug == image).all()

    # Resize operator:Test size is a tuple sequence of length 2
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    resize_op_cpu = v_trans.Resize(size=size)
    resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
    dataset1 = dataset1.map(input_columns=["image"], operations=resize_op_cpu)
    dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image_aug == image).all()

    # Resize operator:Test interpolation is Inter.LINEAR and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.LINEAR
    decode = v_trans.Decode().device(device_target="Ascend")
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
    decode_cpu = v_trans.Decode()
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    dataset1 = dataset1.map(input_columns=["image"], operations=[decode_cpu, resize_op_cpu])

    for data1, data2 in zip(dataset.create_dict_iterator(output_numpy=True),
                            dataset1.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert np.allclose(image, image_aug)

    # Resize operator:Test interpolation is Inter.LINEAR and input is PIL data
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_file)
    size = (500, 520)
    interpolation = v_Inter.LINEAR
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")(image)
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)(image)
    assert (resize_op == resize_op_cpu).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=42.0)
def test_dvpp_resize_operation_02():
    """
    Feature: Resize operation on device
    Description: Testing the normal functionality of the Resize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test interpolation is Inter.NEAREST and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.NEAREST
    decode = v_trans.Decode().device(device_target="Ascend")
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
    decode_cpu = v_trans.Decode()
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    dataset1 = dataset1.map(input_columns=["image"], operations=[decode_cpu, resize_op_cpu])

    for data1, data2 in zip(dataset.create_dict_iterator(output_numpy=True),
                            dataset1.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image == image_aug).all()

    # Resize operator:Test interpolation is Inter.NEAREST and input is PIL data
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = cv2.imread(image_file)
    size = (500, 520)
    interpolation = v_Inter.NEAREST
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")(image)
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)(image)
    assert (resize_op == resize_op_cpu).all()

    # Resize operator:Test interpolation is Inter.BICUBIC and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.BICUBIC
    decode = v_trans.Decode().device(device_target="Ascend")
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
    decode_cpu = v_trans.Decode()
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    dataset1 = dataset1.map(input_columns=["image"], operations=[decode_cpu, resize_op_cpu])

    for data1, data2 in zip(dataset.create_dict_iterator(output_numpy=True),
                            dataset1.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert np.allclose(image, image_aug, rtol=1, atol=1)

    # Resize operator:Test interpolation is Inter.BICUBIC and input is PIL data
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_2.jpg")
    image = cv2.imread(image_file)
    size = (500, 520)
    interpolation = v_Inter.BICUBIC
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")(image)
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)(image)
    assert np.allclose(resize_op, resize_op_cpu, rtol=1, atol=1)

    # Resize operator:Test interpolation is Inter.CUBIC and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.CUBIC
    decode = v_trans.Decode().device(device_target="Ascend")
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
    decode_cpu = v_trans.Decode()
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    dataset1 = dataset1.map(input_columns=["image"], operations=[decode_cpu, resize_op_cpu])

    for data1, data2 in zip(dataset.create_dict_iterator(output_numpy=True),
                            dataset1.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert np.allclose(image, image_aug, rtol=1, atol=1)

    # Resize operator:Test interpolation is Inter.CUBIC and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.NEAREST
    decode = v_trans.Decode()
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    dataset1 = dataset1.map(input_columns=["image"], operations=[decode, resize_op_cpu])

    for data1, data2 in zip(dataset.create_dict_iterator(output_numpy=True),
                            dataset1.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image_aug == image).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
def test_dvpp_resize_operation_03():
    """
    Feature: Resize operation on device
    Description: Testing the normal functionality of the Resize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test interpolation is Inter.BILINEAR and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    dataset1 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.BILINEAR
    decode = v_trans.Decode()
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
    dataset1 = dataset1.map(input_columns=["image"], operations=[decode, resize_op_cpu])

    for data1, data2 in zip(dataset.create_dict_iterator(output_numpy=True),
                            dataset1.create_dict_iterator(output_numpy=True)):
        image = data1["image"]
        image_aug = data2["image"]
        assert (image_aug == image).all()

    # Resize operator:Test interpolation is Inter.BILINEAR and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    decode_op = v_trans.Decode(to_pil=False)
    resize_op = v_trans.Resize(size, interpolation=v_Inter.BILINEAR).device(device_target="Ascend")
    transforms_list = [decode_op, resize_op]
    dataset = dataset.map(input_columns=["image"], operations=transforms_list)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    decode_op = v_trans.Decode(to_pil=False).device(device_target="Ascend")
    resize_op = v_trans.Resize(size).device(device_target="Ascend")
    transforms_list = [decode_op, resize_op]
    dataset = dataset.map(input_columns=["image"], operations=transforms_list)
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Resize operator:Test input is 2d numpy array
    image = np.random.randn(1024, 1024).astype('float32')  # 不支持float64，只支持float32
    size = (500, 520)
    resize_op = v_trans.Resize(size, v_Inter.LINEAR).device(device_target="Ascend")
    resize_op_cpu = v_trans.Resize(size, v_Inter.LINEAR).device(device_target="Ascend")(image)
    out = resize_op(image)
    assert (resize_op_cpu == out).all()

    # Resize operator:Test input is jpg image
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "jpg.jpg")
    image = Image.open(image_file)
    with pytest.raises(TypeError,
                       match="The input PIL Image cannot be executed on Ascend, you can convert "
                             "the input to the numpy ndarray type."):
        size = (50, 60)
        interpolation = v_Inter.BILINEAR
        resize_op = v_trans.Resize(size, interpolation).device(device_target="Ascend")
        _ = resize_op(image)
        image.close()

    # Resize operator:Test input is bmp image
    image_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "bmp.bmp")
    image = cv2.imread(image_file3)
    size = (500, 520)
    interpolation = v_Inter.BILINEAR
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")(image)
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)(image)
    assert (resize_op == resize_op_cpu).all()

    # Resize operator:Test input is png image
    image_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "png.PNG")
    image = cv2.imread(image_file2)
    size = (50, 60)
    interpolation = v_Inter.BILINEAR
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")(image)
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)(image)
    assert (resize_op == resize_op_cpu).all()

    # Resize operator:Test input is gif image
    image_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    image = Image.open(image_file1)
    img_array = np.array(image)
    size = (10, 20)
    interpolation = v_Inter.LINEAR
    resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")(img_array)
    resize_op_cpu = v_trans.Resize(size=size, interpolation=interpolation)(img_array)
    assert np.allclose(resize_op, resize_op_cpu, rtol=1, atol=1)
    image.close()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_resize_operation_04():
    """
    Feature: Resize operation on device
    Description: Testing the normal functionality of the Resize operator on device
    Expectation: The Output is equal to the expected output
    """
    # Resize operator:Test input is image opened using the cv2 method
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_2.jpg")
    image = cv2.imread(image_file)
    size = 128
    resize_op = v_trans.Resize(size, v_Inter.BICUBIC).device(device_target="Ascend")
    out = resize_op(image)
    if np.array(image).shape[0] < np.array(image).shape[1]:
        assert out.shape[0] == 128
        assert out.shape[1] == (out.shape[1] / out.shape[0] * 128)
    else:
        assert out.shape[0] == (out.shape[0] / out.shape[1] * 128)
        assert out.shape[1] == 128

    # Resize operator:Test input is numpy list
    image = np.random.randn(256, 188, 1).tolist()
    size = (100, 100)
    resize_op = v_trans.Resize(size, v_Inter.BICUBIC).device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>"):
        out = resize_op(image)
        assert np.array(out).shape == (100, 100, 1)

    # Resize operator:Test eager interpolation_c is v_Inter.PILCUBIC
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_2.jpg")
    image = cv2.imread(image_file)
    size = (250, 300)
    interpolation_c = v_Inter.PILCUBIC
    interpolation_py = v_Inter.BICUBIC
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST."):
        resize_c_op = v_trans.Resize(size, interpolation=interpolation_c).device(device_target="Ascend")
        resize_py_op = v_trans.Resize(size, interpolation=interpolation_py).device(device_target="Ascend")
        out_c = resize_c_op(image)
        out_py = resize_py_op(image)
        assert (out_c == out_py).all()

    # Resize operator:Test eager interpolation_c is v_Inter.PILCUBIC
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_2.jpg")
    with Image.open(image_file) as image:
        size = (np.array(image).shape[0], np.array(image).shape[1])
    interpolation_c = v_Inter.BILINEAR
    interpolation_py = v_Inter.CUBIC
    resize_c_op = v_trans.Resize(size, interpolation=interpolation_c).device(device_target="Ascend")
    resize_py_op = v_trans.Resize(size, interpolation=interpolation_py).device(device_target="Ascend")
    with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend"):
        out_c = resize_c_op(image)
        out_py = resize_py_op(image)
        assert (out_c == np.array(out_py)).all()
        assert (np.array(image) == np.array(out_c)).all()

    # Resize operator:Test eager interpolation_c is v_Inter.PILCUBIC
    image_file = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train", "class1", "1_2.jpg")
    image = cv2.imread(image_file)
    size = [2500, 3000]
    interpolation_c = v_Inter.PILCUBIC
    interpolation_py = v_Inter.BICUBIC
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST."):
        _ = v_trans.Resize(size, interpolation=interpolation_c).device(device_target="Ascend")(image)
        _ = v_trans.Resize(size, interpolation=interpolation_py).device(device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=14.0)
def test_dvpp_resize_exception_01():
    """
    Feature: Resize operation on device
    Description: Testing the Resize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Resize operator:Test size is 0
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is 16777216
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 32769
    with pytest.raises(RuntimeError, match="the output shape should be from \\[4, 6\\] to \\[32768, 32768\\]"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is 16777217
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is float
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 500.5
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is None
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = None
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is str
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = 'test'
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is ""
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = ""
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_resize_exception_02():
    """
    Feature: Resize operation on device
    Description: Testing the Resize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Resize operator:Test size is a sequence of length 1
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence of length 3
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 500, 520]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence containing a float of 2 lengths
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500.5, 500]
    with pytest.raises(TypeError, match="Argument size at dim 0 with value 500.5 is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'float'>"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence containing a str of 2 lengths
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, 'test']
    with pytest.raises(TypeError, match="Argument size at dim 1 with value test is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test size is a sequence containing bool of 2 lengths
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = [500, True]
    with pytest.raises(TypeError, match="Argument size at dim 1 with value True is not of type " + \
                                        "\\(<class 'int'>,\\), but got <class 'bool'>"):
        resize_op = v_trans.Resize(size=size).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is Inter.PILCUBIC and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.PILCUBIC
    decode = v_trans.Decode()
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST."):
        resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
        dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is Inter.AREA and input is numpy data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    interpolation = v_Inter.AREA
    decode = v_trans.Decode()
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST."):
        resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
        dataset = dataset.map(input_columns=["image"], operations=[decode, resize_op])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_resize_exception_03():
    """
    Feature: Resize operation on device
    Description: Testing the Resize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Resize operator:Test interpolation is Inter.ANTIALIAS and input is PIL data
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir, shuffle=False, decode=False)
    size = (500, 520)
    with pytest.raises(RuntimeError, match="Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST."):
        decode_op = v_trans.Decode()
        resize_op = v_trans.Resize(size, interpolation=v_Inter.ANTIALIAS).device(device_target="Ascend")
        transforms_list = [decode_op, resize_op]
        dataset = dataset.map(input_columns=["image"], operations=transforms_list)
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is ""
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = ""
    with pytest.raises(TypeError, match="Argument interpolation with value \"\" is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>"):
        resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is str
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = "test"
    with pytest.raises(TypeError, match="Argument interpolation with value test is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>"):
        resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is None
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = None
    with pytest.raises(KeyError, match="Interpolation should not be None"):
        resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test interpolation is bool
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = True
    with pytest.raises(TypeError, match="Argument interpolation with value True is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'bool'>"):
        resize_op = v_trans.Resize(size=size, interpolation=interpolation).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test no parameters
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        resize_op = v_trans.Resize().device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=16.0)
def test_dvpp_resize_exception_04():
    """
    Feature: Resize operation on device
    Description: Testing the Resize Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # Resize operator:Test more parameters
    data_dir = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset2 = ds.ImageFolderDataset(data_dir, shuffle=False, decode=True)
    size = (500, 520)
    interpolation = v_Inter.LINEAR
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        resize_op = v_trans.Resize(size, interpolation, more_para).device(device_target="Ascend")
        dataset2 = dataset2.map(input_columns=["image"], operations=resize_op)
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # Resize operator:Test input is 3d numpy array
    image = np.random.randn(1024, 1024, 3).astype('float32')
    size = (500, 520)
    resize_op = v_trans.Resize(size, v_Inter.LINEAR).device(device_target="Ascend")
    resize_op_cpu = v_trans.Resize(size, v_Inter.LINEAR).device(device_target="Ascend")(image)
    out = resize_op(image)
    assert (resize_op_cpu == out).all()

    # Resize operator:Test input is a tensor
    image = Tensor(np.random.randn(10, 10, 3))
    size = (100, 100)
    resize_op = v_trans.Resize(size, v_Inter.BICUBIC).device(device_target="Ascend")
    with pytest.raises(TypeError,
                       match="Input should be NumPy or PIL image, got <class 'mindspore.common.tensor.Tensor'>"):
        resize_op(image)

    # Resize operator: Input a 4D NumPy array
    image = np.random.randn(56, 88, 3, 3)
    size = (100, 100)
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC."):
        resize_op = v_trans.Resize(size, v_Inter.BICUBIC).device(device_target="Ascend")
        resize_op(image)

    # Resize operator:Test size is tensor
    image = np.random.randn(256, 188, 1)
    size = Tensor([128, 128])
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        resize_op = v_trans.Resize(size, v_Inter.BICUBIC).device(device_target="Ascend")
        resize_op(image)

    # Resize operator:Test no image is transferred
    size = [128, 128]
    with pytest.raises(RuntimeError, match="Input Tensor is not valid"):
        resize_op = v_trans.Resize(size, v_Inter.BICUBIC).device(device_target="Ascend")
        resize_op()

    # Resize operator:Test Interpolation mode PILCUBIC and 1 channel numpy data
    image = np.random.randn(1024, 1024, 1).astype('float64')
    size = (100, 100)
    resize_c_op = v_trans.Resize(size, interpolation=v_Inter.BICUBIC).device(device_target="Ascend")
    with pytest.raises(RuntimeError, match="DvppResize: the type of the input is not uint8 or float."):
        resize_c_op(image)

    # Resize operator:Test eager image is gif
    image_gif = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "gif.gif")
    with Image.open(image_gif) as image:
        size = [50, 100]
        interpolation_c = v_Inter.BILINEAR
        resize_c_op = v_trans.Resize(size, interpolation=interpolation_c).device(device_target="Ascend")
        with pytest.raises(TypeError, match="The input PIL Image cannot be executed on Ascend"):
            resize_c_op(image)


if __name__ == '__main__':
    test_dvpp_resize_operation_01()
    test_dvpp_resize_operation_02()
    test_dvpp_resize_operation_03()
    test_dvpp_resize_operation_04()
    test_dvpp_resize_exception_01()
    test_dvpp_resize_exception_02()
    test_dvpp_resize_exception_03()
    test_dvpp_resize_exception_04()
