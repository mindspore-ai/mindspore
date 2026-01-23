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
Testing DVPP AutoContrast operation
"""
import os
import numpy as np
import pytest
from PIL import Image
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
import mindspore.dataset.transforms.transforms as t_trans
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
        essential_mark='essential', mem_peak=62.0)
def test_dvpp_auto_contrast_operation_01():
    """
    Feature: AutoContrast operation on device
    Description: Testing the normal functionality of the AutoContrast operator on device
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast DVPP operator: Common scenario testing
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = [10, 20]
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)]
    auto_contrast_op = [
        v_trans.Decode(to_pil=False),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    ]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # Using AutoContrast operator in pyfunc
    ms.set_context(device_target="Ascend")

    # testcase : map with process mode
    dataset1 = ds.ImageFolderDataset(dataset_dir=dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dataset_dir=dir_data()[0], shuffle=False)

    def pyfunc1(img_bytes):
        img_decode = v_trans.Decode().device("Ascend")(img_bytes)
        img_ops = v_trans.AutoContrast(cutoff=35.55).device("Ascend")(img_decode)

        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_ops)
        return img_normalize

    def pyfunc2(img_bytes):
        img_decode = v_trans.Decode()(img_bytes)
        img_ops = v_trans.AutoContrast(cutoff=35.55)(img_decode)

        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = v_trans.Normalize(mean=mean_vec, std=std_vec)(img_ops)
        return img_normalize

    dataset1 = dataset1.map(pyfunc1, input_columns="image", python_multiprocessing=False)
    dataset2 = dataset2.map(pyfunc2, input_columns="image", python_multiprocessing=False)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], atol=1, rtol=1)

    # AutoContrast DVPP operator: Test cutoff is 0.0
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 0.0
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff)]
    auto_contrast_op = [v_trans.Decode(to_pil=False),
                        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test cutoff is 49.99
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 49.99
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff)]
    auto_contrast_op = [v_trans.Decode(to_pil=False),
                        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test cutoff is 10
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff)]
    auto_contrast_op = [v_trans.Decode(to_pil=False),
                        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is 10
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 10
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)]
    auto_contrast_op = [
        v_trans.Decode(to_pil=False),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    ]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=30.0)
def test_dvpp_auto_contrast_operation_02():
    """
    Feature: AutoContrast operation on device
    Description: Testing the normal functionality of the AutoContrast operator on device
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast DVPP operator: Test ignore is 0
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 0
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)]
    auto_contrast_op = [
        v_trans.Decode(to_pil=False),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    ]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is 255
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 255
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)]
    auto_contrast_op = [
        v_trans.Decode(to_pil=False),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    ]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is (10, 20)
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = (10, 20)
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)]
    auto_contrast_op = [
        v_trans.Decode(to_pil=False),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    ]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is [10, 20]
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = [10, 20]
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)]
    auto_contrast_op = [
        v_trans.Decode(to_pil=False),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    ]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: No parameters passed
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast()]
    auto_contrast_op = [v_trans.Decode(to_pil=False), v_trans.AutoContrast().device(device_target="Ascend")]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test first parameter
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    ignore = 0
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(ignore=ignore)]
    auto_contrast_op = [v_trans.Decode(to_pil=False),
                        v_trans.AutoContrast(ignore=ignore).device(device_target="Ascend")]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test second parameter
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 0.0
    auto_contrast = [v_trans.Decode(to_pil=False), v_trans.AutoContrast(cutoff=cutoff)]
    auto_contrast_op = [v_trans.Decode(to_pil=False),
                        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")]
    dataset1 = dataset1.map(input_columns=["image"], operations=auto_contrast)
    dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
    for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                            dataset2.create_dict_iterator(output_numpy=True)):
        assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=22.0)
def test_dvpp_auto_contrast_operation_03():
    """
    Feature: AutoContrast operation on device
    Description: Testing the normal functionality of the AutoContrast operator on device
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast DVPP operator: cutoff is 8.8, ignore is [0, 20, 15]
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "AutoContrast_021.png")
    auto_contrast = cv2.imread(image_png)
    image = cv2.imread(dir_data()[1])
    cutoff = 8.8
    ignore = [0, 20, 15]
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert (auto_contrast == auto_contrast_op).all()

    # AutoContrast DVPP operator: cutoff is 0.01, ignore is None
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "AutoContrast_022.png")
    auto_contrast = cv2.imread(image_png)
    image = cv2.imread(dir_data()[1])
    cutoff = 0.01
    ignore = None
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert (auto_contrast == auto_contrast_op).all()

    # AutoContrast DVPP operator: cutoff is 0, ignore is 6
    image = cv2.imread(dir_data()[1])
    cutoff = 0
    ignore = 6
    auto_contrast = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)(image)
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: cutoff is 49.9, ignore is (8, 255, 254, 0)
    image = cv2.imread(dir_data()[1])
    cutoff = 49.9
    ignore = (8, 255, 254, 0)
    auto_contrast = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)(image)
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: ignore is [11, 16]
    image = cv2.imread(dir_data()[1])
    ignore = [11, 16]
    auto_contrast = v_trans.AutoContrast(ignore=ignore)(image)
    auto_contrast_op = v_trans.AutoContrast(ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: cutoff is 35.65, ignore is 255
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "AutoContrast_026.jpg")
    auto_contrast = cv2.imread(image_jpg)
    image = np.random.randint(0, 255, (32, 32, 3)).astype(np.uint8)
    cutoff = 35.65
    ignore = 255
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=260)

    # AutoContrast DVPP operator: cutoff is 3, ignore is (0, 10, 18, 255, 253, 110, 120, 188)
    image = np.random.randint(0, 255, (64, 128)).astype(np.uint8)
    cutoff = 3
    ignore = (0, 10, 18, 255, 253, 110, 120, 188)
    auto_contrast = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)(image)
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is 10
    cutoff = 10.0
    ignore = 10
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)

    transforms1 = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True),
                            ds2.create_dict_iterator(output_numpy=True)):
        auto_contrast = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        auto_contrast_op = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_auto_contrast_operation_04():
    """
    Feature: AutoContrast operation on device
    Description: Testing the normal functionality of the AutoContrast operator on device
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast DVPP operator: Test ignore is 0
    cutoff = 10.0
    ignore = 0
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms1 = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True),
                            ds2.create_dict_iterator(output_numpy=True)):
        auto_contrast = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        auto_contrast_op = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is 255
    cutoff = 10.0
    ignore = 255
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms1 = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True),
                            ds2.create_dict_iterator(output_numpy=True)):
        auto_contrast = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        auto_contrast_op = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is (10, 20)
    cutoff = 10.0
    ignore = (10, 20)
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms1 = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True),
                            ds2.create_dict_iterator(output_numpy=True)):
        auto_contrast = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        auto_contrast_op = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=28.0)
def test_dvpp_auto_contrast_operation_05():
    """
    Feature: AutoContrast operation on device
    Description: Testing the normal functionality of the AutoContrast operator on device
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast DVPP operator: Test one parameter
    ignore = [10, 20]
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.AutoContrast(ignore=ignore),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms1 = [
        v_trans.Decode(),
        v_trans.AutoContrast(ignore=ignore).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        auto_contrast = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        auto_contrast_op = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test second parameter
    cutoff = 10.0
    ds1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff),
        v_trans.ToTensor()
    ]
    transform = t_trans.Compose(transforms)
    ds1 = ds1.map(input_columns=["image"], operations=transform)

    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    transforms1 = [
        v_trans.Decode(),
        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend"),
        v_trans.ToTensor()
    ]
    transform1 = t_trans.Compose(transforms1)
    ds2 = ds2.map(input_columns=["image"], operations=transform1)

    for data1, data2 in zip(ds1.create_dict_iterator(output_numpy=True), ds2.create_dict_iterator(output_numpy=True)):
        auto_contrast = (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        auto_contrast_op = (data2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test cutoff parameter is 18.6
    image_png = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "AutoContrast_043.png")
    auto_contrast = cv2.imread(image_png)
    image = cv2.imread(dir_data()[2])
    cutoff = 18.6
    ignore = [0, 20, 15]
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert (auto_contrast == auto_contrast_op).all()

    # AutoContrast DVPP operator: Test ignore parameter is [i for i in range(0, 256)]
    image = cv2.imread(dir_data()[3])
    cutoff = 45
    ignore = list(range(0, 256))
    auto_contrast = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)(image)
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore parameter is (100, 150, 200, 100, 255, 0)
    image = cv2.imread(dir_data()[2])
    cutoff = 49
    ignore = (100, 150, 200, 100, 255, 0)
    auto_contrast = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore)(image)
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test without passing cutoff parameter value
    image = cv2.imread(dir_data()[2])
    auto_contrast = v_trans.AutoContrast()(image)
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test input data is png image
    image = cv2.imread(dir_data()[4])
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(image)
    auto_contrast = v_trans.AutoContrast()(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test input data is jpg image
    image = cv2.imread(dir_data()[2])
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(image)
    auto_contrast = v_trans.AutoContrast()(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test input data is numpy
    image = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
    auto_contrast = v_trans.AutoContrast()(image)
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_auto_contrast_operation_06():
    """
    Feature: AutoContrast operation on device
    Description: Testing the normal functionality of the AutoContrast operator on device
    Expectation: The Output is equal to the expected output
    """
    # AutoContrast DVPP operator: Test input dimension contains 1
    image = np.random.randint(0, 255, (1, 128, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128, 3))
    auto_contrast = v_trans.AutoContrast()(new_arr)
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(new_arr)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)
    assert auto_contrast_op.shape == (128, 128, 3)

    image = np.random.randint(0, 255, (128, 128, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (128, 128))
    auto_contrast = v_trans.AutoContrast()(new_arr)
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(new_arr)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)
    assert auto_contrast_op.shape == (128, 128)

    # AutoContrast DVPP operator: Test input image channel number is 1 normal
    image = np.random.randint(-255, 255, (256, 128, 1)).astype(np.uint8)
    new_arr = np.reshape(image, (256, 128))
    auto_contrast = v_trans.AutoContrast()(new_arr)
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)

    # AutoContrast DVPP operator: Test 4-dimensional image processing normal
    image = np.random.randint(0, 255, (1, 256, 128, 3)).astype(np.uint8)
    new_arr = np.reshape(image, (256, 128, 3))
    auto_contrast = v_trans.AutoContrast()(new_arr)
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")(new_arr)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)
    assert auto_contrast_op.shape == (256, 128, 3)

    # AutoContrast DVPP operator: Test input type float32 normal
    image_jpg = os.path.join(TEST_DATA_DATASET_FUNC, "test_cv_image", "AutoContrast_072.jpg")
    image_jpg1 = cv2.imread(image_jpg)
    auto_contrast = image_jpg1.astype(np.float32)
    image = np.random.randint(0, 255, (256, 128, 3)).astype(np.float32)
    auto_contrast_op = v_trans.AutoContrast(1).device(device_target="Ascend")(image)
    assert np.allclose(auto_contrast, auto_contrast_op, rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_dvpp_auto_contrast_exception_01():
    """
    Feature: AutoContrast operation on device
    Description: Testing the AutoContrast Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # AutoContrast DVPP operator: Test cutoff is 100.1
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 100.1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test cutoff is -0.1
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = -0.1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test cutoff is [10.0]
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = [10.0]
    with pytest.raises(TypeError, match="Argument cutoff"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test cutoff is ''
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = ""
    with pytest.raises(TypeError, match="Argument cutoff"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is 256
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = 256
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is -1
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = -1
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: Test ignore is ''
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 1.5
    ignore = ""
    with pytest.raises(TypeError, match="Argument ignore"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=16.0)
def test_dvpp_auto_contrast_exception_02():
    """
    Feature: AutoContrast operation on device
    Description: Testing the AutoContrast Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # AutoContrast DVPP operator: Test passing extra parameters
    dataset1 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    dataset2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 0.0
    ignore = 0
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        auto_contrast_op = [
            v_trans.Decode(to_pil=False),
            v_trans.AutoContrast(cutoff, ignore, more_para).device(device_target="Ascend")
        ]
        dataset2 = dataset2.map(input_columns=["image"], operations=auto_contrast_op)
        for data1, data2 in zip(dataset1.create_dict_iterator(output_numpy=True),
                                dataset2.create_dict_iterator(output_numpy=True)):
            assert np.allclose(data1["image"], data2["image"], rtol=1, atol=1)

    # AutoContrast DVPP operator: cutoff is -0.1, ignore is [11, 16]
    image = cv2.imread(dir_data()[1])
    cutoff = -0.1
    ignore = [11, 16]
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of"):
        auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        auto_contrast_op(image)

    # AutoContrast DVPP operator: cutoff is 100.01, ignore is [16]
    image = cv2.imread(dir_data()[1])
    cutoff = 100.01
    ignore = [16]
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of"):
        auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        auto_contrast_op(image)

    # AutoContrast DVPP operator: cutoff is [10.0], ignore is [16]
    image = cv2.imread(dir_data()[1])
    cutoff = [10.0]
    ignore = [16]
    with pytest.raises(TypeError, match="is not of type"):
        auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        auto_contrast_op(image)

    # AutoContrast DVPP operator: cutoff is (8.2,), ignore is [16]
    image = cv2.imread(dir_data()[1])
    cutoff = (8.2,)
    ignore = [16]
    with pytest.raises(TypeError, match="is not of type"):
        auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        auto_contrast_op(image)

    # AutoContrast DVPP operator: cutoff is 8.2, ignore is [16.1,8]
    image = cv2.imread(dir_data()[1])
    cutoff = 8.2
    ignore = [16.1, 8]
    with pytest.raises(TypeError, match="Argument item with value 16.1 is not of type"):
        auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        auto_contrast_op(image)

    # AutoContrast DVPP operator: cutoff is 8.2, ignore is -1
    image = cv2.imread(dir_data()[1])
    cutoff = 8.2
    ignore = -1
    with pytest.raises(ValueError, match="Input ignore is not within the required interval of"):
        auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
        auto_contrast_op(image)

    # AutoContrast DVPP operator: cutoff is 20, ignore is (8, 255, 254, 0)
    image = cv2.imread(dir_data()[1])
    image = np.array(image).astype(np.float64)
    cutoff = 20
    ignore = (8, 255, 254, 0)
    auto_contrast_op = v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")
    with pytest.raises(RuntimeError):
        auto_contrast_op(image)

    # AutoContrast DVPP operator: Test ignore is 256
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    with pytest.raises(ValueError, match="Input ignore is not within the required interval"):
        ignore = 256
        transforms1 = [
            v_trans.Decode(),
            v_trans.AutoContrast(ignore=ignore).device(device_target="Ascend"),
            v_trans.ToTensor()
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for _ in ds2:
            pass


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_auto_contrast_exception_03():
    """
    Feature: AutoContrast operation on device
    Description: Testing the AutoContrast Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # AutoContrast DVPP operator: Test passing extra parameters
    ds2 = ds.ImageFolderDataset(dir_data()[0], shuffle=False)
    cutoff = 10.0
    ignore = [10, 20]
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        transforms1 = [
            v_trans.Decode(),
            v_trans.AutoContrast(cutoff, ignore, more_para),
            v_trans.ToTensor()
        ]
        transform1 = t_trans.Compose(transforms1)
        ds2 = ds2.map(input_columns=["image"], operations=transform1)
        for data1 in ds2.create_dict_iterator(output_numpy=True):
            (data1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

    # AutoContrast DVPP operator: Test ignore.len is 100000
    image = cv2.imread(dir_data()[2])
    cutoff = 38.653
    ignore = np.random.randint(0, 255, (100000,)).astype(np.uint8).tolist()
    with pytest.raises(RuntimeError,
                       match="DvppAutoContrast: the length of ignore should be less or equal to 256, but got: 100000"):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")(image)

    # AutoContrast DVPP operator: Test input data is list
    image = np.array(Image.open(dir_data()[4])).tolist()
    auto_contrast_op = v_trans.AutoContrast().device(device_target="Ascend")
    with pytest.raises(TypeError, match="Input should be NumPy or PIL image, got <class 'list'>."):
        auto_contrast_op(image)

    # AutoContrast DVPP operator: Test cutoff is -1
    cutoff = -1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of \\[0, 50\\)."):
        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test cutoff is -100.1
    cutoff = 100.1
    with pytest.raises(ValueError, match="Input cutoff is not within the required interval of \\[0, 50\\)."):
        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test cutoff is [20]
    cutoff = [20]
    with pytest.raises(TypeError,
                       match="Argument cutoff with value \\[20\\] is not of type \\[<class 'int'>, <class 'float'>\\]"):
        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test cutoff is True
    cutoff = True
    with pytest.raises(TypeError,
                       match="Argument cutoff with value True is not of type \\(<class 'int'>, <class 'float'>\\)"):
        v_trans.AutoContrast(cutoff=cutoff).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test ignore is {5, 6}
    cutoff = 10
    ignore = {5, 6}
    with pytest.raises(TypeError, match="Argument ignore with value {5, 6} is not of type \\[<class"
                                        " 'list'>, <class 'tuple'>, <class 'int'>\\]."):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test ignore is -1
    cutoff = 10
    ignore = -1
    with pytest.raises(ValueError, match="Input ignore is not within the required interval of \\[0, 255\\]."):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test ignore is 20.6
    cutoff = 10
    ignore = 20.6
    with pytest.raises(TypeError, match="Argument ignore with value 20.6 is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>, <class 'int'>\\]."):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test ignore is 256
    cutoff = 10
    ignore = 256
    with pytest.raises(ValueError, match="Input ignore is not within the required interval of \\[0, 255\\]."):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test ignore is np.array([10, 20])
    cutoff = 10
    ignore = np.array([10, 20])
    with pytest.raises(TypeError, match="Argument ignore with value \\[10 20\\] is not of "
                                        "type \\[<class 'list'>, <class 'tuple'>, <class 'int'>\\]."):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")

    # AutoContrast DVPP operator: Test ignore is True
    cutoff = 10
    ignore = True
    with pytest.raises(TypeError, match="Argument ignore with value True is not of "
                                        "type \\(<class 'list'>, <class 'tuple'>, <class 'int'>\\)."):
        v_trans.AutoContrast(cutoff=cutoff, ignore=ignore).device(device_target="Ascend")

    # AutoContrast DVPP operator: input 4 channel Numpy
    image = np.random.randn(30, 60, 4).astype(np.float32)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 4"):
        v_trans.AutoContrast().device(device_target="Ascend")(image)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2.0)
def test_dvpp_auto_contrast_exception_04():
    """
    Feature: AutoContrast operation on device
    Description: Testing the AutoContrast Operator in Exceptional Scenarios on device
    Expectation: Throw an exception
    """
    # AutoContrast DVPP operator: Test input image array is not uint8 or float32 error
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float64)
    with pytest.raises(RuntimeError, match="DvppAutoContrast: Error"):
        v_trans.AutoContrast().device(device_target="Ascend")(image)

    # AutoContrast DVPP operator: Test input image channel number is not 1 or 3 error
    image = np.random.randint(0, 255, (128, 128, 2)).astype(np.float32)
    with pytest.raises(RuntimeError, match="The channel of the input tensor of shape .* is not 1, 3, but got: 2"):
        v_trans.AutoContrast().device(device_target="Ascend")(image)

    # AutoContrast DVPP operator: Test device is not Ascend or CPU error
    image = np.random.randint(0, 255, (128, 128, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="Input device_target is not within the valid set of"):
        v_trans.AutoContrast().device(device_target="test")(image)

    # AutoContrast DVPP operator: Test 4-dimensional image N is not 1 error
    image = np.random.randint(0, 255, (2, 256, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor NHWC should be 1HWC or HWC"):
        v_trans.AutoContrast().device(device_target="Ascend")(image)

    # AutoContrast DVPP operator: Test data is 1-dimensional error
    image = np.random.randint(0, 255, (256,)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="DvppAutoContrast: the input tensor is not HW, HWC or 1HWC, but got: 1"):
        v_trans.AutoContrast().device(device_target="Ascend")(image)

    # AutoContrast DVPP operator: Test data is 5-dimensional error
    image = np.random.randint(0, 255, (1, 256, 128, 128, 3)).astype(np.uint8)
    with pytest.raises(RuntimeError, match="The input tensor is not of shape"):
        v_trans.AutoContrast().device(device_target="Ascend")(image)


if __name__ == '__main__':
    test_dvpp_auto_contrast_operation_01()
    test_dvpp_auto_contrast_operation_02()
    test_dvpp_auto_contrast_operation_03()
    test_dvpp_auto_contrast_operation_04()
    test_dvpp_auto_contrast_operation_05()
    test_dvpp_auto_contrast_operation_06()
    test_dvpp_auto_contrast_exception_01()
    test_dvpp_auto_contrast_exception_02()
    test_dvpp_auto_contrast_exception_03()
    test_dvpp_auto_contrast_exception_04()
