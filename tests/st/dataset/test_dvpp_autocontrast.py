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
import cv2
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as v_trans
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=62.0)
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=30.0)
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








@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=2.0)
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=0.01)
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






@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', mem_peak=2.0)
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
