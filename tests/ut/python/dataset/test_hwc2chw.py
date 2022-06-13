# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Testing HWC2CHW op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.transforms as data_trans
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import diff_mse, visualize_list, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_hwc2chw_callable():
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW op is callable.
    Expectation: Valid input succeeds. Invalid input fails.
    """
    logger.info("Test HWC2CHW callable")

    # test one tensor
    img = np.zeros([50, 50, 3])
    assert img.shape == (50, 50, 3)
    img1 = vision.HWC2CHW()(img)
    assert img1.shape == (3, 50, 50)

    # test one tensor with 5 channels
    img2 = np.zeros([50, 50, 5])
    assert img2.shape == (50, 50, 5)
    img3 = vision.HWC2CHW()(img2)
    assert img3.shape == (5, 50, 50)

    # test 2 dim tensor
    img4 = np.zeros([32, 28])
    assert img4.shape == (32, 28)
    img5 = vision.HWC2CHW()(img4)
    assert img5.shape == (32, 28)

    # test input multiple tensors
    with pytest.raises(RuntimeError) as info:
        imgs = [img, img]
        _ = vision.HWC2CHW()(*imgs)
    assert "The op is OneToOne, can only accept one tensor as input" in str(info.value)

    with pytest.raises(RuntimeError) as info:
        _ = vision.HWC2CHW()(img, img)
    assert "The op is OneToOne, can only accept one tensor as input" in str(info.value)


def test_hwc2chw_multi_channels():
    """
    Feature: Test HWC2CHW feature
    Description: The input is a HWC format array with 5 channels
    Expectation: Success
    """
    logger.info("Test HWC2CHW with data of 5 channels")

    # create numpy array in HWC format with shape (4, 2, 5) like a fake image with 5 channels
    raw_data = np.random.rand(4, 2, 5).astype(np.float32)
    expect_output = np.transpose(raw_data, (2, 0, 1))

    # NumpySliceDataset support accept data stored in list, tuple etc, here only one row data in list.
    input_data = np.array([raw_data])
    dataset = ds.NumpySlicesDataset(input_data, column_names=["col1"], shuffle=False)

    hwc2chw = vision.HWC2CHW()
    dataset = dataset.map(hwc2chw, input_columns=["col1"])
    for item in dataset.create_tuple_iterator(output_numpy=True):
        assert np.allclose(item[0], expect_output)


def test_hwc2chw(plot=False):
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW op in pipeline
    Expectation: Pipelines succeed with comparison mse=0
    """
    logger.info("Test HWC2CHW")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    hwc2chw_op = vision.HWC2CHW()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=hwc2chw_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=decode_op, input_columns=["image"])

    image_transposed = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        transposed_item = item1["image"].copy()
        original_item = item2["image"].copy()
        image_transposed.append(transposed_item.transpose(1, 2, 0))
        image.append(original_item)

        # check if the shape of data is transposed correctly
        # transpose the original image from shape (H,W,C) to (C,H,W)
        mse = diff_mse(transposed_item, original_item.transpose(2, 0, 1))
        assert mse == 0
    if plot:
        visualize_list(image, image_transposed)


def test_hwc2chw_md5():
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW op with md5 check.
    Expectation: Pipeline results match in md5 comparison
    """
    logger.info("Test HWC2CHW with md5 comparison")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    hwc2chw_op = vision.HWC2CHW()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=hwc2chw_op, input_columns=["image"])

    # Compare with expected md5 from images
    filename = "HWC2CHW_01_result.npz"
    save_and_check_md5(data1, filename, generate_golden=GENERATE_GOLDEN)


def test_hwc2chw_comp(plot=False):
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW between Python (using ToTensor) and Cpp image augmentation
    Expectation: Image augmentations should be almost the same with mse < 0.001
    """
    logger.info("Test HWC2CHW with C and Python ToTensor image augmentation comparison")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    hwc2chw_op = vision.HWC2CHW()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=hwc2chw_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(True),
        vision.ToTensor()]
    transform = data_trans.Compose(transforms)
    data2 = data2.map(operations=transform, input_columns=["image"])

    image_c_transposed = []
    image_py_transposed = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        c_image = item1["image"]
        py_image = (item2["image"] * 255).astype(np.uint8)

        # Compare images between that applying C implementation and Python implementation image augmentations
        mse = diff_mse(py_image, c_image)
        # Note: The images aren't exactly the same due to rounding error
        assert mse < 0.001
    if plot:
        image_c_transposed.append(c_image.transpose(1, 2, 0))
        image_py_transposed.append(py_image.transpose(1, 2, 0))
        visualize_list(image_c_transposed, image_py_transposed, visualize_mode=2)


def test_hwc2chw_comparison2(plot=False):
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW between Python and C image augmentation
    Expectation: Pipelines succeed with comparison mse=0
    """
    logger.info("Test HWC2CHW with c_transform and py_transform comparison")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    operations_list = [vision.Decode(to_pil=True),
                       vision.ToPIL(),
                       np.array,
                       vision.HWC2CHW()]
    data1 = data1.map(operations=operations_list, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[vision.Decode(to_pil=True)], input_columns=["image"])

    image_transposed = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        transposed_item = item1["image"].copy()
        original_item = item2["image"].copy()
        image_transposed.append(transposed_item.transpose(1, 2, 0))
        image.append(original_item)

        # check if the shape of data is transposed correctly
        # transpose the original image from shape (H,W,C) to (C,H,W)
        mse = diff_mse(transposed_item, original_item.transpose(2, 0, 1))
        assert mse == 0
    if plot:
        visualize_list(image, image_transposed)


def test_hwc2chw_mix(plot=False):
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW C++ implementation in pipeline with mix of prior Python implementation ops
        (and no Compose op)
    Expectation: Pipelines succeed with comparison mse=0
    """
    logger.info("Test HWC2CHW mix")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    operations_list = [vision.Decode(to_pil=True),
                       vision.ToPIL(),
                       vision.HWC2CHW()]
    data1 = data1.map(operations=operations_list, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[vision.Decode(to_pil=True)], input_columns=["image"])

    image_transposed = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        transposed_item = item1["image"].copy()
        original_item = item2["image"].copy()
        image_transposed.append(transposed_item.transpose(1, 2, 0))
        image.append(original_item)

        # check if the shape of data is transposed correctly
        # transpose the original image from shape (H,W,C) to (C,H,W)
        mse = diff_mse(transposed_item, original_item.transpose(2, 0, 1))
        assert mse == 0
    if plot:
        visualize_list(image, image_transposed)


def test_hwc2chw_mix_compose(plot=False):
    """
    Feature: HWC2CHW op
    Description: Test HWC2CHW C++ implementation in pipeline with mix of prior Python implementation ops,
        and with Compose op
    Expectation: Pipelines succeed with comparison mse=0
    """
    logger.info("Test HWC2CHW mix")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    operations_list = [vision.Decode(to_pil=True),
                       vision.ToPIL(),
                       vision.HWC2CHW()]
    compose_op = data_trans.Compose(operations_list)
    data1 = data1.map(operations=compose_op, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=[vision.Decode(to_pil=True)], input_columns=["image"])

    image_transposed = []
    image = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        transposed_item = item1["image"].copy()
        original_item = item2["image"].copy()
        image_transposed.append(transposed_item.transpose(1, 2, 0))
        image.append(original_item)

        # check if the shape of data is transposed correctly
        # transpose the original image from shape (H,W,C) to (C,H,W)
        mse = diff_mse(transposed_item, original_item.transpose(2, 0, 1))
        assert mse == 0
    if plot:
        visualize_list(image, image_transposed)


if __name__ == '__main__':
    test_hwc2chw_callable()
    test_hwc2chw_multi_channels()
    test_hwc2chw(True)
    test_hwc2chw_md5()
    test_hwc2chw_comp(True)
    test_hwc2chw_comparison2(True)
    test_hwc2chw_mix(True)
    test_hwc2chw_mix_compose(True)
