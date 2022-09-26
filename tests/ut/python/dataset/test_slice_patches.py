# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Testing SlicePatches Python API
"""
import functools
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.vision.utils as mode

from mindspore import log as logger
from util import diff_mse, visualize_list

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_slice_patches_01(plot=False):
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on RGB image(100, 200) to 4 patches
    Expectation: Output is equal to the expected output
    """
    slice_to_patches([100, 200], 2, 2, True, plot=plot)


def test_slice_patches_02(plot=False):
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on RGB image(100, 200) to 1 patch (no operation being applied)
    Expectation: Output is equal to the expected output
    """
    slice_to_patches([100, 200], 1, 1, True, plot=plot)


def test_slice_patches_03(plot=False):
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on RGB image(99, 199) to 4 patches in pad mode
    Expectation: Output is equal to the expected output
    """
    slice_to_patches([99, 199], 2, 2, True, plot=plot)


def test_slice_patches_04(plot=False):
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on RGB image(99, 199) to 4 patches in drop mode
    Expectation: Output is equal to the expected output
    """
    slice_to_patches([99, 199], 2, 2, False, plot=plot)


def test_slice_patches_05(plot=False):
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on RGB image(99, 199) to 4 patches in pad mode with fill_value=255
    Expectation: Output is equal to the expected output
    """
    slice_to_patches([99, 199], 2, 2, True, 255, plot=plot)


def slice_to_patches(ori_size, num_h, num_w, pad_or_drop, fill_value=0, plot=False):
    """
    Tool function for slice patches
    """
    logger.info("test_slice_patches_pipeline")

    cols = ['img' + str(x) for x in range(num_h*num_w)]
    # First dataset
    dataset1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize(ori_size)  # H, W
    slice_patches_op = vision.SlicePatches(
        num_h, num_w, mode.SliceMode.PAD, fill_value)
    if not pad_or_drop:
        slice_patches_op = vision.SlicePatches(
            num_h, num_w, mode.SliceMode.DROP)
    dataset1 = dataset1.map(operations=decode_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=resize_op, input_columns=["image"])
    dataset1 = dataset1.map(operations=slice_patches_op,
                            input_columns=["image"], output_columns=cols)
    # Second dataset
    dataset2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    dataset2 = dataset2.map(operations=decode_op, input_columns=["image"])
    dataset2 = dataset2.map(operations=resize_op, input_columns=["image"])
    func_slice_patches = functools.partial(
        slice_patches, num_h=num_h, num_w=num_w, pad_or_drop=pad_or_drop, fill_value=fill_value)
    dataset2 = dataset2.map(operations=func_slice_patches,
                            input_columns=["image"], output_columns=cols)

    num_iter = 0
    patches_c = []
    patches_py = []
    for data1, data2 in zip(dataset1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            dataset2.create_dict_iterator(num_epochs=1, output_numpy=True)):

        for x in range(num_h*num_w):
            col = "img" + str(x)
            mse = diff_mse(data1[col], data2[col])
            logger.info("slice_patches_{}, mse: {}".format(num_iter + 1, mse))
            assert mse == 0
            patches_c.append(data1[col])
            patches_py.append(data2[col])
        num_iter += 1
    if plot:
        visualize_list(patches_py, patches_c)


def test_slice_patches_exception_01():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op with invalid parameters
    Expectation: Correct error is raised as expected
    """
    logger.info("test_Slice_Patches_exception")
    try:
        _ = vision.SlicePatches(0, 2)
    except ValueError as e:
        logger.info("Got an exception in SlicePatches: {}".format(str(e)))
        assert "Input num_height is not within" in str(e)

    try:
        _ = vision.SlicePatches(2, 0)
    except ValueError as e:
        logger.info("Got an exception in SlicePatches: {}".format(str(e)))
        assert "Input num_width is not within" in str(e)

    try:
        _ = vision.SlicePatches(2, 2, 1)
    except TypeError as e:
        logger.info("Got an exception in SlicePatches: {}".format(str(e)))
        assert "Argument slice_mode with value" in str(e)

    try:
        _ = vision.SlicePatches(2, 2, mode.SliceMode.PAD, -1)
    except ValueError as e:
        logger.info("Got an exception in SlicePatches: {}".format(str(e)))
        assert "Input fill_value is not within" in str(e)

def test_slice_patches_06():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on random RGB image(158, 126, 1) to 16 patches
    Expectation: Output's shape is equal to the expected output's shape
    """
    image = np.random.randint(0, 255, (158, 126, 1)).astype(np.int32)
    slice_patches_op = vision.SlicePatches(2, 8)
    patches = slice_patches_op(image)
    assert len(patches) == 16
    assert patches[0].shape == (79, 16, 1)

def test_slice_patches_07():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on random RGB image(158, 126) to 16 patches
    Expectation: Output's shape is equal to the expected output's shape
    """
    image = np.random.randint(0, 255, (158, 126)).astype(np.int32)
    slice_patches_op = vision.SlicePatches(2, 8)
    patches = slice_patches_op(image)
    assert len(patches) == 16
    assert patches[0].shape == (79, 16)

def test_slice_patches_08():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on random RGB image(1, 56, 82, 256) to 4 patches
    Expectation: Output's shape is equal to the expected output's shape
    """
    np_data = np.random.randint(0, 255, (1, 56, 82, 256)).astype(np.uint8)
    dataset = ds.NumpySlicesDataset(np_data, column_names=["image"])
    slice_patches_op = vision.SlicePatches(2, 2)
    dataset = dataset.map(input_columns=["image"], output_columns=["img0", "img1", "img2", "img3"],
                          operations=slice_patches_op)
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        patch_shape = item['img0'].shape
        assert patch_shape == (28, 41, 256)

def test_slice_patches_09():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on random RGB image(56, 82, 256) to 12 patches with pad mode
    Expectation: Output's shape is equal to the expected output's shape
    """
    image = np.random.randint(0, 255, (56, 82, 256)).astype(np.uint8)
    slice_patches_op = vision.SlicePatches(4, 3, mode.SliceMode.PAD)
    patches = slice_patches_op(image)
    assert len(patches) == 12
    assert patches[0].shape == (14, 28, 256)

def skip_test_slice_patches_10():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on random RGB image(7000, 7000, 255) to 130 patches with drop mode
    Expectation: Output's shape is equal to the expected output's shape
    """
    image = np.random.randint(0, 255, (7000, 7000, 255)).astype(np.uint8)
    slice_patches_op = vision.SlicePatches(10, 13, mode.SliceMode.DROP)
    patches = slice_patches_op(image)
    assert patches[0].shape == (700, 538, 255)

def skip_test_slice_patches_11():
    """
    Feature: SlicePatches op
    Description: Test SlicePatches op on random RGB image(1, 7000, 7000, 256) to 130 patches with drop mode
    Expectation: Output's shape is equal to the expected output's shape
    """
    np_data = np.random.randint(0, 255, (1, 7000, 7000, 256)).astype(np.uint8)
    dataset = ds.NumpySlicesDataset(np_data, column_names=["image"])
    slice_patches_op = vision.SlicePatches(10, 13, mode.SliceMode.DROP)
    cols = ['img' + str(x) for x in range(10*13)]
    dataset = dataset.map(input_columns=["image"], output_columns=cols,
                          operations=slice_patches_op)
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        patch_shape = item['img0'].shape
        assert patch_shape == (700, 538, 256)

def slice_patches(image, num_h, num_w, pad_or_drop, fill_value):
    """ help function which slice patches with numpy """
    if num_h == 1 and num_w == 1:
        return image
    # (H, W, C)
    H, W, C = image.shape
    patch_h = H // num_h
    patch_w = W // num_w
    if H % num_h != 0:
        if pad_or_drop:
            patch_h += 1
    if W % num_w != 0:
        if pad_or_drop:
            patch_w += 1
    img = image[:, :, :]
    if pad_or_drop:
        img = np.full([patch_h*num_h, patch_w*num_w, C], fill_value, dtype=np.uint8)
        img[:H, :W] = image[:, :, :]
    patches = []
    for top in range(num_h):
        for left in range(num_w):
            patches.append(img[top*patch_h:(top+1)*patch_h,
                               left*patch_w:(left+1)*patch_w, :])

    return (*patches,)


if __name__ == "__main__":
    test_slice_patches_01(plot=True)
    test_slice_patches_02(plot=True)
    test_slice_patches_03(plot=True)
    test_slice_patches_04(plot=True)
    test_slice_patches_05(plot=True)
    test_slice_patches_06()
    test_slice_patches_07()
    test_slice_patches_08()
    test_slice_patches_09()
    test_slice_patches_exception_01()
