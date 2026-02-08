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
Testing RandomCropWithBBox op in DE
"""
import numpy as np
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
import mindspore.dataset.vision.utils as mode

from mindspore import log as logger
from util import config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5, \
    helper_perform_ops_bbox, helper_test_visual_bbox, helper_invalid_bounding_box_test

GENERATE_GOLDEN = False

# Updated VOC dataset with correct annotations - DATA_DIR
DATA_DIR_VOC = "../data/dataset/testVOC2012_2"
# COCO dataset - DATA_DIR, ANNOTATION_DIR
DATA_DIR_COCO = ["../data/dataset/testCOCO/train/",
                 "../data/dataset/testCOCO/annotations/train.json"]
TEST_DATA_DATASET_FUNC ="../data/dataset/"
DATA_DIR_IMAGE = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")


def test_random_crop_with_bbox_op_c(plot_vis=False):
    """
    Feature: RandomCropWithBBox op
    Description: Prints images and bboxes side by side with and without RandomCropWithBBox Op applied
    Expectation: Images and bboxes are printed side by side as expected
    """
    logger.info("test_random_crop_with_bbox_op_c")

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op UT
    test_op = vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_crop_with_bbox_op_coco_c(plot_vis=False):
    """
    Feature: RandomCropWithBBox op
    Description: Prints images and bboxes side by side with and without RandomCropWithBBox Op applied with CocoDataset
    Expectation: Images and bboxes are printed side by side as expected
    """
    logger.info("test_random_crop_with_bbox_op_coco_c")
    # load dataset
    data_coco1 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)


def test_random_crop_with_bbox_op2_c(plot_vis=False):
    """
    Feature: RandomCropWithBBox op
    Description: Prints images and bboxes side by side with and without RandomCropWithBBox Op applied with md5 check
    Expectation: Passes the md5 check test
    """
    logger.info("test_random_crop_with_bbox_op2_c")
    original_seed = config_get_set_seed(593447)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op unit - test
    test_op = vision.RandomCropWithBBox(
        512, [200, 200, 200, 200], fill_value=(255, 255, 255))

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "random_crop_with_bbox_01_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_crop_with_bbox_op3_c(plot_vis=False):
    """
    Feature: RandomCropWithBBox op
    Description: Prints images and bboxes side by side with and without RandomCropWithBBox Op applied with padding_mode
    Expectation: Images and bboxes are printed side by side as expected
    """
    logger.info("test_random_crop_with_bbox_op3_c")

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op unit - test
    test_op = vision.RandomCropWithBBox(
        512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_crop_with_bbox_op_edge_c(plot_vis=False):
    """
    Feature: RandomCropWithBBox op
    Description: Prints images and bboxes side by side with and without RandomCropWithBBox Op applied on edge case
    Expectation: Passes the dynamically generated edge case
    """
    logger.info("test_random_crop_with_bbox_op_edge_c")

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    # define test OP with values to match existing Op unit - test
    test_op = vision.RandomCropWithBBox(
        512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    # maps to convert data into valid edge case data
    data_voc1 = helper_perform_ops_bbox(data_voc1, None, True)

    # Test Op added to list of Operations here
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op, True)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_crop_with_bbox_op_invalid_c():
    """
    Feature: RandomCropWithBBox op
    Description: Test RandomCropWithBBox Op on invalid constructor parameters
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_with_bbox_op_invalid_c")

    # Load dataset
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        # define test OP with values to match existing Op unit - test
        test_op = vision.RandomCropWithBBox([512, 512, 375])

        # map to apply ops
        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

        for _ in data_voc2.create_dict_iterator(num_epochs=1):
            break
    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Size should be a single integer" in str(err)


def test_random_crop_with_bbox_op_bad_c():
    """
    Feature: RandomCropWithBBox op
    Description: Test RandomCropWithBBox Op with invalid bounding boxes
    Expectation: Multiple errors are caught as expected
    """
    logger.info("test_random_crop_with_bbox_op_bad_c")
    test_op = vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    helper_invalid_bounding_box_test(DATA_DIR_VOC, test_op)


def test_random_crop_with_bbox_op_bad_padding():
    """
    Feature: RandomCropWithBBox op
    Description: Test RandomCropWithBBox Op on invalid constructor parameters for padding
    Expectation: Error is raised as expected
    """
    logger.info("test_random_crop_with_bbox_op_invalid_c")

    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        test_op = vision.RandomCropWithBBox([512, 512], padding=-1)

        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

        for _ in data_voc2.create_dict_iterator(num_epochs=1):
            break
    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input padding is not within the required interval of [0, 2147483647]." in str(
            err)

    try:
        test_op = vision.RandomCropWithBBox(
            [512, 512], padding=[16777216, 16777216, 16777216, 16777216])

        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

        for _ in data_voc2.create_dict_iterator(num_epochs=1):
            break
    except RuntimeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "padding size is three times bigger than the image size" in str(
            err)


def test_random_crop_with_bbox_padded_dataset():
    """
    Feature: RandomCropWithBBox op
    Description: RandomCropWithBBox need to copy its input image and image bbox, otherwise numpy memory will be reused
    Expectation: Images and bboxes are transformed as expected
    """
    original_seed = ds.config.get_seed()
    ds.config.set_seed(1234)
    # load dataset
    dataset = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                             decode=True, shuffle=True, extra_metadata=True, num_samples=2)

    for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        image_data = data['image']
        bbox_data = data['bbox']
        break

    padded_samples = [{'image': image_data, 'bbox': bbox_data, 'category_id': np.zeros((2, 1), np.uint32),
                       'iscrowd': np.zeros((2, 1), np.uint32), '_meta-filename': np.array('0', dtype=str)}]

    padded_ds = ds.PaddedDataset(padded_samples)
    dataset = dataset + padded_ds
    dataset = dataset.repeat(5)

    randomcrop_op = vision.RandomCropWithBBox(size=(300, 300), padding=[200, 200, 200, 200],
                                              pad_if_needed=True, fill_value=(1, 1, 0), padding_mode=vision.Border.EDGE)
    dataset = dataset.map(input_columns=["image", "bbox"], operations=randomcrop_op, num_parallel_workers=1)
    dataset = dataset.map(input_columns=["image", "bbox"], operations=randomcrop_op, num_parallel_workers=1)

    for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        pass

    ds.config.set_seed(original_seed)


def test_random_crop_with_bbox_operation_01():
    """
    Feature: RandomCropWithBBox operation
    Description: Testing the normal functionality of the RandomCropWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # When parameter size is 1, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 1
    test_op = vision.RandomCropWithBBox(size=size)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for data2 in dataset2.create_dict_iterator(output_numpy=True):
        image_aug = data2["image"]
        assert image_aug.shape == (1, 1, 3)

    # When parameter size is 500, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 500
    test_op = vision.RandomCropWithBBox(size=size)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for data2 in dataset2.create_dict_iterator(output_numpy=True):
        image_aug = data2["image"]
        assert image_aug.shape == (500, 500, 3)

    # When parameter size is a list, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, 520]
    test_op = vision.RandomCropWithBBox(size=size)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter padding is [1, 1, 1, 1], RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 512
    padding = [1, 1, 1, 1]
    test_op = vision.RandomCropWithBBox(size=size, padding=padding)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter fill_value is 0, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = 0
    test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                        fill_value=fill_value)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass


def test_random_crop_with_bbox_operation_02():
    """
    Feature: RandomCropWithBBox operation
    Description: Testing the normal functionality of the RandomCropWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # When all parameters are set, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.SYMMETRIC
    test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                        fill_value=fill_value, padding_mode=padding_mode)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter padding_mode is Border.EDGE, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.EDGE
    test_op = vision.RandomCropWithBBox(size=size, pad_if_needed=pad_if_needed,
                                        fill_value=fill_value, padding_mode=padding_mode)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter padding_mode is Border.REFLECT, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 300
    padding = (100, 100, 100, 100)
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.REFLECT
    test_op = vision.RandomCropWithBBox(size=size, padding=padding, fill_value=fill_value, padding_mode=padding_mode)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameter padding_mode is Border.SYMMETRIC, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 180
    padding = (100, 100, 100, 100)
    pad_if_needed = False
    padding_mode = mode.Border.SYMMETRIC
    test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                        padding_mode=padding_mode)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # When parameters use default values, RandomCropWithBBox interface is successfully called
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 300
    test_op = vision.RandomCropWithBBox(size=size)
    dataset2 = dataset2.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset2 = dataset2.project(columns=["image", "bbox"])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass


def test_random_crop_with_bbox_exception_01():
    """
    Feature: RandomCropWithBBox operation
    Description: Testing the RandomCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Using ImageFolderDataset to get the dataset, RandomCropWithBBox interface call fails
    dataset = ds.ImageFolderDataset(DATA_DIR_IMAGE, decode=True, shuffle=False)
    test_op = vision.RandomCropWithBBox(512)
    dataset = dataset.map(input_columns=["image", "label"],
                          output_columns=["image", "label"],
                          operations=[test_op])
    dataset = dataset.project(columns=["image", "label"])
    with pytest.raises(RuntimeError,
                       match="BoundingBox: bounding boxes should have to be two-dimensional matrix at least."):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # When parameter size is 0, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        test_op = vision.RandomCropWithBBox(size=size)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter size is 1000, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 1000
    with pytest.raises(RuntimeError,
                       match=" RandomCrop: invalid crop size, crop size is bigger than the image dimensions."):
        test_op = vision.RandomCropWithBBox(size=size)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter size is float, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 500.5
    with pytest.raises(TypeError, match="Argument size with value 500.5 is not of type \\[<class 'int'>,"
                                        " <class 'list'>, <class 'tuple'>\\], but got <class 'float'>."):
        test_op = vision.RandomCropWithBBox(size=size)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter size is a 3-tuple, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 500, 520)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.RandomCropWithBBox(size=size)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter size is string, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = ""
    with pytest.raises(TypeError, match="Argument size with value .*, but got <class 'str'>."):
        test_op = vision.RandomCropWithBBox(size=size)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


def test_random_crop_with_bbox_exception_02():
    """
    Feature: RandomCropWithBBox operation
    Description: Testing the RandomCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter padding is negative, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 512
    padding = -1
    with pytest.raises(ValueError, match="Input padding is not within the required interval"):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter padding is 16777216, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 512
    padding = 16777216
    with pytest.raises(RuntimeError, match="RandomCrop: padding size is three times bigger than the image size, "
                                           "padding top: 16777216, padding bottom: 16777216, padding pad_left_: "
                                           "16777216, padding padding right:16777216"):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter padding length is 3, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 512
    padding = (1, 1, 1)
    with pytest.raises(ValueError, match="The size of the padding list or tuple should be 2 or 4."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter padding is float, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 512
    padding = 1.5
    with pytest.raises(TypeError, match="Argument padding with value 1.5 is not of type "
                                        "\\[<class 'int'>\\], but got <class 'float'>."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter padding is string, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 512
    padding = ""
    with pytest.raises(TypeError, match="Argument padding with value .*, but got <class 'str'>."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


def test_random_crop_with_bbox_exception_03():
    """
    Feature: RandomCropWithBBox operation
    Description: Testing the RandomCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter size is larger than image dimensions, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = 100
    pad_if_needed = False
    with pytest.raises(RuntimeError,
                       match="RandomCrop: invalid crop size, crop size is bigger than the image dimensions."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter pad_if_needed is string, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = 100
    pad_if_needed = ""
    with pytest.raises(TypeError, match="Argument pad_if_needed with value .*, but got <class 'str'>."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter fill_value is 256, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = 256
    with pytest.raises(ValueError, match=r"Input fill_value is not within the required interval of \[0, 255\]."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                            fill_value=fill_value)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter fill_value is a 3-tuple, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = [255, 255, 255]
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                            fill_value=fill_value)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter fill_value is float, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = 1.5
    with pytest.raises(TypeError, match="fill_value should be a single integer or a 3-tuple."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                            fill_value=fill_value)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


def test_random_crop_with_bbox_exception_04():
    """
    Feature: RandomCropWithBBox operation
    Description: Testing the RandomCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # When parameter padding_mode is string, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = ""
    with pytest.raises(TypeError, match="Argument padding_mode with value .*, but got <class 'str'>."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed,
                                            fill_value=fill_value, padding_mode=padding_mode)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When no parameters are set, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="missing a required argument"):
        test_op = vision.RandomCropWithBBox()
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When no size parameter is provided, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    with pytest.raises(TypeError, match="missing a required argument"):
        test_op = vision.RandomCropWithBBox(padding=padding, pad_if_needed=pad_if_needed,
                                            fill_value=fill_value, padding_mode=padding_mode)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When setting extra parameters, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    more_para = None
    size = 3000
    padding = (100, 100, 100, 100)
    pad_if_needed = True
    fill_value = (255, 255, 255)
    padding_mode = mode.Border.CONSTANT
    with pytest.raises(TypeError, match="too many positional arguments"):
        test_op = vision.RandomCropWithBBox(size, padding, pad_if_needed, fill_value, padding_mode, more_para)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # When parameter pad_if_needed is 1, RandomCropWithBBox interface call fails
    dataset2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", decode=True, shuffle=False)
    size = 3000
    padding = 100
    pad_if_needed = 1
    with pytest.raises(TypeError, match="Argument pad_if_needed with value 1 is not of type \\[<class 'bool'>\\],"
                                        " but got <class 'int'>."):
        test_op = vision.RandomCropWithBBox(size=size, padding=padding, pad_if_needed=pad_if_needed)
        dataset2 = dataset2.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset2 = dataset2.project(columns=["image", "bbox"])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


if __name__ == "__main__":
    test_random_crop_with_bbox_op_c(plot_vis=True)
    test_random_crop_with_bbox_op_coco_c(plot_vis=True)
    test_random_crop_with_bbox_op2_c(plot_vis=True)
    test_random_crop_with_bbox_op3_c(plot_vis=True)
    test_random_crop_with_bbox_op_edge_c(plot_vis=True)
    test_random_crop_with_bbox_op_invalid_c()
    test_random_crop_with_bbox_op_bad_c()
    test_random_crop_with_bbox_op_bad_padding()
    test_random_crop_with_bbox_padded_dataset()
    test_random_crop_with_bbox_operation_01()
    test_random_crop_with_bbox_operation_02()
    test_random_crop_with_bbox_exception_01()
    test_random_crop_with_bbox_exception_02()
    test_random_crop_with_bbox_exception_03()
    test_random_crop_with_bbox_exception_04()
