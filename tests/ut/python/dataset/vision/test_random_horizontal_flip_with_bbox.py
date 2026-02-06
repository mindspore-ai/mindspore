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
Testing the random horizontal flip with bounding boxes op in DE
"""
import os
import pytest

import mindspore.log as logger
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from util import InvalidBBoxType, check_bad_bbox, \
    config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5, \
    helper_perform_ops_bbox, helper_test_visual_bbox, helper_perform_ops_bbox_edgecase_float

GENERATE_GOLDEN = False

# updated VOC dataset with correct annotations
DATA_DIR = "../data/dataset/testVOC2012_2"
DATA_DIR_2 = ["../data/dataset/testCOCO/train/",
              "../data/dataset/testCOCO/annotations/train.json"]  # DATA_DIR, ANNOTATION_DIR
TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_random_horizontal_flip_with_bbox_op_c(plot_vis=False):
    """
    Feature: RandomHorizontalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without RandomHorizontalFlipWithBBox Op applied
    Expectation: Prints images and bboxes side by side
    """
    logger.info("test_random_horizontal_flip_with_bbox_op_c")

    # Load dataset
    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    test_op = vision.RandomHorizontalFlipWithBBox(1)

    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_horizontal_flip_with_bbox_op_coco_c(plot_vis=False):
    """
    Feature: RandomHorizontalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied using CocoDataset
    Expectation: Prints images and bboxes side by side
    """
    logger.info("test_random_horizontal_flip_with_bbox_op_coco_c")

    data_coco1 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = vision.RandomHorizontalFlipWithBBox(1)

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)


def test_random_horizontal_flip_with_bbox_valid_rand_c(plot_vis=False):
    """
    Feature: RandomHorizontalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without aug and the Op applied valid non-default input
    Expectation: Passes the comparison test
    """
    logger.info("test_random_horizontal_bbox_valid_rand_c")

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    test_op = vision.RandomHorizontalFlipWithBBox(0.6)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "random_horizontal_flip_with_bbox_01_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_flip_with_bbox_valid_edge_c(plot_vis=False):
    """
    Feature: RandomHorizontalFlipWithBBox op
    Description: Prints images side by side with and without aug applied and the Op to compare and test on edge case
    Expectation: Passes the edge case, box covering full image
    """
    logger.info("test_horizontal_flip_with_bbox_valid_edge_c")

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    test_op = vision.RandomHorizontalFlipWithBBox(1)

    # map to apply ops
    data_voc1 = helper_perform_ops_bbox_edgecase_float(data_voc1)
    data_voc2 = helper_perform_ops_bbox_edgecase_float(data_voc2)
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_horizontal_flip_with_bbox_invalid_prob_c():
    """
    Feature: RandomHorizontalFlipWithBBox op
    Description:test RandomHorizonFlipWithBBox op with invalid input probability
    Expectation: Error is raised as expected
    """
    logger.info("test_random_horizontal_bbox_invalid_prob_c")

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        test_op = vision.RandomHorizontalFlipWithBBox(1.5)
        # map to apply ops
        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            error)


def test_random_horizontal_flip_with_bbox_invalid_bounds_c():
    """
    Feature: RandomHorizontalFlipWithBBox op
    Description: Test RandomHorizontalFlipWithBBox op with invalid bounding boxes
    Expectation: Correct error is thrown as expected
    """
    logger.info("test_random_horizontal_bbox_invalid_bounds_c")

    test_op = vision.RandomHorizontalFlipWithBBox(1)

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow,
                   "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow,
                   "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op,
                   InvalidBBoxType.NegativeXY, "negative value")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op,
                   InvalidBBoxType.WrongShape, "4 features")


def test_random_horizontal_flip_with_bbox_operation_01():
    """
    Feature: RandomHorizontalFlipWithBBox operation
    Description: Testing the normal functionality of the RandomHorizontalFlipWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomHorizontalFlipWithBBox function with prob = 1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = 1
    test_op = vision.RandomHorizontalFlipWithBBox(prob=prob)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomHorizontalFlipWithBBox function with prob = 0
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = 0
    test_op = vision.RandomHorizontalFlipWithBBox(prob=prob)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomHorizontalFlipWithBBox function with all parameters
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = 1
    test_op = vision.RandomHorizontalFlipWithBBox(prob=prob)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomHorizontalFlipWithBBox function with no parameters
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomHorizontalFlipWithBBox()
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_horizontal_flip_with_bbox_exception_01():
    """
    Feature: RandomHorizontalFlipWithBBox operation
    Description: Testing the RandomHorizontalFlipWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomHorizontalFlipWithBBox function with image dataset
    data_dir_image = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_image, decode=True, shuffle=False)
    test_op = vision.RandomHorizontalFlipWithBBox(0.8)
    dataset = dataset.map(input_columns=["image", "label"],
                          output_columns=["image", "label"],
                          operations=[test_op])
    dataset = dataset.project(columns=["image", "label"])
    with pytest.raises(RuntimeError,
                       match="BoundingBox: bounding boxes should have to be two-dimensional matrix at least."):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomHorizontalFlipWithBBox function with prob = 1.1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = 1.1
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        test_op = vision.RandomHorizontalFlipWithBBox(prob=prob)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomHorizontalFlipWithBBox function with prob = -1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = -1
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        test_op = vision.RandomHorizontalFlipWithBBox(prob=prob)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomHorizontalFlipWithBBox function with prob = ""
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = ""
    with pytest.raises(TypeError, match="Argument prob"):
        test_op = vision.RandomHorizontalFlipWithBBox(prob=prob)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomHorizontalFlipWithBBox function with extra parameters
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = 1
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        test_op = vision.RandomHorizontalFlipWithBBox(prob, more_para)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


if __name__ == "__main__":
    # set to false to not show plots
    test_random_horizontal_flip_with_bbox_op_c(plot_vis=False)
    test_random_horizontal_flip_with_bbox_op_coco_c(plot_vis=False)
    test_random_horizontal_flip_with_bbox_valid_rand_c(plot_vis=False)
    test_random_horizontal_flip_with_bbox_valid_edge_c(plot_vis=False)
    test_random_horizontal_flip_with_bbox_invalid_prob_c()
    test_random_horizontal_flip_with_bbox_invalid_bounds_c()
    test_random_horizontal_flip_with_bbox_operation_01()
    test_random_horizontal_flip_with_bbox_exception_01()
