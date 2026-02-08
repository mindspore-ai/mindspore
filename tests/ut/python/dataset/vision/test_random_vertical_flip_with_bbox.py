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
Testing RandomVerticalFlipWithBBox op in DE
"""
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision

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


def test_random_vertical_flip_with_bbox_op_c(plot_vis=False):
    """
    Feature: RandomVerticalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without RandomVerticalFlipWithBBox Op applied
    Expectation: Prints images and bboxes side by side
    """
    logger.info("test_random_vertical_flip_with_bbox_op_c")
    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomVerticalFlipWithBBox(1)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_vertical_flip_with_bbox_op_coco_c(plot_vis=False):
    """
    Feature: RandomVerticalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied with CocoDataset
    Expectation: Prints images and bboxes side by side
    """
    logger.info("test_random_vertical_flip_with_bbox_op_coco_c")
    # load dataset
    data_coco1 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = vision.RandomVerticalFlipWithBBox(1)

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)


def test_random_vertical_flip_with_bbox_op_rand_c(plot_vis=False):
    """
    Feature: RandomVerticalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied based on MD5 check
    Expectation: Passes the MD5 check test
    """
    logger.info("test_random_vertical_flip_with_bbox_op_rand_c")
    original_seed = config_get_set_seed(29847)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomVerticalFlipWithBBox(0.8)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "random_vertical_flip_with_bbox_01_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_vertical_flip_with_bbox_op_edge_c(plot_vis=False):
    """
    Feature: RandomVerticalFlipWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied on edge case
    Expectation: Passes the dynamically generated edge cases
    """
    logger.info("test_random_vertical_flip_with_bbox_op_edge_c")
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomVerticalFlipWithBBox(1)

    # maps to convert data into valid edge case data
    data_voc1 = helper_perform_ops_bbox(data_voc1, None, True)

    # Test Op added to list of Operations here
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op, True)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_vertical_flip_with_bbox_op_invalid_c():
    """
    Feature: RandomVerticalFlipWithBBox op
    Description: Tests RandomVerticalFlipWithBBox Op on invalid constructor parameters
    Expectation: Error is raised as expected
    """
    logger.info("test_random_vertical_flip_with_bbox_op_invalid_c")
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        test_op = vision.RandomVerticalFlipWithBBox(2)

        # map to apply ops
        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

        for _ in data_voc2.create_dict_iterator(num_epochs=1):
            break

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input prob is not within the required interval of [0.0, 1.0]." in str(
            err)


def test_random_vertical_flip_with_bbox_op_bad_c():
    """
    Feature: RandomVerticalFlipWithBBox op
    Description: Tests RandomVerticalFlipWithBBox Op with invalid bounding boxes
    Expectation: Multiple correct errors are caught as expected
    """
    logger.info("test_random_vertical_flip_with_bbox_op_bad_c")
    test_op = vision.RandomVerticalFlipWithBBox(1)

    helper_invalid_bounding_box_test(DATA_DIR_VOC, test_op)


def test_random_vertical_flip_with_bbox_operation_01():
    """
    Feature: RandomVerticalFlipWithBBox operation
    Description: Testing the normal functionality of the RandomVerticalFlipWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # RandomVerticalFlipWithBBox operator:Test prob is default
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox()
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 0.5
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(prob=0.5)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(1)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 0
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(0)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 0.9
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(0.9)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 0.1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(0.1)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 1.0
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(1.0)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass

    # RandomVerticalFlipWithBBox operator:Test prob is 0.0
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox(0.0)
    dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset2.create_dict_iterator(output_numpy=True):
        pass


def test_random_vertical_flip_with_bbox_exception_01():
    """
    Feature: RandomVerticalFlipWithBBox operation
    Description: Testing the RandomVerticalFlipWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # RandomVerticalFlipWithBBox operator:Test prob>1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=1.1)
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob<0
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=-0.1)
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob is bool
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="Argument prob with value True is not of type " + \
                                        "\\(<class 'float'>, <class 'int'>\\), but got <class 'bool'>"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=True)
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob is None
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="Argument prob with value None is not of type " + \
                                        "\\[<class 'float'>, <class 'int'>\\], but got <class 'NoneType'>"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=None)
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob is str
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="Argument prob with value test is not of type " + \
                                        "\\[<class 'float'>, <class 'int'>\\], but got <class 'str'>"):
        test_op = vision.RandomVerticalFlipWithBBox(prob='test')
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob is int and >1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(ValueError, match="Input prob is not within the required interval"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=2)
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob is list
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="Argument prob with value \\[0.1, 0.2\\] is not of type " + \
                                        "\\[<class 'float'>, <class 'int'>\\], but got <class 'list'>"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=[0.1, 0.2])
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


def test_random_vertical_flip_with_bbox_exception_02():
    """
    Feature: RandomVerticalFlipWithBBox operation
    Description: Testing the RandomVerticalFlipWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # RandomVerticalFlipWithBBox operator:Test prob is tuple
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="Argument prob with value \\(0.1, 0.2\\) is not of type " + \
                                        "\\[<class 'float'>, <class 'int'>\\], but got <class 'tuple'>"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=(0.1, 0.2))
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test prob is ""
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = ""
    with pytest.raises(TypeError, match="Argument prob with value .*, but got <class 'str'>"):
        test_op = vision.RandomVerticalFlipWithBBox(prob=prob)
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test image datasets without bbox columns
    data_dir_image = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_image, decode=True, shuffle=False)
    test_op = vision.RandomVerticalFlipWithBBox()
    dataset = dataset.map(input_columns=["image", "label"], operations=[test_op])
    with pytest.raises(RuntimeError,
                       match="BoundingBox: bounding boxes should have to be two-dimensional matrix at least."):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # RandomVerticalFlipWithBBox operator:Test input a parameter that does not exist
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset2 = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    prob = 1
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'test'"):
        test_op = vision.RandomVerticalFlipWithBBox(prob, test='test')
        dataset2 = dataset2.map(input_columns=["image", "bbox"], operations=[test_op])
        for _ in dataset2.create_dict_iterator(output_numpy=True):
            pass


if __name__ == "__main__":
    test_random_vertical_flip_with_bbox_op_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_coco_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_rand_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_edge_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_invalid_c()
    test_random_vertical_flip_with_bbox_op_bad_c()
    test_random_vertical_flip_with_bbox_operation_01()
    test_random_vertical_flip_with_bbox_exception_01()
    test_random_vertical_flip_with_bbox_exception_02()
