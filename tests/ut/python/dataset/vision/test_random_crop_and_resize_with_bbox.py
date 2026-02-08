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
Testing RandomCropAndResizeWithBBox op in DE
"""
import os
import pytest

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision import Inter
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


def test_random_resized_crop_with_bbox_op_c(plot_vis=False):
    """
    Feature: RandomResizedCropWithBBox op
    Description: Prints images and bboxes side by side with and without RandomResizedCropWithBBox Op applied
    Expectation: Passes the MD5 check test
    """
    logger.info("test_random_resized_crop_with_bbox_op_c")

    original_seed = config_get_set_seed(23415)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomResizedCropWithBBox(
        (256, 512), (0.5, 0.5), (0.5, 0.5))

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "random_resized_crop_with_bbox_01_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_resized_crop_with_bbox_op_coco_c(plot_vis=False):
    """
    Feature: RandomResizedCropWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied with CocoDataset
    Expectation: Prints images and bboxes side by side as expected
    """
    logger.info("test_random_resized_crop_with_bbox_op_coco_c")
    # load dataset
    data_coco1 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = vision.RandomResizedCropWithBBox((512, 512), (0.5, 1), (0.5, 1))

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)


def test_random_resized_crop_with_bbox_op_edge_c(plot_vis=False):
    """
    Feature: RandomResizedCropWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied on edge case
    Expectation: Passes the dynamically generated edge case
    """
    logger.info("test_random_resized_crop_with_bbox_op_edge_c")

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomResizedCropWithBBox(
        (256, 512), (0.5, 0.5), (0.5, 0.5))

    # maps to convert data into valid edge case data
    data_voc1 = helper_perform_ops_bbox(data_voc1, None, True)

    # Test Op added to list of Operations here
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op, True)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_resized_crop_with_bbox_op_invalid_c():
    """
    Feature: RandomResizedCropWithBBox op
    Description: Test RandomResizedCropWithBBox on invalid constructor parameters (range of scale)
    Expectation: Error is raised as expected
    """
    logger.info("test_random_resized_crop_with_bbox_op_invalid_c")

    # Load dataset, only Augmented Dataset as test will raise ValueError
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        # If input range of scale is not in the order of (min, max), ValueError will be raised.
        test_op = vision.RandomResizedCropWithBBox(
            (256, 512), (1, 0.5), (0.5, 0.5))

        # map to apply ops
        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

        for _ in data_voc2.create_dict_iterator(num_epochs=1):
            break

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "scale should be in (min,max) format. Got (max,min)." in str(
            err)


def test_random_resized_crop_with_bbox_op_invalid2_c():
    """
    Feature: RandomResizedCropWithBBox op
    Description: Test RandomResizedCropWithBBox on invalid constructor parameters (range of ratio)
    Expectation: Error is raised as expected
    """
    logger.info("test_random_resized_crop_with_bbox_op_invalid2_c")
    # Load dataset # only loading the to AugDataset as test will fail on this
    data_voc2 = ds.VOCDataset(
        DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        # If input range of ratio is not in the order of (min, max), ValueError will be raised.
        test_op = vision.RandomResizedCropWithBBox(
            (256, 512), (1, 1), (1, 0.5))

        # map to apply ops
        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)

        for _ in data_voc2.create_dict_iterator(num_epochs=1):
            break

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "ratio should be in (min,max) format. Got (max,min)." in str(
            err)


def test_random_resized_crop_with_bbox_op_bad_c():
    """
    Feature: RandomResizedCropWithBBox op
    Description: Test RandomResizedCropWithBBox op with invalid bounding boxes
    Expectation: Multiple correct errors are caught as expected
    """
    logger.info("test_random_resized_crop_with_bbox_op_bad_c")
    test_op = vision.RandomResizedCropWithBBox(
        (256, 512), (0.5, 0.5), (0.5, 0.5))

    helper_invalid_bounding_box_test(DATA_DIR_VOC, test_op)


def test_random_resized_crop_with_bbox_operation_01():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the normal functionality of the RandomResizedCropWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCropWithBBox function size is 1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 1
    test_op = vision.RandomResizedCropWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "label"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function size is 500
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 500
    test_op = vision.RandomResizedCropWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function size is [500,520]
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, 520]
    test_op = vision.RandomResizedCropWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function scale is [10, 21]
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = [10, 21]
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function ratio is [0.5, 1.0]
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = [10, 21]
    ratio = [0.5, 1.0]
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  interpolation is Inter.BILINEAR
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_resized_crop_with_bbox_operation_02():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the normal functionality of the RandomResizedCropWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCropWithBBox function  interpolation is Inter.NEAREST
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.NEAREST
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  interpolation is Inter.BICUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BICUBIC
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function interpolation is Inter.CUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.CUBIC
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  interpolation is Inter.LINEAR
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.LINEAR
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  max_attempts is Inter.AREA
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.AREA
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_resized_crop_with_bbox_operation_03():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the normal functionality of the RandomResizedCropWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.NEAREST
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)

    v_p = vision.ToPIL()
    dataset = dataset.map(input_columns=["image"], operations=v_p)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.NEAREST
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.BILINEAR
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=v_d)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.BICUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=v_d)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BICUBIC
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.LINEAR
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=v_d)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.LINEAR
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.AREA
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=v_d)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.AREA
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.CUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=v_d)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.CUBIC
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_resized_crop_with_bbox_operation_04():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the normal functionality of the RandomResizedCropWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # Test RandomResizedCropWithBBox function input PIL data interpolation is Inter.PILCUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=v_d)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.PILCUBIC
    v_r = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"], output_columns=["image", "bbox"], operations=[v_r])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  all para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = [0.5, 1.0]
    interpolation = Inter.BILINEAR
    max_attempts = 10
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  no 2 para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    ratio = [0.5, 1.0]
    interpolation = Inter.BILINEAR
    max_attempts = 10
    test_op = vision.RandomResizedCropWithBBox(size=size, ratio=ratio, interpolation=interpolation,
                                                max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  no 3 para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 10
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, interpolation=interpolation,
                                                max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  no 4 para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = (0.5, 1.0)
    max_attempts = 10
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio,
                                                max_attempts=max_attempts)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # Test RandomResizedCropWithBBox function  no 5 para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                )
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_random_resized_crop_with_bbox_exception_01():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the RandomResizedCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCropWithBBox function with image dataset
    data_dir_image = os.path.join(TEST_DATA_DATASET_FUNC, "test_data", "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_image, decode=True, shuffle=False)
    test_op = vision.RandomResizedCropWithBBox(100)
    dataset = dataset.map(input_columns=["image", "label"],
                          output_columns=["image", "label"],
                          operations=[test_op])
    dataset = dataset.project(columns=["image", "label"])
    with pytest.raises(RuntimeError,
                       match="BoundingBox: bounding boxes should have to be two-dimensional matrix at least."):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomResizedCropWithBBox function size is 0
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        vision.RandomResizedCropWithBBox(size=size)

    # Test RandomResizedCropWithBBox function size is 16777217
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]."):
        vision.RandomResizedCropWithBBox(size=size)

    # Test RandomResizedCropWithBBox function size is -1
    size = -1
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[1, 16777216\\]"):
        vision.RandomResizedCropWithBBox(size=size)

    # Test RandomResizedCropWithBBox function size is 500.5
    size = 500.5
    with pytest.raises(TypeError, match="Argument size with value 500.5 is not of type \\[<class 'int'>, "
                                        "<class 'list'>, <class 'tuple'>\\], but got <class 'float'>."):
        vision.RandomResizedCropWithBBox(size=size)

    # Test RandomResizedCropWithBBox function size is 16777216
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 16777216
    test_op = vision.RandomResizedCropWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(["image", "bbox"])
    with pytest.raises(RuntimeError, match="CropAndResize: the resizing width or height 1\\) is too big, "
                                           "it's up to 1000 times the original image"):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomResizedCropWithBBox function size is (500, 500, 520)
    size = (500, 500, 520)
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        vision.RandomResizedCropWithBBox(size=size)

    # Test RandomResizedCropWithBBox function size is ""
    size = ""
    with pytest.raises(TypeError, match='''Argument size with value "" is not of type \\[<class 'int'>, '''
                                        '''<class 'list'>, <class 'tuple'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCropWithBBox(size=size)

    # Test RandomResizedCropWithBBox function scale is (-0.5, 1.5)
    size = (500, 520)
    scale = (-0.5, 1.5)
    with pytest.raises(ValueError, match="Input is not within the required interval of \\[0, 16777216\\]"):
        vision.RandomResizedCropWithBBox(size=size, scale=scale)

    # Test RandomResizedCropWithBBox function scale is (1.5, 0.5)
    size = (500, 520)
    scale = (1.5, 0.5)
    with pytest.raises(ValueError, match="scale should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomResizedCropWithBBox(size=size, scale=scale)

    # Test RandomResizedCropWithBBox function scale is ("", "")
    size = (500, 520)
    scale = ("", "")
    with pytest.raises(TypeError, match='''Argument scale\\[0\\] with value "" is not of type \\[<class 'float'>, '''
                                        '''<class 'int'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCropWithBBox(size=size, scale=scale)

    # Test RandomResizedCropWithBBox function scale is ms.Tensor([0.5, 1.0])
    size = (500, 520)
    scale = ms.Tensor([0.5, 1.0])
    with pytest.raises(TypeError) as e:
        vision.RandomResizedCropWithBBox(size=size, scale=scale)
    assert "Argument scale with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(scale) in str(e)

    # Test RandomResizedCropWithBBox function scale is ""
    size = (500, 520)
    scale = ""
    with pytest.raises(TypeError, match='''Argument scale with value "" is not of type \\[<class 'tuple'>, '''
                                        '''<class 'list'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCropWithBBox(size=size, scale=scale)

    # Test RandomResizedCropWithBBox function ratio is (-0.5, 1.0)
    size = (500, 520)
    scale = [0.5, 1.0]
    ratio = (-0.5, 1.0)
    with pytest.raises(ValueError, match="ratio\\[0\\] is not within the required interval of \\(0, 16777216\\]."):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio)


def test_random_resized_crop_with_bbox_exception_02():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the RandomResizedCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCropWithBBox function ratio = (1.5, 0.5)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (1.5, 0.5)
    with pytest.raises(ValueError, match="ratio should be in \\(min,max\\) format. Got \\(max,min\\)."):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCropWithBBox function ratio is ("", "")
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = ("", "")
    with pytest.raises(TypeError, match='''Argument ratio\\[0\\] with value "" is not of type \\[<class 'float'>, '''
                                        '''<class 'int'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCropWithBBox function ratio is ms.Tensor([0.5, 1.0])
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = ms.Tensor([0.5, 1.0])
    with pytest.raises(TypeError) as e:
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio)
    assert "Argument ratio with value {} is not of type [<class 'tuple'>, <class 'list'>]".format(ratio) in str(e)

    # Test RandomResizedCropWithBBox function ratio is ""
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = ""
    with pytest.raises(TypeError, match='''Argument ratio with value "" is not of type \\[<class 'tuple'>, '''
                                        '''<class 'list'>\\], but got <class 'str'>.'''):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio)

    # Test RandomResizedCropWithBBox function interpolation is ""
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = ""
    with pytest.raises(TypeError, match='''Argument interpolation with value "" is not of type \\[<enum 'Inter'>\\], '''
                                        '''but got <class 'str'>.'''):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    # Test RandomResizedCropWithBBox function   max_attempts is 0
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 0
    with pytest.raises(ValueError, match="max_attempts is not within the required interval of \\[1, 2147483647\\]"):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                          max_attempts=max_attempts)

    # Test RandomResizedCropWithBBox function   max_attempts is 1.5 ,required int
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 1.5
    with pytest.raises(TypeError, match="Argument max_attempts with value 1.5 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>."):
        test_op = vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomResizedCropWithBBox function   max_attempts is "" ,required int
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = ""
    with pytest.raises(TypeError, match="""Argument max_attempts with value "" is not of type \\[<class 'int'>\\], """
                                        """but got <class 'str'>."""):
        vision.RandomResizedCropWithBBox(size=size, scale=scale, ratio=ratio, interpolation=interpolation,
                                          max_attempts=max_attempts)

    # Test RandomResizedCropWithBBox function  no para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="missing a required argument"):
        test_op = vision.RandomResizedCropWithBBox()
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_random_resized_crop_with_bbox_exception_03():
    """
    Feature: RandomResizedCropWithBBox operation
    Description: Testing the RandomResizedCropWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # Test RandomResizedCropWithBBox function no 1 para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 10
    with pytest.raises(TypeError, match="missing a required argument"):
        test_op = vision.RandomResizedCropWithBBox(scale=scale, ratio=ratio, interpolation=interpolation,
                                                    max_attempts=max_attempts)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # Test RandomResizedCropWithBBox function  more para
    size = (500, 520)
    scale = (0.5, 1.0)
    ratio = (0.5, 1.0)
    interpolation = Inter.BILINEAR
    max_attempts = 10
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        vision.RandomResizedCropWithBBox(size, scale, ratio, interpolation, max_attempts, more_para)


if __name__ == "__main__":
    test_random_resized_crop_with_bbox_op_c(plot_vis=False)
    test_random_resized_crop_with_bbox_op_coco_c(plot_vis=False)
    test_random_resized_crop_with_bbox_op_edge_c(plot_vis=False)
    test_random_resized_crop_with_bbox_op_invalid_c()
    test_random_resized_crop_with_bbox_op_invalid2_c()
    test_random_resized_crop_with_bbox_op_bad_c()
    test_random_resized_crop_with_bbox_operation_01()
    test_random_resized_crop_with_bbox_operation_02()
    test_random_resized_crop_with_bbox_operation_03()
    test_random_resized_crop_with_bbox_operation_04()
    test_random_resized_crop_with_bbox_exception_01()
    test_random_resized_crop_with_bbox_exception_02()
    test_random_resized_crop_with_bbox_exception_03()
