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
Testing the bounding box augment op in DE
"""
import mindspore.log as logger
import mindspore.dataset as ds
import mindspore.dataset.vision as c_vision

from util import config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5, \
    helper_perform_ops_bbox, helper_test_visual_bbox, helper_invalid_bounding_box_test, \
    helper_perform_ops_bbox_edgecase_float

GENERATE_GOLDEN = False

# updated VOC dataset with correct annotations
DATA_DIR = "../data/dataset/testVOC2012_2"
DATA_DIR_2 = ["../data/dataset/testCOCO/train/",
              "../data/dataset/testCOCO/annotations/train.json"]  # DATA_DIR, ANNOTATION_DIR


def test_bounding_box_augment_with_rotation_op(plot_vis=False):
    """
    Feature: BoundingBoxAugment op
    Description: Prints images and bboxes side by side with and without aug to compare by passing rotation op
    Expectation: Passes the md5 check test
    """
    logger.info("test_bounding_box_augment_with_rotation_op")

    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    # Ratio is set to 1 to apply rotation on all bounding boxes.
    test_op = c_vision.BoundingBoxAugment(c_vision.RandomRotation(90), 1)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "bounding_box_augment_rotation_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_bounding_box_augment_with_crop_op(plot_vis=False):
    """
    Feature: BoundingBoxAugment op
    Description: Prints images and bboxes side by side with and without aug to compare by passing crop op
    Expectation: Passes the md5 check test
    """
    logger.info("test_bounding_box_augment_with_crop_op")

    original_seed = config_get_set_seed(0)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    # Ratio is set to 0.9 to apply RandomCrop of size (50, 50) on 90% of the bounding boxes.
    test_op = c_vision.BoundingBoxAugment(c_vision.RandomCrop(50), 0.9)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "bounding_box_augment_crop_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_bounding_box_augment_valid_ratio_c(plot_vis=False):
    """
    Feature: BoundingBoxAugment op
    Description: Prints images and bboxes side by side with and without aug to compare with valid ratio
    Expectation: Passes the md5 check test
    """
    logger.info("test_bounding_box_augment_valid_ratio_c")

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    test_op = c_vision.BoundingBoxAugment(
        c_vision.RandomHorizontalFlip(1), 0.9)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "bounding_box_augment_valid_ratio_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_bounding_box_augment_op_coco_c(plot_vis=False):
    """
    Feature: BoundingBoxAugment op
    Description: Prints images and bboxes side by side with and without BoundingBoxAugment Op applied with CocoDataset
    Expectation: Passes the test
    """
    logger.info("test_bounding_box_augment_op_coco_c")

    data_coco1 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = c_vision.BoundingBoxAugment(c_vision.RandomHorizontalFlip(1), 1)

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)


def test_bounding_box_augment_valid_edge_c(plot_vis=False):
    """
    Feature: BoundingBoxAugment op
    Description: Prints images and bboxes side by side with and without aug to compare with valid edge case
    Expectation: Passes the md5 check test on edge case where box covering full image
    """
    logger.info("test_bounding_box_augment_valid_edge_c")

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    data_voc1 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    test_op = c_vision.BoundingBoxAugment(c_vision.RandomHorizontalFlip(1), 1)

    # map to apply ops
    data_voc1 = helper_perform_ops_bbox_edgecase_float(data_voc1)
    data_voc2 = helper_perform_ops_bbox_edgecase_float(data_voc2)
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])
    filename = "bounding_box_augment_valid_edge_c_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_bounding_box_augment_invalid_ratio_c():
    """
    Feature: BoundingBoxAugment op
    Description: Test BoundingBoxAugment op with invalid input ratio
    Expectation: Error is raised as expected
    """
    logger.info("test_bounding_box_augment_invalid_ratio_c")

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection",
                              usage="train", shuffle=False, decode=True)

    try:
        # ratio range is from 0 - 1
        test_op = c_vision.BoundingBoxAugment(
            c_vision.RandomHorizontalFlip(1), 1.5)
        # map to apply ops
        data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input ratio is not within the required interval of [0.0, 1.0]." in str(
            error)


def test_bounding_box_augment_invalid_bounds_c():
    """
    Feature: BoundingBoxAugment op
    Description: Test BoundingBoxAugment op with invalid bboxes
    Expectation: Correct error is thrown as expected
    """
    logger.info("test_bounding_box_augment_invalid_bounds_c")

    test_op = c_vision.BoundingBoxAugment(c_vision.RandomHorizontalFlip(1),
                                          1)

    helper_invalid_bounding_box_test(DATA_DIR, test_op)


if __name__ == "__main__":
    # set to false to not show plots
    test_bounding_box_augment_with_rotation_op(plot_vis=False)
    test_bounding_box_augment_with_crop_op(plot_vis=False)
    test_bounding_box_augment_op_coco_c(plot_vis=False)
    test_bounding_box_augment_valid_ratio_c(plot_vis=False)
    test_bounding_box_augment_valid_edge_c(plot_vis=False)
    test_bounding_box_augment_invalid_ratio_c()
    test_bounding_box_augment_invalid_bounds_c()
