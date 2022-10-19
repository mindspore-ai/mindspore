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
Testing RandomVerticalFlipWithBBox op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import log as logger
from util import config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5, \
    helper_perform_ops_bbox, helper_test_visual_bbox, helper_invalid_bounding_box_test

GENERATE_GOLDEN = False

# Updated VOC dataset with correct annotations - DATA_DIR
DATA_DIR_VOC = "../data/dataset/testVOC2012_2"
# COCO dataset - DATA_DIR, ANNOTATION_DIR
DATA_DIR_COCO = ["../data/dataset/testCOCO/train/",
                 "../data/dataset/testCOCO/annotations/train.json"]


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


if __name__ == "__main__":
    test_random_vertical_flip_with_bbox_op_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_coco_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_rand_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_edge_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_invalid_c()
    test_random_vertical_flip_with_bbox_op_bad_c()
