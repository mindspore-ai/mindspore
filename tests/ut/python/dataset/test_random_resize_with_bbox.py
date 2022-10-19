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
Testing the random resize with bounding boxes op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import log as logger
from util import config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5, \
    helper_perform_ops_bbox, helper_test_visual_bbox, helper_invalid_bounding_box_test

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"
DATA_DIR_2 = ["../data/dataset/testCOCO/train/",
              "../data/dataset/testCOCO/annotations/train.json"]  # DATA_DIR, ANNOTATION_DIR


def test_random_resize_with_bbox_op_voc_c(plot_vis=False):
    """
    Feature: RandomResizeWithBBox op
    Description: Prints images and bboxes side by side with and without RandomResizeWithBBox Op applied using VOCDataset
    Expectation: Images and bboxes are printed side by side
    """
    logger.info("test_random_resize_with_bbox_op_voc_c")
    original_seed = config_get_set_seed(123)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)
    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomResizeWithBBox(100)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "random_resize_with_bbox_op_01_c_voc_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_resize_with_bbox_op_rand_coco_c(plot_vis=False):
    """
    Feature: RandomResizeWithBBox op
    Description: Prints images and bboxes side by side with and without the Op applied using CocoDataset
    Expectation: Passes the MD5 check
    """
    logger.info("test_random_resize_with_bbox_op_rand_coco_c")
    original_seed = config_get_set_seed(231)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    data_coco1 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = vision.RandomResizeWithBBox(200)

    # map to apply ops

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)
    data_coco2 = data_coco2.project(["image", "bbox"])

    filename = "random_resize_with_bbox_op_01_c_coco_result.npz"
    save_and_check_md5(data_coco2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_resize_with_bbox_op_edge_c(plot_vis=False):
    """
    Feature: RandomResizeWithBBox op
    Description: Prints images and bboxes side by side with and without thr Op applied on edge case
    Expectation: Passes the dynamically generated edge case when bounding box has dimensions as the image itself
    """
    logger.info("test_random_resize_with_bbox_op_edge_c")
    data_voc1 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.RandomResizeWithBBox(500)

    # maps to convert data into valid edge case data
    data_voc1 = helper_perform_ops_bbox(data_voc1, None, True)

    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op, True)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_random_resize_with_bbox_op_invalid_c():
    """
    Feature: RandomResizeWithBBox op
    Description: Test RandomResizeWithBBox op on invalid constructor parameters
    Expectation: Error is raised as expected
    """
    logger.info("test_random_resize_with_bbox_op_invalid_c")

    try:
        # zero value for resize
        vision.RandomResizeWithBBox(0)

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input is not within the required interval of [1, 16777216]." in str(
            err)

    try:
        # one of the size values is zero
        vision.RandomResizeWithBBox((0, 100))

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input size at dim 0 is not within the required interval of [1, 2147483647]." in str(
            err)

    try:
        # negative value for resize
        vision.RandomResizeWithBBox(-10)

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input is not within the required interval of [1, 16777216]." in str(
            err)

    try:
        # invalid input shape
        vision.RandomResizeWithBBox((100, 100, 100))

    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Size should be" in str(err)


def test_random_resize_with_bbox_op_bad_c():
    """
    Feature: RandomResizeWithBBox op
    Description: Tests RandomResizeWithBBox Op with invalid bounding boxes
    Expectation: Multiple errors are caught as expected
    """
    logger.info("test_random_resize_with_bbox_op_bad_c")
    test_op = vision.RandomResizeWithBBox((400, 300))

    helper_invalid_bounding_box_test(DATA_DIR, test_op)


if __name__ == "__main__":
    test_random_resize_with_bbox_op_voc_c(plot_vis=False)
    test_random_resize_with_bbox_op_rand_coco_c(plot_vis=False)
    test_random_resize_with_bbox_op_edge_c(plot_vis=False)
    test_random_resize_with_bbox_op_invalid_c()
    test_random_resize_with_bbox_op_bad_c()
