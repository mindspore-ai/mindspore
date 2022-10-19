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
Testing the resize with bounding boxes op in DE
"""
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import log as logger
from util import save_and_check_md5, helper_perform_ops_bbox, helper_test_visual_bbox, helper_invalid_bounding_box_test

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"
DATA_DIR_2 = ["../data/dataset/testCOCO/train/",
              "../data/dataset/testCOCO/annotations/train.json"]  # DATA_DIR, ANNOTATION_DIR


def test_resize_with_bbox_op_voc_c(plot_vis=False):
    """
    Feature: ResizeWithBBox op
    Description: Prints images and bboxes side by side with and without ResizeWithBBox Op applied with VOCDataset
    Expectation: Passes the md5 check test
    """
    logger.info("test_resize_with_bbox_op_voc_c")

    # Load dataset
    data_voc1 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.ResizeWithBBox(100)

    # map to apply ops
    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op)
    data_voc2 = data_voc2.project(["image", "bbox"])

    filename = "resize_with_bbox_op_01_c_voc_result.npz"
    save_and_check_md5(data_voc2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_resize_with_bbox_op_coco_c(plot_vis=False):
    """
    Feature: ResizeWithBBox op
    Description: Prints images and bboxes side by side with and without ResizeWithBBox Op applied with CocoDataset
    Expectation: Prints images and bboxes side by side
    """
    logger.info("test_resize_with_bbox_op_coco_c")

    # Load dataset
    data_coco1 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    data_coco2 = ds.CocoDataset(DATA_DIR_2[0], annotation_file=DATA_DIR_2[1], task="Detection",
                                decode=True, shuffle=False)

    test_op = vision.ResizeWithBBox(200)

    # map to apply ops

    data_coco2 = helper_perform_ops_bbox(data_coco2, test_op)
    data_coco2 = data_coco2.project(["image", "bbox"])

    filename = "resize_with_bbox_op_01_c_coco_result.npz"
    save_and_check_md5(data_coco2, filename, generate_golden=GENERATE_GOLDEN)

    helper_test_visual_bbox(plot_vis, data_coco1, data_coco2)


def test_resize_with_bbox_op_edge_c(plot_vis=False):
    """
    Feature: ResizeWithBBox op
    Description: Prints images and bboxes side by side with and without ResizeWithBBox Op applied on edge case
    Expectation: Passes the dynamically generated edge case when bounding box has dimensions as the image itself
    """
    logger.info("test_resize_with_bbox_op_edge_c")
    data_voc1 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    data_voc2 = ds.VOCDataset(
        DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = vision.ResizeWithBBox(500)

    # maps to convert data into valid edge case data
    data_voc1 = helper_perform_ops_bbox(data_voc1, None, True)

    data_voc2 = helper_perform_ops_bbox(data_voc2, test_op, True)

    helper_test_visual_bbox(plot_vis, data_voc1, data_voc2)


def test_resize_with_bbox_op_invalid_c():
    """
    Feature: ResizeWithBBox op
    Description: Test ResizeWithBBox Op on invalid constructor parameters
    Expectation: Error is raised as expected
    """
    logger.info("test_resize_with_bbox_op_invalid_c")

    try:
        # invalid interpolation value
        vision.ResizeWithBBox(400, interpolation="invalid")

    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "interpolation" in str(err)


def test_resize_with_bbox_op_bad_c():
    """
    Feature: ResizeWithBBox op
    Description: Test ResizeWithBBox Op with invalid bounding boxes
    Expectation: Multiple errors are expected to be caught
    """
    logger.info("test_resize_with_bbox_op_bad_c")
    test_op = vision.ResizeWithBBox((200, 300))

    helper_invalid_bounding_box_test(DATA_DIR, test_op)


def test_resize_with_bbox_op_params_outside_of_interpolation_dict():
    """
    Feature: ResizeWithBBox op
    Description: Test ResizeWithBBox Op by passing an invalid key for interpolation
    Expectation: Error is raised as expected
    """
    logger.info("test_resize_with_bbox_op_params_outside_of_interpolation_dict")

    size = (500, 500)
    more_para = None
    with pytest.raises(KeyError, match="None"):
        vision.ResizeWithBBox(size, more_para)


if __name__ == "__main__":
    test_resize_with_bbox_op_voc_c(plot_vis=False)
    test_resize_with_bbox_op_coco_c(plot_vis=False)
    test_resize_with_bbox_op_edge_c(plot_vis=False)
    test_resize_with_bbox_op_invalid_c()
    test_resize_with_bbox_op_bad_c()
