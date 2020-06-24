# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.log as logger
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox, \
    config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"


def fix_annotate(bboxes):
    """
    Fix annotations to format followed by mindspore.
    :param bboxes: in [label, x_min, y_min, w, h, truncate, difficult] format
    :return: annotation in [x_min, y_min, w, h, label, truncate, difficult] format
    """
    for bbox in bboxes:
        tmp = bbox[0]
        bbox[0] = bbox[1]
        bbox[1] = bbox[2]
        bbox[2] = bbox[3]
        bbox[3] = bbox[4]
        bbox[4] = tmp
    return bboxes


def test_random_horizontal_flip_with_bbox_op_c(plot_vis=False):
    """
    Prints images side by side with and without Aug applied + bboxes to
    compare and test
    """
    logger.info("test_random_horizontal_flip_with_bbox_op_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    test_op = c_vision.RandomHorizontalFlipWithBBox(1)

    # maps to fix annotations to minddata standard
    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    # map to apply ops
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[test_op])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_horizontal_bbox_with_bbox_valid_rand_c(plot_vis=False):
    """
    Uses a valid non-default input, expect to pass
    Prints images side by side with and without Aug applied + bboxes to
    compare and test
    """
    logger.info("test_random_horizontal_bbox_valid_rand_c")

    original_seed = config_get_set_seed(1)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    test_op = c_vision.RandomHorizontalFlipWithBBox(0.6)

    # maps to fix annotations to minddata standard
    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    # map to apply ops
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[test_op])

    filename = "random_horizontal_flip_with_bbox_01_c_result.npz"
    save_and_check_md5(dataVoc2, filename, generate_golden=GENERATE_GOLDEN)

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_horizontal_flip_with_bbox_valid_edge_c(plot_vis=False):
    """
    Test RandomHorizontalFlipWithBBox op (testing with valid edge case, box covering full image).
    Prints images side by side with and without Aug applied + bboxes to compare and test
    """
    logger.info("test_horizontal_flip_with_bbox_valid_edge_c")

    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    test_op = c_vision.RandomHorizontalFlipWithBBox(1)

    # maps to fix annotations to minddata standard
    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    # map to apply ops
    # Add column for "annotation"
    dataVoc1 = dataVoc1.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=lambda img, bbox:
                            (img, np.array([[0, 0, img.shape[1], img.shape[0], 0, 0, 0]]).astype(np.uint32)))
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=lambda img, bbox:
                            (img, np.array([[0, 0, img.shape[1], img.shape[0], 0, 0, 0]]).astype(np.uint32)))
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[test_op])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_horizontal_flip_with_bbox_invalid_prob_c():
    """
    Test RandomHorizontalFlipWithBBox op with invalid input probability
    """
    logger.info("test_random_horizontal_bbox_invalid_prob_c")

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    try:
        # Note: Valid range of prob should be [0.0, 1.0]
        test_op = c_vision.RandomHorizontalFlipWithBBox(1.5)
        dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                                output_columns=["annotation"],
                                operations=fix_annotate)
        # map to apply ops
        dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                                output_columns=["image", "annotation"],
                                columns_order=["image", "annotation"],
                                operations=[test_op])  # Add column for "annotation"
    except ValueError as error:
        logger.info("Got an exception in DE: {}".format(str(error)))
        assert "Input is not" in str(error)


def test_random_horizontal_flip_with_bbox_invalid_bounds_c():
    """
    Test RandomHorizontalFlipWithBBox op with invalid bounding boxes
    """
    logger.info("test_random_horizontal_bbox_invalid_bounds_c")

    test_op = c_vision.RandomHorizontalFlipWithBBox(1)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(dataVoc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(dataVoc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(dataVoc2, test_op, InvalidBBoxType.NegativeXY, "min_x")
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(dataVoc2, test_op, InvalidBBoxType.WrongShape, "4 features")


if __name__ == "__main__":
    # set to false to not show plots
    test_random_horizontal_flip_with_bbox_op_c(plot_vis=False)
    test_random_horizontal_bbox_with_bbox_valid_rand_c(plot_vis=False)
    test_random_horizontal_flip_with_bbox_valid_edge_c(plot_vis=False)
    test_random_horizontal_flip_with_bbox_invalid_prob_c()
    test_random_horizontal_flip_with_bbox_invalid_bounds_c()
