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
Testing RandomCropWithBBox op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.utils as mode

from mindspore import log as logger
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox, \
    config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5

GENERATE_GOLDEN = False

# updated VOC dataset with correct annotations
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


def test_random_crop_with_bbox_op_c(plot_vis=False):
    """
     Prints images and bboxes side by side with and without RandomCropWithBBox Op applied
    """
    logger.info("test_random_crop_with_bbox_op_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op UT
    test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

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
                            operations=[test_op])  # Add column for "annotation"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_crop_with_bbox_op2_c(plot_vis=False):
    """
     Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
     with md5 check, expected to pass
    """
    logger.info("test_random_crop_with_bbox_op2_c")
    original_seed = config_get_set_seed(593447)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], fill_value=(255, 255, 255))

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
                            operations=[test_op])  # Add column for "annotation"

    filename = "random_crop_with_bbox_01_c_result.npz"
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


def test_random_crop_with_bbox_op3_c(plot_vis=False):
    """
     Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
     with Padding Mode explicitly passed
    """
    logger.info("test_random_crop_with_bbox_op3_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

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
                            operations=[test_op])  # Add column for "annotation"

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_edge_c(plot_vis=False):
    """
     Prints images and bboxes side by side with and without RandomCropWithBBox Op applied,
     applied on dynamically generated edge case, expected to pass
    """
    logger.info("test_random_crop_with_bbox_op_edge_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    # define test OP with values to match existing Op unit - test
    test_op = c_vision.RandomCropWithBBox(512, [200, 200, 200, 200], padding_mode=mode.Border.EDGE)

    dataVoc1 = dataVoc1.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)
    dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                            output_columns=["annotation"],
                            operations=fix_annotate)

    # maps to convert data into valid edge case data
    dataVoc1 = dataVoc1.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype))])

    # Test Op added to list of Operations here
    dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                            output_columns=["image", "annotation"],
                            columns_order=["image", "annotation"],
                            operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype)), test_op])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(), dataVoc2.create_dict_iterator()):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_crop_with_bbox_op_invalid_c():
    """
    Test RandomCropWithBBox Op on invalid constructor parameters, expected to raise ValueError
    """
    logger.info("test_random_crop_with_bbox_op_invalid_c")

    # Load dataset
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)

    try:
        # define test OP with values to match existing Op unit - test
        test_op = c_vision.RandomCropWithBBox([512, 512, 375])

        dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                                output_columns=["annotation"],
                                operations=fix_annotate)

        # map to apply ops
        dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                                output_columns=["image", "annotation"],
                                columns_order=["image", "annotation"],
                                operations=[test_op])  # Add column for "annotation"

        for _ in dataVoc2.create_dict_iterator():
            break
    except TypeError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Size should be a single integer" in str(err)


def test_random_crop_with_bbox_op_bad_c():
    """
    Tests RandomCropWithBBox Op with invalid bounding boxes, expected to catch multiple errors.
    """
    logger.info("test_random_crop_with_bbox_op_bad_c")
    test_op = c_vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.NegativeXY, "min_x")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WrongShape, "4 features")


if __name__ == "__main__":
    test_random_crop_with_bbox_op_c(plot_vis=True)
    test_random_crop_with_bbox_op2_c(plot_vis=True)
    test_random_crop_with_bbox_op3_c(plot_vis=True)
    test_random_crop_with_bbox_op_edge_c(plot_vis=True)
    test_random_crop_with_bbox_op_invalid_c()
    test_random_crop_with_bbox_op_bad_c()
