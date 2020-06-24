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
Testing RandomVerticalFlipWithBBox op
"""

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as c_vision

from mindspore import log as logger
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox

# updated VOC dataset with correct annotations
DATA_DIR = "../data/dataset/testVOC2012_2"


def fix_annotate(bboxes):
    """
    Update Current VOC dataset format to Proposed HQ BBox format

    :param bboxes: as [label, x_min, y_min, w, h, truncate, difficult]
    :return: annotation as [x_min, y_min, w, h, label, truncate, difficult]
    """
    for bbox in bboxes:
        tmp = bbox[0]
        bbox[0] = bbox[1]
        bbox[1] = bbox[2]
        bbox[2] = bbox[3]
        bbox[3] = bbox[4]
        bbox[4] = tmp
    return bboxes


def test_random_vertical_flip_with_bbox_op_c(plot_vis=False):
    """
     Prints images side by side with and without Aug applied + bboxes to
     compare and test
    """
    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    test_op = c_vision.RandomVerticalFlipWithBBox(1)

    # maps to fix annotations to HQ standard
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


def test_random_vertical_flip_with_bbox_op_rand_c(plot_vis=False):
    """
     Prints images side by side with and without Aug applied + bboxes to
     compare and test
    """

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    test_op = c_vision.RandomVerticalFlipWithBBox(0.6)

    # maps to fix annotations to HQ standard
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


def test_random_vertical_flip_with_bbox_op_invalid_c():
    # Should Fail
    # Load dataset
    dataVoc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train",
                             decode=True, shuffle=False)

    try:
        test_op = c_vision.RandomVerticalFlipWithBBox(2)

        # maps to fix annotations to HQ standard

        dataVoc2 = dataVoc2.map(input_columns=["annotation"],
                                output_columns=["annotation"],
                                operations=fix_annotate)
        # map to apply ops
        dataVoc2 = dataVoc2.map(input_columns=["image", "annotation"],
                                output_columns=["image", "annotation"],
                                columns_order=["image", "annotation"],
                                operations=[test_op])

        for _ in dataVoc2.create_dict_iterator():
            break

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "Input is not" in str(err)


def test_random_vertical_flip_with_bbox_op_bad_c():
    """
    Test RandomHorizontalFlipWithBBox op with invalid bounding boxes
    """
    logger.info("test_random_horizontal_bbox_invalid_bounds_c")
    test_op = c_vision.RandomVerticalFlipWithBBox(1)

    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.NegativeXY, "min_x")
    data_voc2 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WrongShape, "4 features")


if __name__ == "__main__":
    test_random_vertical_flip_with_bbox_op_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_rand_c(plot_vis=True)
    test_random_vertical_flip_with_bbox_op_invalid_c()
    test_random_vertical_flip_with_bbox_op_bad_c()
