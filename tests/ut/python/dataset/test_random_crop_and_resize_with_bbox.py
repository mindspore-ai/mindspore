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
Testing RandomCropAndResizeWithBBox op in DE
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision

from mindspore import log as logger
from util import visualize_with_bounding_boxes, InvalidBBoxType, check_bad_bbox, \
    config_get_set_seed, config_get_set_num_parallel_workers, save_and_check_md5

GENERATE_GOLDEN = False

# Updated VOC dataset with correct annotations - DATA_DIR
DATA_DIR_VOC = "../data/dataset/testVOC2012_2"
# COCO dataset - DATA_DIR, ANNOTATION_DIR
DATA_DIR_COCO = ["../data/dataset/testCOCO/train/", "../data/dataset/testCOCO/annotations/train.json"]


def test_random_resized_crop_with_bbox_op_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomResizedCropWithBBox Op applied,
    tests with MD5 check, expected to pass
    """
    logger.info("test_random_resized_crop_with_bbox_op_c")

    original_seed = config_get_set_seed(23415)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = c_vision.RandomResizedCropWithBBox((256, 512), (0.5, 0.5), (0.5, 0.5))

    # map to apply ops
    dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            column_order=["image", "bbox"])

    filename = "random_resized_crop_with_bbox_01_c_result.npz"
    save_and_check_md5(dataVoc2, filename, generate_golden=GENERATE_GOLDEN)

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)

    # Restore config setting
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers(original_num_parallel_workers)


def test_random_resized_crop_with_bbox_op_coco_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomResizedCropWithBBox Op applied,
    Testing with Coco dataset
    """
    logger.info("test_random_resized_crop_with_bbox_op_coco_c")
    # load dataset
    dataCoco1 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                               decode=True, shuffle=False)

    dataCoco2 = ds.CocoDataset(DATA_DIR_COCO[0], annotation_file=DATA_DIR_COCO[1], task="Detection",
                               decode=True, shuffle=False)

    test_op = c_vision.RandomResizedCropWithBBox((512, 512), (0.5, 1), (0.5, 1))

    dataCoco2 = dataCoco2.map(operations=[test_op], input_columns=["image", "bbox"],
                              output_columns=["image", "bbox"],
                              column_order=["image", "bbox"])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataCoco1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataCoco2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp, "bbox")


def test_random_resized_crop_with_bbox_op_edge_c(plot_vis=False):
    """
    Prints images and bboxes side by side with and without RandomResizedCropWithBBox Op applied,
    tests on dynamically generated edge case, expected to pass
    """
    logger.info("test_random_resized_crop_with_bbox_op_edge_c")

    # Load dataset
    dataVoc1 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    test_op = c_vision.RandomResizedCropWithBBox((256, 512), (0.5, 0.5), (0.5, 0.5))

    # maps to convert data into valid edge case data
    dataVoc1 = dataVoc1.map(
        operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype))],
        input_columns=["image", "bbox"],
        output_columns=["image", "bbox"],
        column_order=["image", "bbox"])

    # Test Op added to list of Operations here
    dataVoc2 = dataVoc2.map(
        operations=[lambda img, bboxes: (img, np.array([[0, 0, img.shape[1], img.shape[0]]]).astype(bboxes.dtype)),
                    test_op], input_columns=["image", "bbox"],
        output_columns=["image", "bbox"],
        column_order=["image", "bbox"])

    unaugSamp, augSamp = [], []

    for unAug, Aug in zip(dataVoc1.create_dict_iterator(num_epochs=1, output_numpy=True),
                          dataVoc2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        unaugSamp.append(unAug)
        augSamp.append(Aug)

    if plot_vis:
        visualize_with_bounding_boxes(unaugSamp, augSamp)


def test_random_resized_crop_with_bbox_op_invalid_c():
    """
    Tests RandomResizedCropWithBBox on invalid constructor parameters, expected to raise ValueError
    """
    logger.info("test_random_resized_crop_with_bbox_op_invalid_c")

    # Load dataset, only Augmented Dataset as test will raise ValueError
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        # If input range of scale is not in the order of (min, max), ValueError will be raised.
        test_op = c_vision.RandomResizedCropWithBBox((256, 512), (1, 0.5), (0.5, 0.5))

        # map to apply ops
        dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                column_order=["image", "bbox"])

        for _ in dataVoc2.create_dict_iterator(num_epochs=1):
            break

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "scale should be in (min,max) format. Got (max,min)." in str(err)


def test_random_resized_crop_with_bbox_op_invalid2_c():
    """
     Tests RandomResizedCropWithBBox Op on invalid constructor parameters, expected to raise ValueError
    """
    logger.info("test_random_resized_crop_with_bbox_op_invalid2_c")
    # Load dataset # only loading the to AugDataset as test will fail on this
    dataVoc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)

    try:
        # If input range of ratio is not in the order of (min, max), ValueError will be raised.
        test_op = c_vision.RandomResizedCropWithBBox((256, 512), (1, 1), (1, 0.5))

        # map to apply ops
        dataVoc2 = dataVoc2.map(operations=[test_op], input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                column_order=["image", "bbox"])

        for _ in dataVoc2.create_dict_iterator(num_epochs=1):
            break

    except ValueError as err:
        logger.info("Got an exception in DE: {}".format(str(err)))
        assert "ratio should be in (min,max) format. Got (max,min)." in str(err)


def test_random_resized_crop_with_bbox_op_bad_c():
    """
    Test RandomCropWithBBox op with invalid bounding boxes, expected to catch multiple errors.
    """
    logger.info("test_random_resized_crop_with_bbox_op_bad_c")
    test_op = c_vision.RandomResizedCropWithBBox((256, 512), (0.5, 0.5), (0.5, 0.5))

    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WidthOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.HeightOverflow, "bounding boxes is out of bounds of the image")
    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.NegativeXY, "negative value")
    data_voc2 = ds.VOCDataset(DATA_DIR_VOC, task="Detection", usage="train", shuffle=False, decode=True)
    check_bad_bbox(data_voc2, test_op, InvalidBBoxType.WrongShape, "4 features")


if __name__ == "__main__":
    test_random_resized_crop_with_bbox_op_c(plot_vis=False)
    test_random_resized_crop_with_bbox_op_coco_c(plot_vis=False)
    test_random_resized_crop_with_bbox_op_edge_c(plot_vis=False)
    test_random_resized_crop_with_bbox_op_invalid_c()
    test_random_resized_crop_with_bbox_op_invalid2_c()
    test_random_resized_crop_with_bbox_op_bad_c()
