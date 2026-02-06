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
Testing the resize with bounding boxes op in DE
"""
import os
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.transforms as vision
from mindspore.dataset.vision import Inter as v_Inter
from mindspore import log as logger
from util import save_and_check_md5, helper_perform_ops_bbox, helper_test_visual_bbox, helper_invalid_bounding_box_test

GENERATE_GOLDEN = False

DATA_DIR = "../data/dataset/testVOC2012_2"
DATA_DIR_2 = ["../data/dataset/testCOCO/train/",
              "../data/dataset/testCOCO/annotations/train.json"]  # DATA_DIR, ANNOTATION_DIR
TEST_DATA_DATASET_FUNC ="../data/dataset/"


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


def test_resize_with_bbox_operation_01():
    """
    Feature: ResizeWithBBox operation
    Description: Testing the normal functionality of the ResizeWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # ResizeWithBBox operator:Test size is 1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 1
    test_op = vision.ResizeWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test size is 500
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 500
    test_op = vision.ResizeWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test size is a list sequence of length 2
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, 520]
    test_op = vision.ResizeWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test size is a tuple sequence of length 2
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    test_op = vision.ResizeWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test interpolation is Inter.LINEAR
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.LINEAR
    test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test interpolation is Inter.NEAREST
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.NEAREST
    test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_resize_with_bbox_operation_02():
    """
    Feature: ResizeWithBBox operation
    Description: Testing the normal functionality of the ResizeWithBBox operator
    Expectation: The Output is equal to the expected output
    """
    # ResizeWithBBox operator:Test interpolation is Inter.BICUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.BICUBIC
    test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test interpolation is Inter.PILCUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.PILCUBIC
    test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test interpolation is Inter.CUBIC
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.CUBIC
    test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test interpolation is Inter.AREA
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.AREA
    test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
    dataset = dataset.map(input_columns=["image", "bbox"],
                            output_columns=["image", "bbox"],
                            operations=[test_op])
    dataset = dataset.project(columns=["image", "bbox"])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

    # ResizeWithBBox operator:Test PIL data
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=False, shuffle=False)
    v_d = vision.Decode(to_pil=True)
    dataset = dataset.map(input_columns=["image"], operations=[v_d])
    size = (500, 520)
    test_op = vision.ResizeWithBBox(size=size)
    dataset = dataset.map(input_columns=["image", "bbox"], operations=[test_op])
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass


def test_resize_with_bbox_exception_01():
    """
    Feature: ResizeWithBBox operation
    Description: Testing the ResizeWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # ResizeWithBBox operator:Test size is 0
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 0
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is 16777217
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 16777217
    with pytest.raises(ValueError, match="Input is not within the required interval"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is 16777216
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 16777216
    with pytest.raises(RuntimeError, match="map operation: \\[ResizeWithBBox\\] failed"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is float
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = 500.5
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is None
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = None
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_resize_with_bbox_exception_02():
    """
    Feature: ResizeWithBBox operation
    Description: Testing the ResizeWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # ResizeWithBBox operator:Test size is str
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = "test"
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is ""
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = ""
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is a sequence of length 1
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [100]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is a sequence of length 3
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, 500, 520]
    with pytest.raises(TypeError, match="Size should be a single integer or a list/tuple"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is a sequence containing a float of 2 lengths
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500.5, 500]
    with pytest.raises(TypeError, match="Argument size at dim 0 with value 500.5 is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'float'>"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_resize_with_bbox_exception_03():
    """
    Feature: ResizeWithBBox operation
    Description: Testing the ResizeWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # ResizeWithBBox operator:Test size is a sequence containing a str of 2 lengths
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, 'test']
    with pytest.raises(TypeError, match="Argument size at dim 1 with value test is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'str'>."):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is a sequence containing None of 2 lengths
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, None]
    with pytest.raises(TypeError, match="Argument size at dim 1 with value None is not of type " + \
                                        "\\[<class 'int'>\\], but got <class 'NoneType'>"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test size is a sequence containing bool of 2 lengths
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = [500, True]
    with pytest.raises(TypeError, match="Argument size at dim 1 with value True is not of type " + \
                                        "\\(<class 'int'>,\\), but got <class 'bool'>"):
        test_op = vision.ResizeWithBBox(size=size)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test interpolation is ""
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = ""
    with pytest.raises(TypeError, match="Argument interpolation with value \"\" is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>"):
        test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test interpolation is str
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = "test"
    with pytest.raises(TypeError, match="Argument interpolation with value test is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'str'>"):
        test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


def test_resize_with_bbox_exception_04():
    """
    Feature: ResizeWithBBox operation
    Description: Testing the ResizeWithBBox Operator in Exceptional Scenarios
    Expectation: Throw an exception
    """
    # ResizeWithBBox operator:Test interpolation is None
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = None
    with pytest.raises(KeyError, match="Interpolation should not be None"):
        test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test interpolation is bool
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = True
    with pytest.raises(TypeError, match="Argument interpolation with value True is not of type " + \
                                        "\\[<enum 'Inter'>\\], but got <class 'bool'>."):
        test_op = vision.ResizeWithBBox(size=size, interpolation=interpolation)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test no para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    with pytest.raises(TypeError, match="missing a required argument: 'size'"):
        test_op = vision.ResizeWithBBox()
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test more para
    data_dir_voc = os.path.join(TEST_DATA_DATASET_FUNC, "testVOC2012_2")
    dataset = ds.VOCDataset(data_dir_voc, task="Detection", usage="train", decode=True, shuffle=False)
    size = (500, 520)
    interpolation = v_Inter.LINEAR
    more_para = None
    with pytest.raises(TypeError, match="too many positional arguments"):
        test_op = vision.ResizeWithBBox(size, interpolation, more_para)
        dataset = dataset.map(input_columns=["image", "bbox"],
                                output_columns=["image", "bbox"],
                                operations=[test_op])
        dataset = dataset.project(columns=["image", "bbox"])
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass

    # ResizeWithBBox operator:Test image dataset without bounding boxes
    data_dir_image = os.path.join(TEST_DATA_DATASET_FUNC, "testImageNetData", "train")
    dataset = ds.ImageFolderDataset(data_dir_image, decode=True, shuffle=False)
    test_op = vision.ResizeWithBBox(512)
    dataset = dataset.map(input_columns=["image", "label"],
                          output_columns=["image", "label"],
                          operations=[test_op])
    dataset = dataset.project(columns=["image", "label"])
    with pytest.raises(RuntimeError,
                       match=" BoundingBox: bounding boxes should have to be two-dimensional matrix at least."):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            pass


if __name__ == "__main__":
    test_resize_with_bbox_op_voc_c(plot_vis=False)
    test_resize_with_bbox_op_coco_c(plot_vis=False)
    test_resize_with_bbox_op_edge_c(plot_vis=False)
    test_resize_with_bbox_op_invalid_c()
    test_resize_with_bbox_op_bad_c()
    test_resize_with_bbox_operation_01()
    test_resize_with_bbox_operation_02()
    test_resize_with_bbox_exception_01()
    test_resize_with_bbox_exception_02()
    test_resize_with_bbox_exception_03()
    test_resize_with_bbox_exception_04()
