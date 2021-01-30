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
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision

DATA_DIR = "../data/dataset/testCOCO/train/"
DATA_DIR_2 = "../data/dataset/testCOCO/train"
ANNOTATION_FILE = "../data/dataset/testCOCO/annotations/train.json"
KEYPOINT_FILE = "../data/dataset/testCOCO/annotations/key_point.json"
PANOPTIC_FILE = "../data/dataset/testCOCO/annotations/panoptic.json"
INVALID_FILE = "../data/dataset/testCOCO/annotations/invalid.json"
LACKOFIMAGE_FILE = "../data/dataset/testCOCO/annotations/lack_of_images.json"
INVALID_CATEGORY_ID_FILE = "../data/dataset/testCOCO/annotations/invalid_category_id.json"


def test_coco_detection():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection",
                           decode=True, shuffle=False)
    num_iter = 0
    image_shape = []
    bbox = []
    category_id = []
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image_shape.append(data["image"].shape)
        bbox.append(data["bbox"])
        category_id.append(data["category_id"])
        num_iter += 1
    assert num_iter == 6
    assert image_shape[0] == (2268, 4032, 3)
    assert image_shape[1] == (561, 595, 3)
    assert image_shape[2] == (607, 585, 3)
    assert image_shape[3] == (642, 675, 3)
    assert image_shape[4] == (2268, 4032, 3)
    assert image_shape[5] == (2268, 4032, 3)
    np.testing.assert_array_equal(np.array([[10., 10., 10., 10.], [70., 70., 70., 70.]]), bbox[0])
    np.testing.assert_array_equal(np.array([[20., 20., 20., 20.], [80., 80., 80.0, 80.]]), bbox[1])
    np.testing.assert_array_equal(np.array([[30.0, 30.0, 30.0, 30.]]), bbox[2])
    np.testing.assert_array_equal(np.array([[40., 40., 40., 40.]]), bbox[3])
    np.testing.assert_array_equal(np.array([[50., 50., 50., 50.]]), bbox[4])
    np.testing.assert_array_equal(np.array([[60., 60., 60., 60.]]), bbox[5])
    np.testing.assert_array_equal(np.array([[1], [7]]), category_id[0])
    np.testing.assert_array_equal(np.array([[2], [8]]), category_id[1])
    np.testing.assert_array_equal(np.array([[3]]), category_id[2])
    np.testing.assert_array_equal(np.array([[4]]), category_id[3])
    np.testing.assert_array_equal(np.array([[5]]), category_id[4])
    np.testing.assert_array_equal(np.array([[6]]), category_id[5])


def test_coco_stuff():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff",
                           decode=True, shuffle=False)
    num_iter = 0
    image_shape = []
    segmentation = []
    iscrowd = []
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image_shape.append(data["image"].shape)
        segmentation.append(data["segmentation"])
        iscrowd.append(data["iscrowd"])
        num_iter += 1
    assert num_iter == 6
    assert image_shape[0] == (2268, 4032, 3)
    assert image_shape[1] == (561, 595, 3)
    assert image_shape[2] == (607, 585, 3)
    assert image_shape[3] == (642, 675, 3)
    assert image_shape[4] == (2268, 4032, 3)
    assert image_shape[5] == (2268, 4032, 3)
    np.testing.assert_array_equal(np.array([[10., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
                                            [70., 72., 73., 74., 75., -1., -1., -1., -1., -1.]]),
                                  segmentation[0])
    np.testing.assert_array_equal(np.array([[0], [0]]), iscrowd[0])
    np.testing.assert_array_equal(np.array([[20.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
                                            [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, -1.0]]),
                                  segmentation[1])
    np.testing.assert_array_equal(np.array([[0], [1]]), iscrowd[1])
    np.testing.assert_array_equal(np.array([[40., 42., 43., 44., 45., 46., 47., 48., 49., 40., 41., 42.]]),
                                  segmentation[2])
    np.testing.assert_array_equal(np.array([[0]]), iscrowd[2])
    np.testing.assert_array_equal(np.array([[50., 52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63.]]),
                                  segmentation[3])
    np.testing.assert_array_equal(np.array([[0]]), iscrowd[3])
    np.testing.assert_array_equal(np.array([[60., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.]]),
                                  segmentation[4])
    np.testing.assert_array_equal(np.array([[0]]), iscrowd[4])
    np.testing.assert_array_equal(np.array([[60., 62., 63., 64., 65., 66., 67.], [68., 69., 70., 71., 72., 73., 74.]]),
                                  segmentation[5])
    np.testing.assert_array_equal(np.array([[0]]), iscrowd[5])


def test_coco_keypoint():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint",
                           decode=True, shuffle=False)
    num_iter = 0
    image_shape = []
    keypoints = []
    num_keypoints = []
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image_shape.append(data["image"].shape)
        keypoints.append(data["keypoints"])
        num_keypoints.append(data["num_keypoints"])
        num_iter += 1
    assert num_iter == 2
    assert image_shape[0] == (2268, 4032, 3)
    assert image_shape[1] == (561, 595, 3)
    np.testing.assert_array_equal(np.array([[368., 61., 1., 369., 52., 2., 0., 0., 0., 382., 48., 2., 0., 0., 0., 368.,
                                             84., 2., 435., 81., 2., 362., 125., 2., 446., 125., 2., 360., 153., 2., 0.,
                                             0., 0., 397., 167., 1., 439., 166., 1., 369., 193., 2., 461., 234., 2.,
                                             361., 246., 2., 474., 287., 2.]]), keypoints[0])
    np.testing.assert_array_equal(np.array([[14]]), num_keypoints[0])
    np.testing.assert_array_equal(np.array([[244., 139., 2., 0., 0., 0., 226., 118., 2., 0., 0., 0., 154., 159., 2.,
                                             143., 261., 2., 135., 312., 2., 271., 423., 2., 184., 530., 2., 261., 280.,
                                             2., 347., 592., 2., 0., 0., 0., 123., 596., 2., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0.]]),
                                  keypoints[1])
    np.testing.assert_array_equal(np.array([[10]]), num_keypoints[1])


def test_coco_panoptic():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic", decode=True, shuffle=False)
    num_iter = 0
    image_shape = []
    bbox = []
    category_id = []
    iscrowd = []
    area = []
    for data in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image_shape.append(data["image"].shape)
        bbox.append(data["bbox"])
        category_id.append(data["category_id"])
        iscrowd.append(data["iscrowd"])
        area.append(data["area"])
        num_iter += 1
    assert num_iter == 2
    assert image_shape[0] == (2268, 4032, 3)
    np.testing.assert_array_equal(np.array([[472, 173, 36, 48], [340, 22, 154, 301], [486, 183, 30, 35]]), bbox[0])
    np.testing.assert_array_equal(np.array([[1], [1], [2]]), category_id[0])
    np.testing.assert_array_equal(np.array([[0], [0], [0]]), iscrowd[0])
    np.testing.assert_array_equal(np.array([[705], [14062], [626]]), area[0])
    assert image_shape[1] == (642, 675, 3)
    np.testing.assert_array_equal(np.array([[103, 133, 229, 422], [243, 175, 93, 164]]), bbox[1])
    np.testing.assert_array_equal(np.array([[1], [3]]), category_id[1])
    np.testing.assert_array_equal(np.array([[0], [0]]), iscrowd[1])
    np.testing.assert_array_equal(np.array([[43102], [6079]]), area[1])


def test_coco_detection_classindex():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", decode=True)
    class_index = data1.get_class_indexing()
    assert class_index == {'person': [1], 'bicycle': [2], 'car': [3], 'cat': [4], 'dog': [5], 'monkey': [6],
                           'bag': [7], 'orange': [8]}
    num_iter = 0
    for _ in data1.__iter__():
        num_iter += 1
    assert num_iter == 6


def test_coco_panootic_classindex():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic", decode=True)
    class_index = data1.get_class_indexing()
    assert class_index == {'person': [1, 1], 'bicycle': [2, 1], 'car': [3, 1]}
    num_iter = 0
    for _ in data1.__iter__():
        num_iter += 1
    assert num_iter == 2


def test_coco_case_0():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", decode=True)
    data1 = data1.shuffle(10)
    data1 = data1.batch(3, pad_info={})
    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 2


def test_coco_case_1():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", decode=True)
    sizes = [0.5, 0.5]
    randomize = False
    dataset1, dataset2 = data1.split(sizes=sizes, randomize=randomize)

    num_iter = 0
    for _ in dataset1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3
    num_iter = 0
    for _ in dataset2.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 3


def test_coco_case_2():
    data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection", decode=True)
    resize_op = vision.Resize((224, 224))

    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.repeat(4)
    num_iter = 0
    for _ in data1.__iter__():
        num_iter += 1
    assert num_iter == 24


def test_coco_case_3():
    data1 = ds.CocoDataset(DATA_DIR_2, annotation_file=ANNOTATION_FILE, task="Detection", decode=True)
    resize_op = vision.Resize((224, 224))

    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.repeat(4)
    num_iter = 0
    for _ in data1.__iter__():
        num_iter += 1
    assert num_iter == 24


def test_coco_case_exception():
    try:
        data1 = ds.CocoDataset("path_not_exist/", annotation_file=ANNOTATION_FILE, task="Detection")
        for _ in data1.__iter__():
            pass
        assert False
    except ValueError as e:
        assert "does not exist or permission denied" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file="./file_not_exist", task="Detection")
        for _ in data1.__iter__():
            pass
        assert False
    except ValueError as e:
        assert "does not exist or permission denied" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Invalid task")
        for _ in data1.__iter__():
            pass
        assert False
    except ValueError as e:
        assert "Invalid task type" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=LACKOFIMAGE_FILE, task="Detection")
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "invalid node found in json" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=INVALID_CATEGORY_ID_FILE, task="Detection")
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "category_id can't find in categories" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=INVALID_FILE, task="Detection")
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "failed to open json file" in str(e)

    try:
        sampler = ds.PKSampler(3)
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=INVALID_FILE, task="Detection", sampler=sampler)
        for _ in data1.__iter__():
            pass
        assert False
    except ValueError as e:
        assert "CocoDataset doesn't support PKSampler" in str(e)

    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection")
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection")
        data1 = data1.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection")
        data1 = data1.map(operations=exception_func, input_columns=["bbox"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Detection")
        data1 = data1.map(operations=exception_func, input_columns=["category_id"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff")
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff")
        data1 = data1.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff")
        data1 = data1.map(operations=exception_func, input_columns=["segmentation"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=ANNOTATION_FILE, task="Stuff")
        data1 = data1.map(operations=exception_func, input_columns=["iscrowd"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint")
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint")
        data1 = data1.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint")
        data1 = data1.map(operations=exception_func, input_columns=["keypoints"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=KEYPOINT_FILE, task="Keypoint")
        data1 = data1.map(operations=exception_func, input_columns=["num_keypoints"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic")
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic")
        data1 = data1.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data1 = data1.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic")
        data1 = data1.map(operations=exception_func, input_columns=["bbox"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic")
        data1 = data1.map(operations=exception_func, input_columns=["category_id"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data1 = ds.CocoDataset(DATA_DIR, annotation_file=PANOPTIC_FILE, task="Panoptic")
        data1 = data1.map(operations=exception_func, input_columns=["area"], num_parallel_workers=1)
        for _ in data1.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


if __name__ == '__main__':
    test_coco_detection()
    test_coco_stuff()
    test_coco_keypoint()
    test_coco_panoptic()
    test_coco_detection_classindex()
    test_coco_panootic_classindex()
    test_coco_case_0()
    test_coco_case_1()
    test_coco_case_2()
    test_coco_case_3()
    test_coco_case_exception()
