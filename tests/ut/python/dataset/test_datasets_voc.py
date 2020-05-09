# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.dataset.transforms.vision.c_transforms as vision

import mindspore.dataset as ds

DATA_DIR = "../data/dataset/testVOC2012"
IMAGE_SHAPE = [2268, 2268, 2268, 2268, 642, 607, 561, 596, 612, 2268]
TARGET_SHAPE = [680, 680, 680, 680, 642, 607, 561, 596, 612, 680]

def test_voc_segmentation():
    data1 = ds.VOCDataset(DATA_DIR, task="Segmentation", mode="train", decode=True, shuffle=False)
    num = 0
    for item in data1.create_dict_iterator():
        assert (item["image"].shape[0] == IMAGE_SHAPE[num])
        assert (item["target"].shape[0] == TARGET_SHAPE[num])
        num += 1
    assert (num == 10)

def test_voc_detection():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True, shuffle=False)
    num = 0
    count = [ 0, 0, 0, 0, 0, 0 ]
    for item in data1.create_dict_iterator():
        assert (item["image"].shape[0] == IMAGE_SHAPE[num])
        for bbox in item["annotation"]:
            count[bbox[0]] += 1
        num += 1
    assert (num == 9)
    assert (count == [3,2,1,2,4,3])

def test_voc_class_index():
    class_index = { 'car': 0, 'cat': 1, 'train': 5 }
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", class_indexing=class_index, decode=True)
    class_index1 = data1.get_class_indexing()
    assert (class_index1 == { 'car': 0, 'cat': 1, 'train': 5 })
    data1 = data1.shuffle(4)
    class_index2 = data1.get_class_indexing()
    assert (class_index2 == { 'car': 0, 'cat': 1, 'train': 5 })
    num = 0
    count = [0,0,0,0,0,0]
    for item in data1.create_dict_iterator():
        for bbox in item["annotation"]:
            assert (bbox[0] == 0 or bbox[0] == 1 or bbox[0] == 5)
            count[bbox[0]] += 1
        num += 1
    assert (num == 6)
    assert (count == [3,2,0,0,0,3])

def test_voc_get_class_indexing():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True)
    class_index1 = data1.get_class_indexing()
    assert (class_index1 == { 'car': 0, 'cat': 1, 'chair': 2, 'dog': 3, 'person': 4, 'train': 5 })
    data1 = data1.shuffle(4)
    class_index2 = data1.get_class_indexing()
    assert (class_index2 == { 'car': 0, 'cat': 1, 'chair': 2, 'dog': 3, 'person': 4, 'train': 5 })
    num = 0
    count = [0,0,0,0,0,0]
    for item in data1.create_dict_iterator():
        for bbox in item["annotation"]:
            assert (bbox[0] == 0 or bbox[0] == 1 or bbox[0] == 2 or bbox[0] == 3 or bbox[0] == 4 or bbox[0] == 5)
            count[bbox[0]] += 1
        num += 1
    assert (num == 9)
    assert (count == [3,2,1,2,4,3])

def test_case_0():
    data1 = ds.VOCDataset(DATA_DIR, task="Segmentation", mode="train", decode=True)

    resize_op = vision.Resize((224, 224))

    data1 = data1.map(input_columns=["image"], operations=resize_op)
    data1 = data1.map(input_columns=["target"], operations=resize_op)
    repeat_num = 4
    data1 = data1.repeat(repeat_num)
    batch_size = 2
    data1 = data1.batch(batch_size, drop_remainder=True)

    num = 0
    for item in data1.create_dict_iterator():
        num += 1
    assert (num == 20)

def test_case_1():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", mode="train", decode=True)

    resize_op = vision.Resize((224, 224))

    data1 = data1.map(input_columns=["image"], operations=resize_op)
    repeat_num = 4
    data1 = data1.repeat(repeat_num)
    batch_size = 2
    data1 = data1.batch(batch_size, drop_remainder=True, pad_info={})

    num = 0
    for item in data1.create_dict_iterator():
        num += 1
    assert (num == 18)

def test_voc_exception():
    try:
        data1 = ds.VOCDataset(DATA_DIR, task="InvalidTask", mode="train", decode=True)
        for _ in data1.create_dict_iterator():
            pass
        assert False
    except ValueError:
        pass

    try:
        data2 = ds.VOCDataset(DATA_DIR, task="Segmentation", mode="train", class_indexing={ "cat":0 }, decode=True)
        for _ in data2.create_dict_iterator():
            pass
        assert False
    except ValueError:
        pass

    try:
        data3 = ds.VOCDataset(DATA_DIR, task="Detection", mode="notexist", decode=True)
        for _ in data3.create_dict_iterator():
            pass
        assert False
    except ValueError:
        pass

    try:
        data4 = ds.VOCDataset(DATA_DIR, task="Detection", mode="xmlnotexist", decode=True)
        for _ in data4.create_dict_iterator():
            pass
        assert False
    except RuntimeError:
        pass

    try:
        data5 = ds.VOCDataset(DATA_DIR, task="Detection", mode="invalidxml", decode=True)
        for _ in data5.create_dict_iterator():
            pass
        assert False
    except RuntimeError:
        pass

    try:
        data6 = ds.VOCDataset(DATA_DIR, task="Detection", mode="xmlnoobject", decode=True)
        for _ in data6.create_dict_iterator():
            pass
        assert False
    except RuntimeError:
        pass

if __name__ == '__main__':
    test_voc_segmentation()
    test_voc_detection()
    test_voc_class_index()
    test_voc_get_class_indexing()
    test_case_0()
    test_case_1()
    test_voc_exception()
