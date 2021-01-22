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
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision

DATA_DIR = "../data/dataset/testVOC2012"
IMAGE_SHAPE = [2268, 2268, 2268, 2268, 642, 607, 561, 596, 612, 2268]
TARGET_SHAPE = [680, 680, 680, 680, 642, 607, 561, 596, 612, 680]


def test_voc_segmentation():
    data1 = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", shuffle=False, decode=True)
    num = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["image"].shape[0] == IMAGE_SHAPE[num]
        assert item["target"].shape[0] == TARGET_SHAPE[num]
        num += 1
    assert num == 10


def test_voc_detection():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)
    num = 0
    count = [0, 0, 0, 0, 0, 0]
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["image"].shape[0] == IMAGE_SHAPE[num]
        for label in item["label"]:
            count[label[0]] += 1
        num += 1
    assert num == 9
    assert count == [3, 2, 1, 2, 4, 3]


def test_voc_class_index():
    class_index = {'car': 0, 'cat': 1, 'train': 5}
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", class_indexing=class_index, decode=True)
    class_index1 = data1.get_class_indexing()
    assert (class_index1 == {'car': 0, 'cat': 1, 'train': 5})
    data1 = data1.shuffle(4)
    class_index2 = data1.get_class_indexing()
    assert (class_index2 == {'car': 0, 'cat': 1, 'train': 5})
    num = 0
    count = [0, 0, 0, 0, 0, 0]
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        for label in item["label"]:
            count[label[0]] += 1
            assert label[0] in (0, 1, 5)
        num += 1
    assert num == 6
    assert count == [3, 2, 0, 0, 0, 3]


def test_voc_get_class_indexing():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", decode=True)
    class_index1 = data1.get_class_indexing()
    assert (class_index1 == {'car': 0, 'cat': 1, 'chair': 2, 'dog': 3, 'person': 4, 'train': 5})
    data1 = data1.shuffle(4)
    class_index2 = data1.get_class_indexing()
    assert (class_index2 == {'car': 0, 'cat': 1, 'chair': 2, 'dog': 3, 'person': 4, 'train': 5})
    num = 0
    count = [0, 0, 0, 0, 0, 0]
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        for label in item["label"]:
            count[label[0]] += 1
            assert label[0] in (0, 1, 2, 3, 4, 5)
        num += 1
    assert num == 9
    assert count == [3, 2, 1, 2, 4, 3]


def test_case_0():
    data1 = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", decode=True)

    resize_op = vision.Resize((224, 224))

    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["target"])
    repeat_num = 4
    data1 = data1.repeat(repeat_num)
    batch_size = 2
    data1 = data1.batch(batch_size, drop_remainder=True)

    num = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num += 1
    assert num == 20


def test_case_1():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", decode=True)

    resize_op = vision.Resize((224, 224))

    data1 = data1.map(operations=resize_op, input_columns=["image"])
    repeat_num = 4
    data1 = data1.repeat(repeat_num)
    batch_size = 2
    data1 = data1.batch(batch_size, drop_remainder=True, pad_info={})

    num = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num += 1
    assert num == 18


def test_case_2():
    data1 = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", decode=True)
    sizes = [0.5, 0.5]
    randomize = False
    dataset1, dataset2 = data1.split(sizes=sizes, randomize=randomize)

    num_iter = 0
    for _ in dataset1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 5

    num_iter = 0
    for _ in dataset2.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 5


def test_voc_exception():
    try:
        data1 = ds.VOCDataset(DATA_DIR, task="InvalidTask", usage="train", decode=True)
        for _ in data1.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except ValueError:
        pass

    try:
        data2 = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", class_indexing={"cat": 0}, decode=True)
        for _ in data2.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except ValueError:
        pass

    try:
        data3 = ds.VOCDataset(DATA_DIR, task="Detection", usage="notexist", decode=True)
        for _ in data3.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except ValueError:
        pass

    try:
        data4 = ds.VOCDataset(DATA_DIR, task="Detection", usage="xmlnotexist", decode=True)
        for _ in data4.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError:
        pass

    try:
        data5 = ds.VOCDataset(DATA_DIR, task="Detection", usage="invalidxml", decode=True)
        for _ in data5.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError:
        pass

    try:
        data6 = ds.VOCDataset(DATA_DIR, task="Detection", usage="xmlnoobject", decode=True)
        for _ in data6.create_dict_iterator(num_epochs=1):
            pass
        assert False
    except RuntimeError:
        pass

    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False)
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["bbox"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["difficult"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["truncate"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", shuffle=False)
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", shuffle=False)
        data = data.map(operations=exception_func, input_columns=["target"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.VOCDataset(DATA_DIR, task="Segmentation", usage="train", shuffle=False)
        data = data.map(operations=vision.Decode(), input_columns=["target"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["target"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


def test_voc_num_classes():
    data1 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", shuffle=False, decode=True)
    assert data1.num_classes() is None

    class_index = {'car': 0, 'cat': 1, 'train': 5}
    data2 = ds.VOCDataset(DATA_DIR, task="Detection", usage="train", class_indexing=class_index, decode=True)
    assert data2.num_classes() is None


if __name__ == '__main__':
    test_voc_segmentation()
    test_voc_detection()
    test_voc_class_index()
    test_voc_get_class_indexing()
    test_case_0()
    test_case_1()
    test_case_2()
    test_voc_exception()
    test_voc_num_classes()
