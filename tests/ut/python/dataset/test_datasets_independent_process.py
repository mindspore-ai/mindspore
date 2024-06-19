# Copyright 2024 Huawei Technologies Co., Ltd
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
import copy
import os
import time
import pytest

import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms


apple_jpg = "../data/dataset/apple.jpg"


@pytest.mark.skip(reason="timeout")
def test_dataset_with_independent_process():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._label = np.zeros((1), dtype=np.uint32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)

        def __getitem__(self, index):
            return self.image, self._label, np.array("./abcdefg.jpg")

        def __len__(self):
            return 10

    def PyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((224, 224))(img)
        img = vision.Rescale(1.0 / 255.0, 0.0)(img)
        img = vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        return img

    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label", "file_name"])
    dataset = dataset.map(operations=PyFunc, input_columns=["data"], num_parallel_workers=4,
                          python_multiprocessing=True)
    dataset = dataset.batch(batch_size=2)

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    assert dataset.get_dataset_size() == 5
    assert dataset.output_shapes() == [[2, 224, 224, 3], [2, 1], [2,]]
    assert dataset.output_types()[0:2] == [np.float32, np.uint32]
    assert dataset.get_col_names() == ["data", "label", "file_name"]
    assert dataset.get_batch_size() == 2
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    for _ in range(epochs):
        for item in ds_iter:
            assert len(item["file_name"]) == 2
            assert item["file_name"][0] == np.array("./abcdefg.jpg")
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()
        epoch += 1
    assert count == 15
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


@pytest.mark.skip(reason="timeout")
def test_dataset_with_independent_process_dynamic_shape():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with dynamic shape
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self, diff_shapes):
            self._label = np.zeros((1), dtype=np.float32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)
            self.sizes = diff_shapes

        def __getitem__(self, index):
            img = vision.Decode()(self.image)
            img = vision.Resize(self.sizes[index % 5])(img)
            return img, self._label

        def __len__(self):
            return 10

    diff_shapes = [(548, 506), (778, 578), (1024, 700), (1358, 734), (1570, 882)]
    loader = RandomAccessDataset(diff_shapes)
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label"])

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    shapes_count = [0, 0, 0, 0, 0]
    assert dataset.get_dataset_size() == 10
    shapes = dataset.output_shapes()
    assert tuple(shapes[0][0:2]) in diff_shapes
    assert shapes[1] == [1]
    assert dataset.output_types() == [np.uint8, np.float32]
    assert dataset.get_col_names() == ["data", "label"]
    assert dataset.get_batch_size() == 1
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    for _ in range(epochs):
        for item in ds_iter:
            shapes_count[diff_shapes.index(item["data"].shape[0:2])] += 1
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()
        epoch += 1
    assert len(np.unique(np.array(shapes_count))) == 1
    assert shapes_count[0] == 6
    assert sum(shapes_count) == 30
    assert count == 30
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


@pytest.mark.skip(reason="timeout")
def test_dataset_with_independent_process_train_and_eval():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with train and eval dataset
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    class TrainUDFDataset:
        def __init__(self):
            self._label = np.zeros((1), dtype=np.uint32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)

        def __getitem__(self, index):
            return self.image, self._label

        def __len__(self):
            return 10

    class EvalUDFDataset:
        def __init__(self):
            self._label = np.zeros((2), dtype=np.float32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)

        def __getitem__(self, index):
            return self.image, self._label

        def __len__(self):
            return 4

    def TrainPyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((224, 224))(img)
        img = vision.Rescale(1.0 / 255.0, 0.0)(img)
        img = vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        return img

    def EvalPyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((187, 187))(img)
        return img

    dataset = ds.GeneratorDataset(source=TrainUDFDataset(), column_names=["data", "label"])
    dataset = dataset.map(operations=TrainPyFunc, input_columns=["data"], num_parallel_workers=4,
                          python_multiprocessing=True)
    dataset = dataset.batch(batch_size=2)

    dataset2 = ds.GeneratorDataset(source=EvalUDFDataset(), column_names=["data", "label"], shuffle=False)
    dataset2 = dataset2.map(operations=EvalPyFunc, input_columns=["data"], num_parallel_workers=4,
                            python_multiprocessing=True)

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    assert dataset.get_dataset_size() == 5
    assert dataset.output_shapes() == [[2, 224, 224, 3], [2, 1]]
    assert dataset.output_types() == [np.float32, np.uint32]
    assert dataset.get_col_names() == ["data", "label"]
    assert dataset.get_batch_size() == 2
    assert dataset2.get_dataset_size() == 4
    assert dataset2.output_shapes() == [[187, 187, 3], [2]]
    assert dataset2.output_types() == [np.uint8, np.float32]
    assert dataset2.get_col_names() == ["data", "label"]
    assert dataset2.get_batch_size() == 1
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    # train process
    for _ in range(epochs):
        for item in ds_iter:
            assert item['data'].shape == (2, 224, 224, 3)
            assert item['data'].dtype == np.float32
            assert item['label'][0] == np.zeros((1), dtype=np.uint32)
            assert item['label'].shape == (2, 1)
            assert item['label'].dtype == np.uint32
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()

            # eval process
            if count % 100 == 0:
                ds_iter2 = dataset2.create_dict_iterator(output_numpy=True, num_epochs=1)
                count2 = 0
                for item2 in ds_iter2:
                    assert item2['data'].shape == (187, 187, 3)
                    assert item2['data'].dtype == np.uint8
                    assert item['label'][0] == np.zeros((2), dtype=np.float32)
                    assert item['label'].shape == (1)
                    assert item['label'].dtype == np.float32
                    print("count2: {}".format(count2), flush=True)
                    count2 += 1
                assert count2 == 4
        epoch += 1
    assert count == 15
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


def print_psutil(name):
    print("============== {} =============".format(name), flush=True)
    os.system("ps -ef | grep python")


@pytest.mark.skip(reason="timeout")
def test_dataset_with_independent_process_two_stage_pipeline():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with two stage pipeline
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    class FristUDFDataset:
        def __init__(self):
            self._image = np.fromfile(apple_jpg, dtype=np.int8)
            self._label = np.zeros((1), dtype=np.uint32)

        def __getitem__(self, index):
            return self._image, self._label

        def __len__(self):
            return 10

    def FirstPyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((300, 300))(img)
        return img

    first_dataset = ds.GeneratorDataset(source=FristUDFDataset(), column_names=["data", "label"])
    first_dataset = first_dataset.map(operations=FirstPyFunc, input_columns=["data"], num_parallel_workers=4,
                                      python_multiprocessing=True)

    class SecondUDFDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            self.dataset_size = self.dataset.get_dataset_size()
            self.iterator = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

        def __next__(self):
            data = next(self.iterator)
            assert data["data"].shape == (300, 300, 3)
            assert data["data"].dtype == np.uint8
            assert data["label"].shape == (1,)
            assert data["label"].dtype == np.uint32
            return data["data"], data["label"]

        def __iter__(self):
            self.iterator = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
            return self

        def __len__(self):
            return self.dataset_size

    def SecondPyFunc(img):
        img = vision.Resize((64, 64))(img)
        img = vision.Rescale(1.0 / 255.0, 0.0)(img)
        img = vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        return img

    second_dataset = ds.GeneratorDataset(source=SecondUDFDataset(first_dataset), column_names=["data", "label"],
                                         shuffle=False)
    second_dataset = second_dataset.map(operations=SecondPyFunc, input_columns=["data"], num_parallel_workers=2)
    second_dataset = second_dataset.map(operations=transforms.TypeCast(mstype.float32), input_columns=["label"])
    print_psutil("init")

    assert second_dataset.get_dataset_size() == 10
    print_psutil("after dataset_size")
    # TODO: hung with output_shapes & output_types
    ## assert second_dataset.output_shapes() == [[64, 64, 3], [1]]
    ## print_psutil("after shapes")

    ## assert second_dataset.output_types() == [np.float32, np.float32]
    ## print_psutil("after types")

    assert second_dataset.get_col_names() == ["data", "label"]
    print_psutil("after col_names")

    assert second_dataset.get_batch_size() == 1
    print_psutil("batch_size")

    count = 0
    epochs = 3
    epoch = 0
    ds_iter = second_dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    print_psutil("after iterator")
    for _ in range(epochs):
        for item in ds_iter:
            print_psutil("after get item")
            print("epoch: {}, count: {}".format(epoch, count), flush=True)
            assert item['data'].shape == (64, 64, 3)
            assert item['data'].dtype == np.float32
            assert item['label'].dtype == np.float32
            count += 1
        epoch += 1
    assert count == 30
    assert epoch == 3
    print_psutil("end")
    del os.environ["MS_INDEPENDENT_DATASET"]


@pytest.mark.skip(reason="timeout")
def test_dataset_with_independent_process_with_dict():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with python dict
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    diff_shapes = [(548, 507), (778, 577), (1024, 700), (1359, 733), (1570, 882)]
    python_dict = {"filename": "1.jpg", "object": {"truncated": 0, "difficult": 1}, "bndbox": [1, 2, 3, 4]}
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._label = np.zeros((1), dtype=np.float32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)
            self.sizes = diff_shapes
            self.attr = python_dict

        def __getitem__(self, index):
            img = vision.Decode()(self.image)
            img = vision.Resize(self.sizes[index % 5])(img)
            return self.attr, img, self._label

        def __len__(self):
            return 10

    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["attr", "data", "label"], shuffle=False)
    def add_new_dict(old_dict):
        new_dict = copy.deepcopy(old_dict)
        new_dict["class"] = "cat"
        return old_dict, new_dict
    dataset = dataset.map(operations=add_new_dict, input_columns=["attr"], output_columns=["attr", "attr2"],
                          num_parallel_workers=2, python_multiprocessing=True)

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    shapes_count = [0, 0, 0, 0, 0]
    assert dataset.get_dataset_size() == 10
    assert dataset.output_shapes() == [[0], [0], [548, 507, 3], [1]]
    assert dataset.output_types() == [np.dtype(object), np.dtype(object), np.uint8, np.float32]
    assert dataset.get_col_names() == ["attr", "attr2", "data", "label"]
    assert dataset.get_batch_size() == 1
    new_dict = copy.deepcopy(python_dict)
    new_dict["class"] = "cat"
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    for _ in range(epochs):
        for item in ds_iter:
            shapes_count[diff_shapes.index(item["data"].shape[0:2])] += 1
            assert isinstance(item["attr"], dict)
            assert item["attr"] == python_dict
            assert isinstance(item["attr2"], dict)
            assert item["attr2"] == new_dict
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()
        epoch += 1
    assert len(np.unique(np.array(shapes_count))) == 1
    assert shapes_count[0] == 6
    assert sum(shapes_count) == 30
    assert count == 30
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


if __name__ == "__main__":
    test_dataset_with_independent_process()
    test_dataset_with_independent_process_dynamic_shape()
    test_dataset_with_independent_process_train_and_eval()
    test_dataset_with_independent_process_two_stage_pipeline()
    test_dataset_with_independent_process_with_dict()
