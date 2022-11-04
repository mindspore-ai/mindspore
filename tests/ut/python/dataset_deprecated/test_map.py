# Copyright 2022 Huawei Technologies Co., Ltd
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
Test Map op in Dataset
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore.dataset.transforms import c_transforms
from mindspore.dataset.transforms import py_transforms
import mindspore.dataset.transforms.transforms as data_trans
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
import mindspore.dataset.vision.transforms as vision

DATA_DIR_PK = "../data/dataset/testPK/data"
DATA_DIR_VOCAB = "../data/dataset/testVocab/words.txt"


def test_map_c_transform_exception():
    """
    Feature: test c error op def
    Description: op defined like c_vision.HWC2CHW
    Expectation: success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR_PK, num_parallel_workers=1, shuffle=True)

    train_image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    random_crop_decode_resize_op = c_vision.RandomCropDecodeResize(train_image_size,
                                                                   scale=(0.08, 1.0),
                                                                   ratio=(0.75, 1.333))
    random_horizontal_flip_op = c_vision.RandomHorizontalFlip(prob=0.5)
    normalize_op = c_vision.Normalize(mean=mean, std=std)
    hwc2chw_op = c_vision.HWC2CHW  # exception

    data_set = data_set.map(operations=random_crop_decode_resize_op, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=random_horizontal_flip_op, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=normalize_op, input_columns="image", num_parallel_workers=1)
    with pytest.raises(ValueError) as info:
        data_set = data_set.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=1)
    assert "Parameter operations's element of method map should be a " in str(info.value)

    # compose exception
    with pytest.raises(ValueError) as info:
        c_transforms.Compose([
            c_vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            c_vision.RandomHorizontalFlip,
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()])
    assert " should be a " in str(info.value)

    # randomapply exception
    with pytest.raises(ValueError) as info:
        c_transforms.RandomApply([
            c_vision.RandomCropDecodeResize,
            c_vision.RandomHorizontalFlip(prob=0.5),
            c_vision.Normalize(mean=mean, std=std),
            c_vision.HWC2CHW()])
    assert " should be a " in str(info.value)

    # randomchoice exception
    with pytest.raises(ValueError) as info:
        c_transforms.RandomChoice([
            c_vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            c_vision.RandomHorizontalFlip(prob=0.5),
            c_vision.Normalize,
            c_vision.HWC2CHW()])
    assert " should be a " in str(info.value)


def test_map_py_transform_exception():
    """
    Feature: test python error op def
    Description: op defined like py_vision.RandomHorizontalFlip
    Expectation: success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR_PK, num_parallel_workers=1, shuffle=True)

    # define map operations
    decode_op = py_vision.Decode()
    random_horizontal_flip_op = py_vision.RandomHorizontalFlip  # exception
    to_tensor_op = py_vision.ToTensor()
    trans = [decode_op, random_horizontal_flip_op, to_tensor_op]

    with pytest.raises(ValueError) as info:
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    assert "Parameter operations's element of method map should be a " in str(info.value)

    # compose exception
    with pytest.raises(ValueError) as info:
        py_transforms.Compose([
            py_vision.Decode,
            py_vision.RandomHorizontalFlip(),
            py_vision.ToTensor()])
    assert " should be a " in str(info.value)

    # randomapply exception
    with pytest.raises(ValueError) as info:
        py_transforms.RandomApply([
            py_vision.Decode(),
            py_vision.RandomHorizontalFlip,
            py_vision.ToTensor()])
    assert " should be a " in str(info.value)

    # randomchoice exception
    with pytest.raises(ValueError) as info:
        py_transforms.RandomChoice([
            py_vision.Decode(),
            py_vision.RandomHorizontalFlip(),
            py_vision.ToTensor])
    assert " should be a " in str(info.value)


def test_map_text_and_data_transforms():
    """
    Feature: Map op
    Description: Test Map op with both Text Transforms and Data Transforms
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data = ds.TextFileDataset(DATA_DIR_VOCAB, shuffle=False)

    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None,
                                    special_tokens=["<pad>", "<unk>"],
                                    special_first=True)

    padend_op = c_transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    lookup_op = text.Lookup(vocab, "<unk>")

    # Use both Text Lookup op and Data Transforms PadEnd op in operations list for Map
    data = data.map(operations=[lookup_op, padend_op], input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(d["text"].item())
    assert res == [4, 5, 3, 6, 7, 2]


def test_map_mix_vision_transforms():
    """
    Feature: Map op
    Description: Test Map op with mixing of old legacy vision c/py_transforms and new unified vision transforms
    Expectation: RuntimeError is detected
    """

    def test_config(my_operations):
        # Not valid to mix legacy c/py_transforms with new unified transforms
        data_set = ds.ImageFolderDataset(DATA_DIR_PK, num_parallel_workers=1)
        data_set = data_set.map(operations=my_operations, input_columns="image")

        with pytest.raises(RuntimeError) as error_info:
            for _ in enumerate(data_set):
                pass
        assert "Mixing old legacy c/py_transforms and new unified transforms is not allowed" in str(error_info.value)

    # Test old legacy transform before new unified transform
    test_config([c_vision.Decode(), vision.RandomHorizontalFlip()])
    test_config([py_vision.Decode(), lambda x: x, vision.RandomHorizontalFlip()])

    # Test old legacy transform after new unified transform
    test_config([lambda x: x, vision.Decode(), c_vision.RandomHorizontalFlip(), c_vision.RandomVerticalFlip()])
    test_config([vision.Decode(True), py_vision.RandomHorizontalFlip(), py_vision.ToTensor()])


def test_map_mix_data_transforms():
    """
    Feature: Map op
    Description: Test Map op with mixing of old legacy data c/py_transforms and new unified data transforms
    Expectation: RuntimeError is detected
    """

    def test_config(my_operations):
        # Not valid to mix legacy c/py_transforms with new unified transforms
        data_set = ds.NumpySlicesDataset([1, 2, 3], column_names="x")

        data_set = data_set.map(operations=my_operations, input_columns="x")

        with pytest.raises(RuntimeError) as error_info:
            for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
                pass
        assert "Mixing old legacy c/py_transforms and new unified transforms is not allowed" in str(error_info.value)

    # Test old legacy transform before new unified transform
    test_config([c_transforms.Duplicate(), data_trans.Concatenate(), lambda x: x])

    # Test old legacy transform after new unified transform
    test_config([data_trans.Duplicate(), c_transforms.Concatenate()])


def test_map_operations1():
    """
    Feature: Map op
    Description: Test Map op with operations in multiple formats
    Expectation: Dataset pipeline runs successfully and results are verified
    """

    class RandomHorizontal(py_vision.RandomHorizontalFlip):
        def __init__(self, p):
            self.p = p
            super().__init__(p)

    data1 = ds.ImageFolderDataset(DATA_DIR_PK, num_samples=5)
    # Use 2 different formats to list ops for map operations
    data1 = data1.map(operations=[py_vision.Decode(),
                                  py_vision.RandomCrop(512),
                                  RandomHorizontal(0.5)], input_columns=["image"])

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        num_iter += 1
    assert num_iter == 5


def test_map_with_exact_log():
    """
    Feature: Map op
    Description: Python operation just print once log
    Expectation: Raise exact error info
    """

    class GetDatasetGenerator:
        """Get DatasetGenerator"""
        def __init__(self):
            np.random.seed(58)
            self.__data = np.random.sample((50, 2))
            self.__label = np.random.sample((50, 1))
            self.__label2 = np.random.sample((50, 1))
            self.__label3 = np.random.sample((50, 1))
            self.__label4 = np.random.sample((50, 1))

        def __getitem__(self, index):
            return (self.__data[index], self.__label[index], self.__label2[index],
                    self.__label3[index], self.__label4[index])

        def __len__(self):
            return len(self.__data)

    dataset_generator = GetDatasetGenerator()
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2", "label3", "label4"], shuffle=False)

    def pyfunc(x, y, z, m, n):
        return (x, y, z, m, n)

    dataset = dataset.map(operations=pyfunc, input_columns=["data", "label", "label2", "label3", "label4"])

    py_trans = [py_vision.Resize((388, 388))]
    dataset = dataset.map(operations=py_trans, input_columns=["data"])

    # output exact info without duplicate info
    with pytest.raises(RuntimeError) as info:
        for data in dataset.create_dict_iterator():
            print(data["data"], data["label"])
    print("-----{}++++".format(info.value), flush=True)
    assert str(info.value).count("Exception thrown from user defined Python function") == 1
    assert str(info.value).count("map operation") == 1
    assert str(info.value).count("img should be PIL image") == 1


if __name__ == '__main__':
    test_map_c_transform_exception()
    test_map_py_transform_exception()
    test_map_text_and_data_transforms()
    test_map_mix_vision_transforms()
    test_map_mix_data_transforms()
    test_map_operations1()
    test_map_with_exact_log()
