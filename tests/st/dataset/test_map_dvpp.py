# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
import os
import sys

import cv2
import numpy as np
import pytest

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import transforms
from mindspore.dataset.vision import Inter

# pylint: disable=W0212
# W0212: protected-access


data_dir = "/home/workspace/mindspore_dataset/910B_dvpp/testImageNetData2/train"
result_data_dir = "/home/workspace/mindspore_dataset/910B_dvpp/testAscend910BDvpp"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_pyfunc_with_multi_op_process_mode():
    """
    Feature: Map op with pyfunc contains dvpp ops & cpu ops
    Description: Test map with dvpp resize operation
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    # can resolve tbe error when map with pyfun in process mode
    os.environ["MIN_COMPILE_RESOURCE_USAGE_CTRL"] = "ub_fusion,coretype_check,op_compile"

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # testcase2 : map with process mode
    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc2(img_bytes):
        ms.set_context(max_device_memory="2GB")

        length = len(img_bytes)
        print("image len: {}".format(length), flush=True)

        img_decode = vision.Decode().device("Ascend")(img_bytes)

        # resize(cpu)
        img_resize = vision.Resize(size=(64, 32))(img_decode)

        # normalize(dvpp)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
        return img_normalize

    # multi process mode
    data2 = data2.map(pyfunc2, input_columns="image", python_multiprocessing=True)
    for item in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_pyfunc_with_multi_op_thread_mode():
    """
    Feature: Map op with pyfunc contains dvpp ops & cpu ops
    Description: Test map with dvpp resize operation
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # testcase1 : map with thread mode
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)

    def pyfunc(img_bytes):
        length = len(img_bytes)
        print("image len: {}".format(length), flush=True)

        img_decode = vision.Decode().device("Ascend")(img_bytes)

        # resize(cpu)
        img_resize = vision.Resize(size=(64, 32))(img_decode)

        # normalize(dvpp)
        mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
        std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
        img_normalize = vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")(img_resize)
        return img_normalize

    # multi thread mode
    data1 = data1.map(pyfunc, input_columns="image")
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.float32


def map_with_dvpp_resize(num_workers=1, python_multiprocess=False):
    """
    Feature: Map op
    Description: Test map with dvpp resize operation
    Expectation: The result is equal to the expected
    """

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode(), input_columns="image", num_parallel_workers=num_workers)
    data1 = data1.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)

    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
        assert item[0].shape == (224, 224, 3)
        assert item[0].dtype == np.uint8
        print("count: {}".format(count), flush=True)
    assert count == 6

    class RandomAccessDataset:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 3), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    # interpolation is BILINEAR
    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset = dataset.map(vision.Resize([64, 32], interpolation=vision.Inter.BILINEAR).device("Ascend"),
                          input_columns="image", num_parallel_workers=num_workers,
                          python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.uint8
    assert count == 6

    # interpolation is NEAREST
    dataset2 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset2 = dataset2.map(vision.Resize([64, 32], interpolation=vision.Inter.NEAREST).device("Ascend"),
                            input_columns="image", num_parallel_workers=num_workers,
                            python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset2.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.uint8
    assert count == 6

    # interpolation is CUBIC
    dataset3 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset3 = dataset3.map(vision.Resize([64, 32], interpolation=vision.Inter.CUBIC).device("Ascend"),
                            input_columns="image", num_parallel_workers=num_workers,
                            python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset3.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.uint8
    assert count == 6

    # interpolation is BICUBIC
    dataset4 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset4 = dataset4.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image", num_parallel_workers=num_workers,
                            python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset4.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.uint8
    assert count == 6

    # the input is HW
    class RandomAccessDatasetHW:
        def __init__(self):
            self._data = np.ones((6, 224, 224), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    loader = RandomAccessDatasetHW()
    dataset5 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset5 = dataset5.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image", num_parallel_workers=num_workers,
                            python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset5.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32)
        assert item[0].dtype == np.uint8
    assert count == 6

    # the input is HW1
    class RandomAccessDatasetHW1:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 1), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    loader = RandomAccessDatasetHW1()
    dataset6 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset6 = dataset6.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image", num_parallel_workers=num_workers,
                            python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset6.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32)
        assert item[0].dtype == np.uint8
    assert count == 6

    # the input is 1HW1
    class RandomAccessDataset1HW1:
        def __init__(self):
            self._data = np.ones((6, 1, 224, 224, 1), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    loader = RandomAccessDataset1HW1()
    dataset7 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset7 = dataset7.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image", num_parallel_workers=num_workers,
                            python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset7.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32)
        assert item[0].dtype == np.uint8
    assert count == 6

    # the input is float HW3
    class RandomAccessDatasetHW3:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 3), dtype=np.float32)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW3()
    dataset11 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset11 = dataset11.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                              input_columns="image", num_parallel_workers=num_workers,
                              python_multiprocessing=python_multiprocess)
    count = 0
    for item in dataset11.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (64, 32, 3)
        assert item[0].dtype == np.float32
    assert count == 6


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_resize():
    """
    Feature: Map op
    Description: Test map with dvpp resize operation
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    map_with_dvpp_resize(1)
    map_with_dvpp_resize(3)
    map_with_dvpp_resize(8)
    map_with_dvpp_resize(1, True)
    map_with_dvpp_resize(3, True)
    map_with_dvpp_resize(8, True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_resize_mixed_op():
    """
    Feature: Map op
    Description: Test map with dvpp resize operation and mixed op
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(vision.Decode(), input_columns="image")
    data = data.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image")
    data = data.map(vision.HWC2CHW(), input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = check_img.transpose(2, 0, 1)

    # Expect to equal
    count = 0
    for item in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    # map with [decode, resize(dvpp)]
    data1 = data1.map([vision.Decode(), vision.Resize([224, 224]).device("Ascend")], input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)

    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # map with [decode, resize(dvpp), hwc2chw]
    # dataset
    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data2 = data2.map([vision.Decode(), vision.Resize([224, 224]).device("Ascend"), vision.HWC2CHW()],
                      input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = check_img.transpose(2, 0, 1)

    # Expect to equal
    count = 0
    for item in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # map with [resize(dvpp), hwc2chw]
    # dataset
    data3 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data3 = data3.map(vision.Decode(), input_columns="image")
    data3 = data3.map([vision.Resize([224, 224]).device("Ascend"), vision.HWC2CHW()], input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = check_img.transpose(2, 0, 1)

    # Expect to equal
    count = 0
    for item in data3.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # map with [decode, resize(dvpp), resize(dvpp), hwc2chw]
    # dataset
    data4 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data4 = data4.map([vision.Decode(), vision.Resize([224, 224]).device("Ascend"),
                       vision.Resize([64, 48]).device("Ascend"), vision.HWC2CHW()],
                      input_columns="image")
    # Expect to equal
    count = 0
    for item in data4.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (3, 64, 48)
    assert count == 6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_resize_with_exception():
    """
    Feature: Map op
    Description: Test map with dvpp resize operation when exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # dataset
    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image")

    # Expect to equal
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "the input tensor is not HW, HWC or 1HWC." in str(info.value)

    # the input is HW2
    class RandomAccessDatasetHW2:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 2), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset1 = dataset1.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is HW4
    class RandomAccessDatasetHW4:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW4()
    dataset2 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset2 = dataset2.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is 23HW4
    class RandomAccessDataset23HW4:
        def __init__(self):
            self._data = np.ones((6, 2, 3, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset23HW4()
    dataset3 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset3 = dataset3.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset3.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(info.value)

    # the input is 3HW1
    class RandomAccessDataset3HW1:
        def __init__(self):
            self._data = np.ones((6, 3, 224, 224, 1), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    loader = RandomAccessDataset3HW1()
    dataset8 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset8 = dataset8.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset8.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor NHWC should be 1HWC or HWC." in str(info.value)

    # the input is 3HW3
    class RandomAccessDataset3HW3:
        def __init__(self):
            self._data = np.ones((6, 3, 224, 224, 3), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset3HW3()
    dataset9 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset9 = dataset9.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset9.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor NHWC should be 1HWC or HWC." in str(info.value)

    # the input is 6HW3
    class RandomAccessDataset6HW3:
        def __init__(self):
            self._data = np.ones((6, 6, 224, 224, 3), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset6HW3()
    dataset10 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset10 = dataset10.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                              input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset10.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor NHWC should be 1HWC or HWC." in str(info.value)

    # the input is float 9HW3
    class RandomAccessDataset9HW3:
        def __init__(self):
            self._data = np.ones((6, 9, 224, 224, 3), dtype=np.float32)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset9HW3()
    dataset12 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset12 = dataset12.map(vision.Resize([64, 32], interpolation=vision.Inter.BICUBIC).device("Ascend"),
                              input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset12.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor NHWC should be 1HWC or HWC." in str(info.value)

    # dataset with interpolation=ANTIALIAS
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode(), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        _ = data1.map(vision.Resize([224, 224], interpolation=vision.Inter.ANTIALIAS).device("Ascend"),
                      input_columns="image")
    assert "Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST" in str(info.value)

    # dataset with interpolation=AREA
    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data2 = data2.map(vision.Decode(), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        _ = data2.map(vision.Resize([224, 224], interpolation=vision.Inter.AREA).device("Ascend"),
                      input_columns="image")
    assert "Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST" in str(info.value)

    # dataset with interpolation=PILCUBIC
    data3 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data3 = data3.map(vision.Decode(), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        _ = data3.map(vision.Resize([224, 224], interpolation=vision.Inter.PILCUBIC).device("Ascend"),
                      input_columns="image")
    assert "Invalid interpolation mode, only support BILINEAR, CUBIC and NEAREST" in str(info.value)

def map_with_dvpp_decode(num_workers=1, python_multiprocess=False):
    """
    Feature: Map op
    Description: Test map with dvpp decode operation
    Expectation: The result is equal to the expected
    """

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode().device("Ascend"), input_columns="image", num_parallel_workers=num_workers,
                      python_multiprocessing=python_multiprocess)
    data1 = data1.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image",
                      num_parallel_workers=num_workers)

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)

    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
        assert item[0].shape == (224, 224, 3)
        assert item[0].dtype == np.uint8
        print("count: {}".format(count), flush=True)
    assert count == 6


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_decode():
    """
    Feature: Map op
    Description: Test map with dvpp decode operation
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    map_with_dvpp_decode(1)
    map_with_dvpp_decode(3)
    map_with_dvpp_decode(8)
    map_with_dvpp_decode(1, True)
    map_with_dvpp_decode(3, True)
    map_with_dvpp_decode(8, True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_decode_with_pre_pyfun():
    """
    Feature: Map op
    Description: Test map with dvpp decode operation and with pre pyfunc
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    def pyfunc(data):
        return data
    data1 = data1.map([pyfunc, vision.Decode().device("Ascend"), vision.Resize([224, 224]).device("Ascend")],
                      input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)

    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
        assert item[0].shape == (224, 224, 3)
        assert item[0].dtype == np.uint8
        print("count: {}".format(count), flush=True)
    assert count == 6


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_decode_mixed_op():
    """
    Feature: Map op
    Description: Test map with dvpp decode operation and mixed op
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(vision.Decode().device("Ascend"), input_columns="image")
    data = data.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image")
    data = data.map(vision.HWC2CHW(), input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = check_img.transpose(2, 0, 1)

    # Expect to equal
    count = 0
    for item in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    # map with [decode(dvpp), resize(dvpp)]
    data1 = data1.map([vision.Decode().device("Ascend"), vision.Resize([224, 224]).device("Ascend")],
                      input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)

    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # map with [decode(dvpp), resize(dvpp), hwc2chw]
    # dataset
    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data2 = data2.map([vision.Decode().device("Ascend"), vision.Resize([224, 224]).device("Ascend"), vision.HWC2CHW()],
                      input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = check_img.transpose(2, 0, 1)

    # Expect to equal
    count = 0
    for item in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # map with [resize(dvpp), hwc2chw]
    # dataset
    data3 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data3 = data3.map(vision.Decode().device("Ascend"), input_columns="image")
    data3 = data3.map([vision.Resize([224, 224]).device("Ascend"), vision.HWC2CHW()], input_columns="image")

    check_img = cv2.imread(os.path.join(result_data_dir, "train/class1/1_1.jpg"))
    check_img = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
    check_img = check_img.transpose(2, 0, 1)

    # Expect to equal
    count = 0
    for item in data3.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert (check_img == item[0]).all()
    assert count == 6

    # map with [decode(dvpp), resize(dvpp), resize(dvpp), hwc2chw]
    # dataset
    data4 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data4 = data4.map([vision.Decode().device("Ascend"), vision.Resize([224, 224]).device("Ascend"),
                       vision.Resize([64, 48]).device("Ascend"), vision.HWC2CHW()],
                      input_columns="image")
    # Expect to equal
    count = 0
    for item in data4.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (3, 64, 48)
    assert count == 6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_decode_with_exception():
    """
    Feature: Map op
    Description: Test map with dvpp decode operation when exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # not 1D
    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(vision.Decode(), input_columns="image")
    data = data.map(vision.Decode().device("Ascend"), input_columns="image")

    # Expect to equal
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "Invalid data shape. Currently only support 1D." in str(info.value)

    # bmp
    class RandomAccessDatasetBMP:
        def __init__(self):
            self._data = np.expand_dims(np.fromfile(os.path.join(result_data_dir, "apple.bmp"), dtype=np.uint8),
                                        axis=0)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetBMP()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset1 = dataset1.map(vision.Decode().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "Invalid image type. Currently only support JPG." in str(info.value)

    # png
    class RandomAccessDatasetPNG:
        def __init__(self):
            self._data = np.expand_dims(np.fromfile(os.path.join(result_data_dir, "apple.png"), dtype=np.uint8),
                                        axis=0)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetPNG()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    dataset1 = dataset1.map(vision.Decode().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "Invalid image type. Currently only support JPG." in str(info.value)

    # dtype is float
    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(transforms.TypeCast(ms.int32), input_columns="image")
    data = data.map(vision.Decode().device("Ascend"), input_columns="image")
    # Expect to equal
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "Invalid image type. Currently only support JPG." in str(info.value)

    # invalid device_target
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    with pytest.raises(ValueError) as error_info:
        _ = data1.map(vision.Decode().device("Ascennd"), input_columns="image")
    assert "Input device_target is not within the valid set of ['CPU', 'Ascend']" in str(error_info.value)


def map_with_dvpp_normalize(num_workers=1, python_multiprocess=False):
    """
    Feature: Map op
    Description: Test map with dvpp normalize operation
    Expectation: The result is equal to the expected
    """

    # HWC
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode(), input_columns="image", num_parallel_workers=num_workers,
                      python_multiprocessing=python_multiprocess)
    data1 = data1.map(vision.Resize([224, 224]), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    data1 = data1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)


    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize([224, 224])
    normalize_op = vision.Normalize(mean=mean_vec, std=std_vec)
    data2 = data2.map(operations=[decode_op, resize_op, normalize_op], input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)

    # Expect to equal
    count = 0
    for item1, item2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                            data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        count += 1
        assert np.allclose(item1[0], item2[0], 1e-6)
        assert item1[0].shape == (224, 224, 3)
        assert item1[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # 1HWC
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode(), input_columns="image", num_parallel_workers=num_workers,
                      python_multiprocessing=python_multiprocess)
    data1 = data1.map(vision.Resize([224, 224]), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    def expand(img):
        img_out = np.expand_dims(img, axis=0)
        assert img_out.shape == (1, 224, 224, 3)
        return img_out
    data1 = data1.map(operations=expand, input_columns="image")
    data1 = data1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224, 3)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # CHW
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode(), input_columns="image", num_parallel_workers=num_workers,
                      python_multiprocessing=python_multiprocess)
    data1 = data1.map(vision.Resize([224, 224]), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    data1 = data1.map(vision.HWC2CHW(), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    data1 = data1.map(vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend"),
                      input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)


    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    decode_op = vision.Decode()
    resize_op = vision.Resize([224, 224])
    hwc2chw_op = vision.HWC2CHW()
    normalize_op = vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False)
    data2 = data2.map(operations=[decode_op, resize_op, hwc2chw_op, normalize_op], input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)

    # Expect to equal
    count = 0
    for item1, item2 in zip(data1.create_tuple_iterator(num_epochs=1, output_numpy=True),
                            data2.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        count += 1
        assert np.allclose(item1[0], item2[0], 1e-6)
        assert item1[0].shape == (3, 224, 224)
        assert item1[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # 1CHW
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data1 = data1.map(vision.Decode(), input_columns="image", num_parallel_workers=num_workers,
                      python_multiprocessing=python_multiprocess)
    data1 = data1.map(vision.Resize([224, 224]), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    data1 = data1.map(vision.HWC2CHW(), input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    def expand2(img):
        img_out = np.expand_dims(img, axis=0)
        assert img_out.shape == (1, 3, 224, 224)
        return img_out
    data1 = data1.map(operations=expand2, input_columns="image")
    data1 = data1.map(vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend"),
                      input_columns="image",
                      num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (3, 224, 224)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # HW
    class RandomAccessDatasetHW:
        def __init__(self):
            self._data = np.ones([6, 224, 224], dtype=np.uint8) * 221
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image",
                            num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # HW1
    class RandomAccessDatasetHW1Test:
        def __init__(self):
            self._data = np.ones([6, 224, 224, 1], dtype=np.uint8) * 221
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW1Test()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image",
                            num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # 1HW
    class RandomAccessDataset1HW:
        def __init__(self):
            self._data = np.ones([6, 1, 224, 224], dtype=np.uint8) * 221
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset1HW()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend"),
                            input_columns="image",
                            num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # 1HW1
    class RandomAccessDatasetHW1:
        def __init__(self):
            self._data = np.ones([6, 1, 224, 224, 1], dtype=np.uint8) * 221
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW1()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image",
                            num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # 11HW
    class RandomAccessDatasetHW2:
        def __init__(self):
            self._data = np.ones([6, 1, 1, 224, 224], dtype=np.uint8) * 221
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255]
    std_vec = [0.275 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend"),
                            input_columns="image",
                            num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6

    # float32 HWC
    class RandomAccessDatasetHWFloat32:
        def __init__(self):
            self._data = np.ones([6, 224, 224, 3], dtype=np.float32) * 221
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHWFloat32()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image",
                            num_parallel_workers=num_workers, python_multiprocessing=python_multiprocess)
    # Expect to equal
    count = 0
    for item in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].shape == (224, 224, 3)
        assert item[0].dtype == np.float32
        print("count: {}".format(count), flush=True)
    assert count == 6


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_normalize():
    """
    Feature: Map op
    Description: Test map with dvpp normalize operation
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    map_with_dvpp_normalize(1)
    map_with_dvpp_normalize(3)
    map_with_dvpp_normalize(8)
    map_with_dvpp_normalize(1, True)
    map_with_dvpp_normalize(3, True)
    map_with_dvpp_normalize(8, True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_normalize_mixed_op():
    """
    Feature: Map op
    Description: Test map with dvpp mixed operation
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(vision.Decode(), input_columns="image")
    data = data.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image")
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    data = data.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image")

    # Expect to equal
    count = 0
    for item in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].dtype == np.float32
        assert item[0].shape == (224, 224, 3)
    assert count == 6

    # dataset
    data1 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    # map with [decode(dvpp), resize(dvpp), normalize(dvpp)]
    data1 = data1.map([vision.Decode().device("Ascend"),
                       vision.Resize([224, 224]).device("Ascend"),
                       vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")], input_columns="image")

    # Expect to equal
    count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].dtype == np.float32
        assert item[0].shape == (224, 224, 3)
    assert count == 6

    # map with [decode(dvpp), resize(dvpp), normalize(dvpp), hwc2chw]
    # dataset
    data2 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data2 = data2.map([vision.Decode().device("Ascend"),
                       vision.Resize([224, 224]).device("Ascend"),
                       vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"),
                       vision.HWC2CHW()],
                      input_columns="image")

    # Expect to equal
    count = 0
    for item in data2.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].dtype == np.float32
        assert item[0].shape == (3, 224, 224)
    assert count == 6

    # map with [normalize(dvpp), hwc2chw]
    # dataset
    data3 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data3 = data3.map(vision.Decode().device("Ascend"), input_columns="image")
    data3 = data3.map(vision.Resize([224, 224]).device("Ascend"), input_columns="image")
    data3 = data3.map([vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), vision.HWC2CHW()],
                      input_columns="image")

    # Expect to equal
    count = 0
    for item in data3.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].dtype == np.float32
        assert item[0].shape == (3, 224, 224)
    assert count == 6

    # map with [decode(dvpp), resize(dvpp), hwc2chw, normalize(dvpp)]
    # dataset
    data4 = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data4 = data4.map([vision.Decode().device("Ascend"),
                       vision.Resize([64, 48]).device("Ascend"),
                       vision.HWC2CHW(),
                       vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")],
                      input_columns="image")
    # Expect to equal
    count = 0
    for item in data4.create_tuple_iterator(num_epochs=1, output_numpy=True):
        count += 1
        assert item[0].dtype == np.float32
        assert item[0].shape == (3, 64, 48)
    assert count == 6


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_normalize_exception():
    """
    Feature: Map op
    Description: Test map with dvpp normalize operation and exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # mean size is 3, input is HW
    class RandomAccessDatasetHW:
        def __init__(self):
            self._data = np.ones((1, 224, 224), dtype=np.uint8)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel is not equal to the size of mean or std." in str(info.value)

    # mean size is 2, input is HWC
    class RandomAccessDatasetHWC:
        def __init__(self):
            self._data = np.ones((1, 224, 224, 3), dtype=np.uint8)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHWC()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255]
    std_vec = [0.275 * 255, 0.267 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel is not equal to the size of mean or std." in str(info.value)

    # mean size is 2, input is CHW
    class RandomAccessDatasetCHW:
        def __init__(self):
            self._data = np.ones((1, 3, 224, 224), dtype=np.uint8)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetCHW()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255]
    std_vec = [0.275 * 255, 0.267 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel is not equal to the size of mean or std." in str(info.value)

    # HW3, but is_hwc=False
    class RandomAccessDatasetHWCIsHWC:
        def __init__(self):
            self._data = np.ones((1, 224, 224, 3), dtype=np.uint8)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHWCIsHWC()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255]
    std_vec = [0.275 * 255, 0.267 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [C,H,W] is not 1 or 3" in str(info.value)

    # 3HW, but is_hwc=True
    class RandomAccessDatasetCHW2:
        def __init__(self):
            self._data = np.ones((1, 3, 224, 224), dtype=np.uint8)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetCHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255]
    std_vec = [0.275 * 255, 0.267 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # float16
    class RandomAccessDatasetHWCFloat16:
        def __init__(self):
            self._data = np.ones((1, 224, 224, 3), dtype=np.float16)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHWCFloat16()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input data is not uint8 or float32." in str(info.value)

    # int32
    class RandomAccessDatasetHWCInt32:
        def __init__(self):
            self._data = np.ones((1, 224, 224, 3), dtype=np.int32)
            self._label = np.zeros((1,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHWCInt32()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    dataset1 = dataset1.map(vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input data is not uint8 or float32." in str(info.value)

    # resize(dvpp), normalize(dvpp, is_hwc=False)
    data = ds.ImageFolderDataset(dataset_dir=data_dir, shuffle=False)
    data = data.map(vision.Decode(), input_columns="image")
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    data = data.map([vision.Resize([224, 224]).device("Ascend"),
                     vision.Normalize(mean=mean_vec, std=std_vec, is_hwc=False).device("Ascend")],
                    input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in data.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
        assert count == 6
    assert "The input data's channel is not 3 or 1." in str(info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_horizontal_flip_with_exception():
    """
    Feature: Map op
    Description: Test map with dvpp horizontal flip operation when exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # the input is HW2
    class RandomAccessDatasetHW2:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 2), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset1 = dataset1.map(vision.HorizontalFlip().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is HW4
    class RandomAccessDatasetHW4:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW4()
    dataset2 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset2 = dataset2.map(vision.HorizontalFlip().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is 23HW4
    class RandomAccessDataset23HW4:
        def __init__(self):
            self._data = np.ones((6, 2, 3, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset23HW4()
    dataset3 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset3 = dataset3.map(vision.HorizontalFlip().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset3.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_vertical_flip_with_exception():
    """
    Feature: Map op
    Description: Test map with dvpp vertical flip operation when exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # the input is HW2
    class RandomAccessDatasetHW2:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 2), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset1 = dataset1.map(vision.VerticalFlip().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is HW4
    class RandomAccessDatasetHW4:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW4()
    dataset2 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset2 = dataset2.map(vision.VerticalFlip().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is 23HW4
    class RandomAccessDataset23HW4:
        def __init__(self):
            self._data = np.ones((6, 2, 3, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset23HW4()
    dataset3 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset3 = dataset3.map(vision.VerticalFlip().device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset3.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_resize_crop_with_exception():
    """
    Feature: Map op
    Description: Test map with dvpp resize crop operation when exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    # the input is HW2
    class RandomAccessDatasetHW2:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 2), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset1 = dataset1.map(vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is HW4
    class RandomAccessDatasetHW4:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW4()
    dataset2 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset2 = dataset2.map(vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is 23HW4
    class RandomAccessDataset23HW4:
        def __init__(self):
            self._data = np.ones((6, 2, 3, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset23HW4()
    dataset3 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset3 = dataset3.map(vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend"), input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset3.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_perspective_with_exception():
    """
    Feature: Map op
    Description: Test map with dvpp perspective operation when exception
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    print("Run testcase: " + sys._getframe().f_code.co_name)

    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]

    # the input is HW2
    class RandomAccessDatasetHW2:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 2), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW2()
    dataset1 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset1 = dataset1.map(vision.Perspective(start_points, end_points).device("Ascend").device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset1.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is HW4
    class RandomAccessDatasetHW4:
        def __init__(self):
            self._data = np.ones((6, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDatasetHW4()
    dataset2 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset2 = dataset2.map(vision.Perspective(start_points, end_points).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset2.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The channel of the input tensor of shape [H,W,C] is not 1, 3" in str(info.value)

    # the input is 23HW4
    class RandomAccessDataset23HW4:
        def __init__(self):
            self._data = np.ones((6, 2, 3, 224, 224, 4), dtype=np.uint8)
            self._label = np.zeros((6,))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)
    loader = RandomAccessDataset23HW4()
    dataset3 = ds.GeneratorDataset(source=loader, column_names=["image", "label"])

    dataset3 = dataset3.map(vision.Perspective(start_points, end_points).device("Ascend"),
                            input_columns="image")
    with pytest.raises(RuntimeError) as info:
        count = 0
        for _ in dataset3.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "The input tensor is not of shape [H,W], [H,W,C] or [N,H,W,C]." in str(info.value)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_map_with_dvpp_shape_and_type():
    """
    Feature: Map op
    Description: Test map with dvpp output_shapes and output_types
    Expectation: The result is equal to the expected
    """
    ms.set_context(device_target="Ascend")

    data = np.random.randint(0, 255, size=(1, 100, 100, 3)).astype(np.uint8)
    resize_op = vision.Resize([100, 75], Inter.BICUBIC).device("Ascend")
    crop_op = vision.Crop((0, 0), (100, 75)).device("Ascend")
    transforms_list = [resize_op, crop_op]
    numpy_slices_dataset = ds.NumpySlicesDataset(data, ["image"])
    numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms_list, input_columns=["image"])
    assert numpy_slices_dataset.output_shapes() == [[100, 75, 3]]
    assert numpy_slices_dataset.output_types() == ['uint8']

    transforms_list2 = [vision.Resize([100, 75], Inter.BICUBIC),
                        vision.ConvertColor(vision.ConvertMode.COLOR_BGR2RGBA).device("Ascend")]
    numpy_slices_dataset2 = ds.NumpySlicesDataset(data, ["image"])
    numpy_slices_dataset2 = numpy_slices_dataset2.map(operations=transforms_list2, input_columns=["image"])
    assert numpy_slices_dataset2.output_shapes() == [[100, 75, 4]]
    assert numpy_slices_dataset2.output_types() == ['uint8']
    for item in numpy_slices_dataset2.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert item["image"].shape == (100, 75, 4)


if __name__ == '__main__':
    test_map_with_pyfunc_with_multi_op_process_mode()
    test_map_with_pyfunc_with_multi_op_thread_mode()
    test_map_with_dvpp_resize()
    test_map_with_dvpp_resize_mixed_op()
    test_map_with_dvpp_resize_with_exception()
    test_map_with_dvpp_decode()
    test_map_with_dvpp_decode_mixed_op()
    test_map_with_dvpp_decode_with_exception()
    test_map_with_dvpp_decode_with_pre_pyfun()
    test_map_with_dvpp_normalize()
    test_map_with_dvpp_normalize_mixed_op()
    test_map_with_dvpp_normalize_exception()
    test_map_with_dvpp_horizontal_flip_with_exception()
    test_map_with_dvpp_vertical_flip_with_exception()
    test_map_with_dvpp_resize_crop_with_exception()
    test_map_with_dvpp_perspective_with_exception()
    test_map_with_dvpp_shape_and_type()
