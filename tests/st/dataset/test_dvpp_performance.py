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
"""Test Dvpp Performance, should run it in directory [tests/st/dataset] by `python test_dvpp_performance.py`"""

import numpy as np
import time

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision


ms.set_context(device_target="Ascend")


input_apple_jpg = "../../ut/data/dataset/apple.jpg"


def run_transform_which_input_image_bytes(transform):
    """the input transform must start from decode"""
    transform_name, dvpp_transform, cpu_transform = transform

    # eager mode
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    _ = dvpp_transform(img_bytes)

    start = time.time()
    for _ in range(1000):
        _ = dvpp_transform(img_bytes)
    print("Run dvpp [{}] 1000 times in eager mode, cost: {}".format(transform_name, time.time() - start), flush=True)

    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    _ = cpu_transform(img_bytes)

    start = time.time()
    for _ in range(1000):
        _ = cpu_transform(img_bytes)
    print("Run cpu [{}] 1000 times in eager mode, cost: {}".format(transform_name, time.time() - start), flush=True)

    # pipeline mode
    class RandomAccessDataset:
        def __init__(self):
            self._data = np.fromfile(input_apple_jpg, dtype=np.uint8)

        def __getitem__(self, index):
            return (self._data,)

        def __len__(self):
            return 1000

    loader1 = RandomAccessDataset()
    data1 = ds.GeneratorDataset(source=loader1, column_names=["image"], shuffle=False)
    data1 = data1.map(dvpp_transform, input_columns="image")

    start = time.time()
    count = 0
    data_iter1 = data1.create_tuple_iterator(num_epochs=1, output_numpy=True)
    for _ in data_iter1:
        count += 1
    print("Run dvpp [{}] 1000 times in pipeline mode, cost: {}".format(transform_name, time.time() - start), flush=True)
    assert count == 1000

    loader2 = RandomAccessDataset()
    data2 = ds.GeneratorDataset(source=loader2, column_names=["image"], shuffle=False)
    data2 = data2.map(cpu_transform, input_columns="image")

    start = time.time()
    count = 0
    data_iter2 = data2.create_tuple_iterator(num_epochs=1, output_numpy=True)
    for _ in data_iter2:
        count += 1
    print("Run cpu [{}] 1000 times in pipeline mode, cost: {}".format(transform_name, time.time() - start), flush=True)
    assert count == 1000


def perf_transforms_which_input_image_bytes():
    mean_vec = [0.475 * 255, 0.451 * 255, 0.392 * 255]
    std_vec = [0.275 * 255, 0.267 * 255, 0.278 * 255]
    transform_list = [["Decode",
                       vision.Decode().device("Ascend"),
                       vision.Decode()],
                      ["Compose(Decode, Resize, Normalize)",
                       transforms.Compose([vision.Decode().device("Ascend"),
                                           vision.Resize(224).device("Ascend"),
                                           vision.Normalize(mean=mean_vec, std=std_vec).device("Ascend")]),
                       transforms.Compose([vision.Decode(),
                                           vision.Resize(224),
                                           vision.Normalize(mean=mean_vec, std=std_vec)])]]

    for transform in transform_list:
        print("=======================================================================")
        run_transform_which_input_image_bytes(transform)


def run_transform_which_input_hwc(transform):
    """the input of the transform must be hwc"""
    transform_name, dvpp_transform, cpu_transform = transform

    # eager mode
    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode()(img_bytes)
    _ = dvpp_transform(img_decode)

    start = time.time()
    for _ in range(1000):
        _ = dvpp_transform(img_decode)
    print("Run dvpp [{}] 1000 times in eager mode, cost: {}".format(transform_name, time.time() - start), flush=True)

    img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
    img_decode = vision.Decode()(img_bytes)
    _ = cpu_transform(img_decode)

    start = time.time()
    for _ in range(1000):
        _ = cpu_transform(img_decode)
    print("Run cpu [{}] 1000 times in eager mode, cost: {}".format(transform_name, time.time() - start), flush=True)

    # pipeline mode
    class RandomAccessDataset:
        def __init__(self):
            img_bytes = np.fromfile(input_apple_jpg, dtype=np.uint8)
            self._data = vision.Decode()(img_bytes)

        def __getitem__(self, index):
            return (self._data,)

        def __len__(self):
            return 1000
    loader1 = RandomAccessDataset()
    data1 = ds.GeneratorDataset(source=loader1, column_names=["image"], shuffle=False)
    data1 = data1.map(dvpp_transform, input_columns="image")

    start = time.time()
    count = 0
    data_iter1 = data1.create_tuple_iterator(num_epochs=1, output_numpy=True)
    for _ in data_iter1:
        count += 1
    print("Run dvpp [{}] 1000 times in pipeline mode, cost: {}".format(transform_name, time.time() - start), flush=True)
    assert count == 1000

    loader2 = RandomAccessDataset()
    data2 = ds.GeneratorDataset(source=loader2, column_names=["image"], shuffle=False)
    data2 = data2.map(cpu_transform, input_columns="image")

    start = time.time()
    count = 0
    data_iter2 = data2.create_tuple_iterator(num_epochs=1, output_numpy=True)
    for _ in data_iter2:
        count += 1
    print("Run cpu [{}] 1000 times in pipeline mode, cost: {}".format(transform_name, time.time() - start), flush=True)
    assert count == 1000


def perf_transforms_which_input_hwc():
    transform_list = [["Resize",
                       vision.Resize(224).device("Ascend"),
                       vision.Resize(224)],
                      ["Normalize",
                       vision.Normalize(mean=[0.475 * 255, 0.451 * 255, 0.392 * 255],
                                        std=[0.275 * 255, 0.267 * 255, 0.278 * 255]).device("Ascend"),
                       vision.Normalize(mean=[0.475 * 255, 0.451 * 255, 0.392 * 255],
                                        std=[0.275 * 255, 0.267 * 255, 0.278 * 255])],
                      ["ResizedCrop",
                       vision.ResizedCrop(0, 0, 128, 128, (100, 75)).device("Ascend"),
                       vision.ResizedCrop(0, 0, 128, 128, (100, 75))],
                      ["HorizontalFlip",
                       vision.HorizontalFlip().device("Ascend"),
                       vision.HorizontalFlip()],
                      ["VerticalFlip",
                       vision.VerticalFlip().device("Ascend"),
                       vision.VerticalFlip()],
                      ["AdjustBrightness",
                       vision.AdjustBrightness(2.0).device("Ascend"),
                       vision.AdjustBrightness(2.0)],
                      ["AdjustContrast",
                       vision.AdjustContrast(2.0).device("Ascend"),
                       vision.AdjustContrast(2.0)],
                      ["AdjustHue",
                       vision.AdjustHue(0.5).device("Ascend"),
                       vision.AdjustHue(0.5)],
                      ["AdjustSaturation",
                       vision.AdjustSaturation(2.0).device("Ascend"),
                       vision.AdjustSaturation(2.0)],
                      ["Perspective",
                       vision.Perspective(start_points=[[0, 63], [63, 63], [63, 0], [0, 0]],
                                          end_points=[[0, 32], [32, 32], [32, 0], [0, 0]]).device("Ascend"),
                       vision.Perspective(start_points=[[0, 63], [63, 63], [63, 0], [0, 0]],
                                          end_points=[[0, 32], [32, 32], [32, 0], [0, 0]])],
                      ["Pad",
                       vision.Pad([100, 100, 100, 100]).device("Ascend"),
                       vision.Pad([100, 100, 100, 100])],
                      ["Crop",
                       vision.Crop((0, 0), (100, 75)).device("Ascend"),
                       vision.Crop((0, 0), (100, 75))],
                      ["Affine",
                       vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1]).device("Ascend"),
                       vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1, shear=[1, 1])],
                      ["GaussianBlur",
                       vision.GaussianBlur(3, 3).device("Ascend"),
                       vision.GaussianBlur(3, 3)]]

    for transform in transform_list:
        print("=======================================================================")
        run_transform_which_input_hwc(transform)


if __name__ == "__main__":
    perf_transforms_which_input_image_bytes()
    perf_transforms_which_input_hwc()
