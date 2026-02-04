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
Used in test_map.py
"""

import time

import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset.vision import Inter

def run(num_workers):
    num_samples = 8

    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._data = np.fromfile("../data/dataset/apple.jpg", dtype=np.int8)
            self._label = np.zeros((5, 1))

        def __getitem__(self, index):
            return self._data, self._label[0]

        def __len__(self):
            return num_samples

    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label"], shuffle=False)

    def map1(data):
        decode = vision.Decode()(data)
        crop = vision.Crop((0, 0), 2000)(decode)
        resize = vision.Resize([1024, 768], Inter.BICUBIC)(crop)
        normalize = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])(resize)
        time.sleep(3)
        print("map time: 3.001")  # fixed time
        return (normalize,)

    dataset = dataset.map(operations=map1, input_columns=['data'], python_multiprocessing=True,
                          num_parallel_workers=num_workers)

    def per_batch_map(data, batch_info):
        start = time.time()
        time.sleep(3)
        print("batch time: {}".format(time.time() - start))
        return data

    dataset = dataset.batch(batch_size=2, per_batch_map=per_batch_map, input_columns=['data'],
                            python_multiprocessing=True, num_parallel_workers=num_workers)

    count = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        count += 1
    assert count == 4


if __name__ == '__main__':
    ds.config.set_multiprocessing_timeout_interval(2)
    run(2)
