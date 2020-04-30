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

DATA_FILE = "../data/dataset/test_flat_map/images1.txt"
INDEX_FILE = "../data/dataset/test_flat_map/image_index.txt"


def test_flat_map_1():
    '''
    DATA_FILE records the path of image folders, load the images from them.
    '''
    import mindspore.dataset.transforms.text.utils as nlp

    def flat_map_func(x):
        data_dir = x[0].item().decode('utf8')
        d = ds.ImageFolderDatasetV2(data_dir)
        return d

    data = ds.TextFileDataset(DATA_FILE)
    data = data.flat_map(flat_map_func)

    count = 0
    for d in data:
        assert isinstance(d[0], np.ndarray)
        count += 1
    assert count == 52


def test_flat_map_2():
    '''
    Flatten 3D structure data
    '''
    import mindspore.dataset.transforms.text.utils as nlp

    def flat_map_func_1(x):
        data_dir = x[0].item().decode('utf8')
        d = ds.ImageFolderDatasetV2(data_dir)
        return d

    def flat_map_func_2(x):
        text_file = x[0].item().decode('utf8')
        d = ds.TextFileDataset(text_file)
        d = d.flat_map(flat_map_func_1)
        return d

    data = ds.TextFileDataset(INDEX_FILE)
    data = data.flat_map(flat_map_func_2)

    count = 0
    for d in data:
        assert isinstance(d[0], np.ndarray)
        count += 1
    assert count == 104


if __name__ == "__main__":
    test_flat_map_1()
    test_flat_map_2()
