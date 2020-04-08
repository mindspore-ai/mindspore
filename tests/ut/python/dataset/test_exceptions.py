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
import pytest

import mindspore.dataset as ds

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def skip_test_exception():
    ds.config.set_num_parallel_workers(1)
    data = ds.TFRecordDataset(DATA_DIR, columns_list=["image"])
    data = data.map(input_columns=["image"], operations=vision.Resize(100, 100))
    with pytest.raises(RuntimeError) as info:
        data.create_tuple_iterator().get_next()
    assert "The shape size 1 of input tensor is invalid" in str(info.value)


def test_sample_exception():
    num_samples = 0
    with pytest.raises(ValueError) as info:
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], num_samples=num_samples)
    assert "num_samples must be greater than 0" in str(info.value)
    num_samples = -1
    with pytest.raises(ValueError) as info:
        data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], num_samples=num_samples)
    assert "num_samples must be greater than 0" in str(info.value)
    num_samples = 1
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], num_samples=num_samples)
    data = data.map(input_columns=["image"], operations=vision.Decode())
    data = data.map(input_columns=["image"], operations=vision.Resize((100, 100)))
    num_iters = 0
    for item in data.create_dict_iterator():
        num_iters += 1
    assert num_iters == 1

if __name__ == '__main__':
    test_exception()
