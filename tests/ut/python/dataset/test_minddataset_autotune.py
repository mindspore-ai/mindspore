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
Testing Autotune support in DE for MindDataset
"""
import os
import pytest
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from util_minddataset import add_and_remove_cv_file


# pylint: disable=unused-variable, redefined-outer-name
@pytest.mark.forked
def test_autotune_simple_pipeline_mindrecord(add_and_remove_cv_file):
    """
    Feature: Autotuning
    Description: Test simple pipeline of autotune - MindDataset -> Map -> Batch -> Repeat
    Expectation: Pipeline runs successfully
    """
    ds.config.set_enable_autotune(True)

    columns_list = ["data", "label"]
    num_readers = 1
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    data1 = ds.MindDataset(file_name + "0", columns_list, num_readers)
    assert data1.get_dataset_size() == 10

    decode_op = vision.Decode()
    data1 = data1.map(
        input_columns=["data"], operations=decode_op, num_parallel_workers=2)
    resize_op = vision.Resize((32, 32), interpolation=Inter.LINEAR)
    data1 = data1.map(operations=resize_op, input_columns=["data"],
                      num_parallel_workers=2)
    data1 = data1.batch(2)
    data1 = data1.repeat(20)

    i = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        i += 1
    assert i == 100

    ds.config.set_enable_autotune(False)


if __name__ == '__main__':
    test_autotune_simple_pipeline_mindrecord(add_and_remove_cv_file)
