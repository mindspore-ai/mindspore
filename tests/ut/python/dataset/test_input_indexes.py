# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Test Input Indexes
"""

import mindspore.dataset as ds
from mindspore import log as logger

def test_basics_input_indexes():
    """
    Feature: input_indexs
    Description: Test input_indexs with basic usage
    Expectation: Output is equal to the expected output
    """
    logger.info("test_basics_input_indexes")
    data = ds.NumpySlicesDataset([1, 2, 3], column_names=["col_1"])
    assert data.input_indexs == ()
    data.input_indexs = 10
    assert data.input_indexs == 10
    data = data.shuffle(2)
    assert data.input_indexs == 10
    data = data.project(["col_1"])
    assert data.input_indexs == 10

    data2 = ds.NumpySlicesDataset([1, 2, 3], column_names=["col_1"])
    assert data2.input_indexs == ()
    data2 = data2.shuffle(2)
    assert data2.input_indexs == ()
    data2 = data2.project(["col_1"])
    assert data2.input_indexs == ()
    data2.input_indexs = 20
    assert data2.input_indexs == 20

    data3 = data + data2
    assert data3.input_indexs == 10

if __name__ == '__main__':
    test_basics_input_indexes()
