# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from util import config_get_set_seed, config_get_set_num_parallel_workers


# Generate 1d int numpy array from 0 - 63
def generator_1d():
    for i in range(4):
        yield (np.array([i]),)


def test_case_0():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator without explicit kwargs for input args
    Expectation: Output is the same as expected output
    """
    original_seed = config_get_set_seed(55)
    original_num_parallel_workers = config_get_set_num_parallel_workers(1)

    # apply dataset qoperations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.shuffle(2)
    data1 = data1.map((lambda x: x), ["data"])
    data1 = data1.batch(2)

    expected_data = np.array([[[1], [2]], [[3], [0]]])
    for i, data_row in enumerate(data1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(data_row[0], expected_data[i])

    # Restore configuration
    ds.config.set_seed(original_seed)
    ds.config.set_num_parallel_workers((original_num_parallel_workers))


if __name__ == "__main__":
    test_case_0()
