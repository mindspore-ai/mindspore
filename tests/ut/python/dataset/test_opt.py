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
import numpy as np
import pytest

import mindspore.dataset as ds

# Generate 1d int numpy array from 0 - 63
def generator_1d():
    for i in range(64):
        yield (np.array([i]),)


def test_case_0():
    """
    Test 1D Generator
    """

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    data1 = data1.shuffle(2)

    data1 = data1.map(["data"], operations=(lambda x : x))

    data1 = data1.batch(2)

    i = 0
    for item in data1.create_dict_iterator():  # each data is a dictionary
        pass


if __name__ == "__main__":
    test_case_0()