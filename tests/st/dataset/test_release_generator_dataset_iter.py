# Copyright 2023 Huawei Technologies Co., Ltd
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
This is the test for release GeneratorDataset iterator
"""

import os
import psutil
import pytest
import numpy as np

import mindspore.dataset as ds
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("num_epochs", (-1, 10))
@pytest.mark.skip(reason="I6UY43")
def test_release_generator_dataset_iter(num_epochs):
    """
    Feature: test release GeneratorDataset iterator
    Description: None
    Expectation: SUCCESS
    """
    orginal_prefetch_size = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    # Iterable object as input source
    class Iterable:
        def __init__(self):
            self.a = [np.ones((2048*2048*3), dtype=np.int64),
                      np.ones((2048*2048*3*2), dtype=np.int64),
                      np.ones((2048*2048*3*3), dtype=np.int64),
                      np.ones((2048*2048*3*4), dtype=np.int64),
                      np.ones((2048*2048*3*5), dtype=np.int64)]
            self.b = [np.ones((1024*1024*3), dtype=np.int64),
                      np.ones((1024*1024*3*2), dtype=np.int64),
                      np.ones((1024*1024*3*3), dtype=np.int64),
                      np.ones((1024*1024*3*4), dtype=np.int64),
                      np.ones((1024*1024*3*5), dtype=np.int64)]
            self.len = len(self.a)

        def __getitem__(self, index):
            return self.a[4 - index], self.b[4 - index]

        def __len__(self):
            return self.len

    data = Iterable()
    dataset = ds.GeneratorDataset(source=data, column_names=["data", "label"], shuffle=False)

    init_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=num_epochs)

    for item in ds_iter:
        break

    iter_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (iter_memory - init_memory) > 1000  # use memory > 1000MB

    del ds_iter
    del item  # pylint: disable=undefined-loop-variable

    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (end_memory - init_memory) < 2      # after del, use memory < 2MB

    ds.config.set_prefetch_size(orginal_prefetch_size)
