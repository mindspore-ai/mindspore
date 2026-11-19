# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
Test Python Multiprocessing with Python functions/ops
"""
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset.core.config import _get_enable_shared_mem, _set_enable_shared_mem
from mindspore.dataset.core.config import _get_prefetch_size

MNIST_DATA_DIR = "../data/dataset/testMnistData"
TF_DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
TF_SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
PYFUNCMAP_DATA_DIR = ["../data/dataset/testPyfuncMap/data.data"]
PYFUNCMAP_SCHEMA_DIR = "../data/dataset/testPyfuncMap/schema.json"


def test_pyfunc_multiproc_shrmem():
    """
    Feature: PyFunc in Map op
    Description: Test python_multiprocessing=True with shared memory enabled
    Expectation: Data results are correct
    """

    def pyfunc(x):
        return x

    # Confirm shared memory optimization is enabled by default
    mem_original = _get_enable_shared_mem()
    assert mem_original

    # Reduce memory needed by reducing queue size
    prefetch_original = _get_prefetch_size()
    ds.config.set_prefetch_size(1)

    max_elements = 2000
    np_data = list(range(0, max_elements))

    data1 = ds.NumpySlicesDataset(np_data, shuffle=False)

    data1 = data1.map(pyfunc, num_parallel_workers=2, python_multiprocessing=True, max_rowsize=1)

    for i, data in enumerate(data1):
        np.testing.assert_equal(data[0].asnumpy(), np_data[i])

    assert data1.get_dataset_size() == max_elements

    ds.config.set_prefetch_size(prefetch_original)














def test_pyfunc_multiproc_mainproc_exception():
    """
    Feature: PyFunc in Map op
    Description: Test python_multiprocessing=True with exception in main process
    Expectation: Exception is received and test ends gracefully
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = _get_enable_shared_mem()
    _set_enable_shared_mem(False)

    # Apply dataset operations
    data1 = ds.TFRecordDataset(PYFUNCMAP_DATA_DIR, PYFUNCMAP_SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x: x + x), input_columns="col0", output_columns="out",
                      python_multiprocessing=True)

    with pytest.raises(ZeroDivisionError) as info:
        i = 0
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            i = i + 4
            if i > 8:
                # Cause division by zero error
                _ = i / 0
        assert "division by zero" in str(info.value)

    _set_enable_shared_mem(mem_original)
