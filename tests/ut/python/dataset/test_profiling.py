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
"""
Testing profiling support in DE
"""
import os
import numpy as np
import mindspore.dataset as ds

FILES = ["../data/dataset/testTFTestAllTypes/test.data"]
DATASET_ROOT = "../data/dataset/testTFTestAllTypes/"
SCHEMA_FILE = "../data/dataset/testTFTestAllTypes/datasetSchema.json"

PIPELINE_FILE_SIZE = "./pipeline_profiling_1.json"
PIPELINE_FILE_THR = "./pipeline_profiling_Connector_Throughput_Sampling_1.json"
DATASET_ITERATOR_FILE = "./dataset_iterator_profiling_1.txt"


def test_profiling_simple_pipeline():
    """
    Generator -> Shuffle -> Batch
    """
    os.environ['PROFILING_MODE'] = 'true'
    os.environ['MINDDATA_PROFILING_DIR'] = '.'
    os.environ['DEVICE_ID'] = '1'

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["data"])
    data1 = data1.shuffle(64)
    data1 = data1.batch(32)

    for _ in data1:
        pass

    assert os.path.exists(PIPELINE_FILE_SIZE) is True
    os.remove(PIPELINE_FILE_SIZE)
    assert os.path.exists(PIPELINE_FILE_THR) is True
    os.remove(PIPELINE_FILE_THR)
    assert os.path.exists(DATASET_ITERATOR_FILE) is True
    os.remove(DATASET_ITERATOR_FILE)
    del os.environ['PROFILING_MODE']
    del os.environ['MINDDATA_PROFILING_DIR']


def test_profiling_complex_pipeline():
    """
    Generator -> Map     ->
                             -> Zip -> Batch
    TFReader  -> Shuffle ->
    """
    os.environ['PROFILING_MODE'] = 'true'
    os.environ['MINDDATA_PROFILING_DIR'] = '.'
    os.environ['DEVICE_ID'] = '1'

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["gen"])
    data1 = data1.map("gen", operations=[(lambda x: x + 1)])

    pattern = DATASET_ROOT + "/test.data"
    data2 = ds.TFRecordDataset(pattern, SCHEMA_FILE, shuffle=ds.Shuffle.FILES)
    data2 = data2.shuffle(4)

    data3 = ds.zip((data1, data2))

    for _ in data3:
        pass

    assert os.path.exists(PIPELINE_FILE_SIZE) is True
    os.remove(PIPELINE_FILE_SIZE)
    assert os.path.exists(PIPELINE_FILE_THR) is True
    os.remove(PIPELINE_FILE_THR)
    assert os.path.exists(DATASET_ITERATOR_FILE) is True
    os.remove(DATASET_ITERATOR_FILE)
    del os.environ['PROFILING_MODE']
    del os.environ['MINDDATA_PROFILING_DIR']


def test_profiling_sampling_iterval():
    """
    Test non-default monitor sampling interval
    """
    os.environ['PROFILING_MODE'] = 'true'
    os.environ['MINDDATA_PROFILING_DIR'] = '.'
    os.environ['DEVICE_ID'] = '1'
    interval_origin = ds.config.get_monitor_sampling_interval()

    ds.config.set_monitor_sampling_interval(30)
    interval = ds.config.get_monitor_sampling_interval()
    assert interval == 30

    source = [(np.array([x]),) for x in range(1024)]
    data1 = ds.GeneratorDataset(source, ["data"])
    data1 = data1.shuffle(64)
    data1 = data1.batch(32)

    for _ in data1:
        pass

    assert os.path.exists(PIPELINE_FILE_SIZE) is True
    os.remove(PIPELINE_FILE_SIZE)
    assert os.path.exists(PIPELINE_FILE_THR) is True
    os.remove(PIPELINE_FILE_THR)
    assert os.path.exists(DATASET_ITERATOR_FILE) is True
    os.remove(DATASET_ITERATOR_FILE)

    ds.config.set_monitor_sampling_interval(interval_origin)
    del os.environ['PROFILING_MODE']
    del os.environ['MINDDATA_PROFILING_DIR']


if __name__ == "__main__":
    test_profiling_simple_pipeline()
    test_profiling_complex_pipeline()
    test_profiling_sampling_iterval()
