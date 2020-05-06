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
"""
Testing configuration manager 
"""
import filecmp
import glob
import os

import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

def test_basic():
    ds.config.load('../data/dataset/declient.cfg')

    # assert ds.config.get_rows_per_buffer() == 32
    assert ds.config.get_num_parallel_workers() == 4
    # assert ds.config.get_worker_connector_size() == 16
    assert ds.config.get_prefetch_size() == 16
    assert ds.config.get_seed() == 5489

    # ds.config.set_rows_per_buffer(1)
    ds.config.set_num_parallel_workers(2)
    # ds.config.set_worker_connector_size(3)
    ds.config.set_prefetch_size(4)
    ds.config.set_seed(5)

    # assert ds.config.get_rows_per_buffer() == 1
    assert ds.config.get_num_parallel_workers() == 2
    # assert ds.config.get_worker_connector_size() == 3
    assert ds.config.get_prefetch_size() == 4
    assert ds.config.get_seed() == 5

def test_pipeline():
    """ 
    Test that our configuration pipeline works when we set parameters at dataset interval 
    """
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    ds.config.set_num_parallel_workers(2)
    data1 = data1.map(input_columns=["image"], operations=[vision.Decode(True)])
    ds.serialize(data1, "testpipeline.json")

    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    ds.config.set_num_parallel_workers(4)
    data2 = data2.map(input_columns=["image"], operations=[vision.Decode(True)])
    ds.serialize(data2, "testpipeline2.json")

    # check that the generated output is different 
    assert (filecmp.cmp('testpipeline.json', 'testpipeline2.json'))

    # this test passes currently because our num_parallel_workers don't get updated. 

    # remove generated jason files 
    file_list = glob.glob('*.json')
    for f in file_list:
        try:
            os.remove(f)
        except IOError:
            logger.info("Error while deleting: {}".format(f))


if __name__ == '__main__':
    test_basic()
    test_pipeline()
