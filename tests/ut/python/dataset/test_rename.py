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
from mindspore import log as logger

DATA_DIR = ["../data/dataset/testTFBert5Rows1/5TFDatas.data"]
DATA_DIR_2 = ["../data/dataset/testTFBert5Rows2/5TFDatas.data"]
SCHEMA_DIR = "../data/dataset/testTFBert5Rows1/datasetSchema.json"
SCHEMA_DIR_2 = "../data/dataset/testTFBert5Rows2/datasetSchema.json"


def test_rename():
    """
    Feature: Rename op
    Description: Test rename op followed by repeat
    Expectation: Output is the same as expected output
    """
    data1 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2, shuffle=False)
    data2 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2, shuffle=False)

    data2 = data2.rename(input_columns=["input_ids", "segment_ids"], output_columns=["masks", "seg_ids"])

    data = ds.zip((data1, data2))
    data = data.repeat(3)

    num_iter = 0

    for _, item in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        logger.info("item[mask] is {}".format(item["masks"]))
        np.testing.assert_equal(item["masks"], item["input_ids"])
        logger.info("item[seg_ids] is {}".format(item["seg_ids"]))
        np.testing.assert_equal(item["segment_ids"], item["seg_ids"])
        # need to consume the data in the buffer
        num_iter += 1
    logger.info("Number of data in data: {}".format(num_iter))
    assert num_iter == 15


if __name__ == '__main__':
    logger.info('===========test Rename Repeat===========')
    test_rename()
    logger.info('\n')
