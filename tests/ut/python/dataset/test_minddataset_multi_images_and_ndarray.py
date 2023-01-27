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
"""
This is the test module for mindrecord
"""
import os
import numpy as np

import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.mindrecord import FileWriter

FILES_NUM = 1


def test_cv_minddataset_reader_multi_image_and_ndarray_tutorial():
    """
    Feature: MindDataset
    Description: Test for MindDataset reader for multiple images and ndarray tutorial
    Expectation: Runs successfully
    """
    try:
        file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        writer = FileWriter(file_name, FILES_NUM)
        cv_schema_json = {"id": {"type": "int32"},
                          "image_0": {"type": "bytes"},
                          "image_2": {"type": "bytes"},
                          "image_3": {"type": "bytes"},
                          "image_4": {"type": "bytes"},
                          "input_mask": {"type": "int32", "shape": [-1]},
                          "segments": {"type": "float32", "shape": [2, 3]}}
        writer.add_schema(cv_schema_json, "two_images_schema")
        with open("../data/mindrecord/testImageNetData/images/image_00010.jpg", "rb") as file_reader:
            img_data = file_reader.read()
        ndarray_1 = np.array([1, 2, 3, 4, 5], np.int32)
        ndarray_2 = np.array(([2, 3, 1], [7, 9, 0]), np.float32)
        data = []
        for i in range(5):
            item = {"id": i, "image_0": img_data, "image_2": img_data, "image_3": img_data, "image_4": img_data,
                    "input_mask": ndarray_1, "segments": ndarray_2}
            data.append(item)
        writer.write_raw_data(data)
        writer.commit()
        assert os.path.exists(file_name)
        assert os.path.exists(file_name + ".db")

        # tutorial for minderdataset.
        columns_list = ["id", "image_0", "image_2", "image_3", "image_4", "input_mask", "segments"]
        num_readers = 1
        data_set = ds.MindDataset(file_name, columns_list, num_readers)
        assert data_set.get_dataset_size() == 5
        num_iter = 0
        for item in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            assert len(item) == 7
            logger.info("item: {}".format(item))
            assert item["image_0"].dtype == np.dtype('S45150')
            assert (item["image_0"] == item["image_2"]).all()
            assert (item["image_3"] == item["image_4"]).all()
            assert (item["image_0"] == item["image_4"]).all()
            assert item["image_2"].dtype == np.dtype('S45150')
            assert item["image_3"].dtype == np.dtype('S45150')
            assert item["image_4"].dtype == np.dtype('S45150')
            assert item["id"].dtype == np.int32
            assert item["input_mask"].shape == (5,)
            assert item["input_mask"].dtype == np.int32
            assert item["segments"].shape == (2, 3)
            assert item["segments"].dtype == np.float32
            num_iter += 1
        assert num_iter == 5
    except Exception as error:
        if os.path.exists("{}".format(file_name + ".db")):
            os.remove(file_name + ".db")
        if os.path.exists("{}".format(file_name)):
            os.remove(file_name)
        raise error
    else:
        if os.path.exists("{}".format(file_name + ".db")):
            os.remove(file_name + ".db")
        if os.path.exists("{}".format(file_name)):
            os.remove(file_name)

if __name__ == '__main__':
    test_cv_minddataset_reader_multi_image_and_ndarray_tutorial()
