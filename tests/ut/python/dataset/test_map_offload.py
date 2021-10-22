# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.dataset.vision.c_transforms as C


DATA_DIR = "../data/dataset/testPK/data"

def test_offload():
    """
    Feature: test map offload flag.
    Description: Input is image dataset.
    Expectation: Output should be same with activated or deactivated offload.
    """
    # Dataset with offload activated.
    dataset_0 = ds.ImageFolderDataset(DATA_DIR)
    dataset_0 = dataset_0.map(operations=[C.Decode()], input_columns="image")
    dataset_0 = dataset_0.map(operations=[C.HWC2CHW()], input_columns="image", offload=True)
    dataset_0 = dataset_0.batch(8, drop_remainder=True)

    # Dataset with offload not activated.
    dataset_1 = ds.ImageFolderDataset(DATA_DIR)
    dataset_1 = dataset_1.map(operations=[C.Decode()], input_columns="image")
    dataset_1 = dataset_1.map(operations=[C.HWC2CHW()], input_columns="image")
    dataset_1 = dataset_1.batch(8, drop_remainder=True)

    for (img_0, _), (img_1, _) in zip(dataset_0.create_tuple_iterator(num_epochs=1, output_numpy=True),
                                      dataset_1.create_tuple_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(img_0, img_1)


if __name__ == "__main__":
    test_offload()
