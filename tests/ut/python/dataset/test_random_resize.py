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
Testing the resize op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore import log as logger
from util import visualize_list

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_random_resize_op(plot=False):
    """
    Test random_resize_op
    """
    logger.info("Test resize")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    resize_op = vision.RandomResize(10)

    # apply map operations on images
    data1 = data1.map(input_columns=["image"], operations=decode_op)

    data2 = data1.map(input_columns=["image"], operations=resize_op)
    image_original = []
    image_resized = []
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        image_1 = item1["image"]
        image_2 = item2["image"]
        image_original.append(image_1)
        image_resized.append(image_2)
        num_iter += 1
    if plot:
        visualize_list(image_original, image_resized)


if __name__ == "__main__":
    test_random_resize_op(plot=True)
