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
Testing TypeCast op in DE
"""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as data_util
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_type_cast():
    """
    Test TypeCast op
    """
    logger.info("test_type_cast")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    type_cast_op = data_util.TypeCast(mstype.float32)

    ctrans = [decode_op,
              type_cast_op,
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [py_vision.Decode(),
                  py_vision.ToTensor()
                  ]
    transform = py_vision.ComposeOp(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform())

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))
        assert c_image.dtype == "float32"


def test_type_cast_string():
    """
    Test TypeCast op
    """
    logger.info("test_type_cast_string")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = c_vision.Decode()

    type_cast_op = data_util.TypeCast(mstype.float16)

    ctrans = [decode_op,
              type_cast_op
              ]

    data1 = data1.map(input_columns=["image"], operations=ctrans)

    # Second dataset
    transforms = [py_vision.Decode(),
                  py_vision.ToTensor()
                  ]
    transform = py_vision.ComposeOp(transforms)
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(input_columns=["image"], operations=transform())

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(), data2.create_dict_iterator()):
        num_iter += 1
        c_image = item1["image"]
        py_image = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)

        logger.info("shape of c_image: {}".format(c_image.shape))
        logger.info("shape of py_image: {}".format(py_image.shape))

        logger.info("dtype of c_image: {}".format(c_image.dtype))
        logger.info("dtype of py_image: {}".format(py_image.dtype))
        assert c_image.dtype == "float16"


if __name__ == "__main__":
    test_type_cast()
    test_type_cast_string()
