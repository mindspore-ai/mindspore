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
import mindspore.dataset.transforms.vision.c_transforms as vision

import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testVOC2012"


def test_voc_normal():
    data1 = ds.VOCDataset(DATA_DIR, decode=True)
    num = 0
    for item in data1.create_dict_iterator():
        logger.info("item[image] is {}".format(item["image"]))
        logger.info("item[image].shape is {}".format(item["image"].shape))
        logger.info("item[target] is {}".format(item["target"]))
        logger.info("item[target].shape is {}".format(item["target"].shape))
        num += 1
    logger.info("num is {}".format(str(num)))


def test_case_0():
    data1 = ds.VOCDataset(DATA_DIR, decode=True)

    resize_op = vision.Resize((224, 224))

    data1 = data1.map(input_columns=["image"], operations=resize_op)
    data1 = data1.map(input_columns=["target"], operations=resize_op)
    repeat_num = 4
    data1 = data1.repeat(repeat_num)
    batch_size = 2
    data1 = data1.batch(batch_size, drop_remainder=True)

    num = 0
    for item in data1.create_dict_iterator():
        logger.info("item[image].shape is {}".format(item["image"].shape))
        logger.info("item[target].shape is {}".format(item["target"].shape))
        num += 1
    logger.info("num is {}".format(str(num)))
