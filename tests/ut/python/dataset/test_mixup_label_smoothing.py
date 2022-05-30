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
import mindspore.dataset.transforms as data_trans
import mindspore.dataset.vision as vision
from mindspore import log as logger

DATA_DIR_2 = "../data/dataset/testImageNetData2/train"


def test_mix_up_single():
    """
    Feature: MixUp Op
    Description: Test Python op, single batch mix up scenario
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("Test single batch mix up op")

    resize_height = 224
    resize_width = 224

    # Create dataset and define map operations
    ds1 = ds.ImageFolderDataset(DATA_DIR_2)

    num_classes = 10
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), vision.Inter.LINEAR)
    one_hot_encode = data_trans.OneHot(num_classes)  # num_classes is input argument

    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.map(operations=resize_op, input_columns=["image"])
    ds1 = ds1.map(operations=one_hot_encode, input_columns=["label"])

    # apply batch operations
    batch_size = 3
    ds1 = ds1.batch(batch_size, drop_remainder=True)

    ds2 = ds1
    alpha = 0.2
    transforms = [vision.MixUp(batch_size=batch_size, alpha=alpha, is_single=True)
                  ]
    ds1 = ds1.map(operations=transforms, input_columns=["image", "label"])

    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            ds2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = data1["image"]
        label = data1["label"]
        logger.info("label is {}".format(label))

        image2 = data2["image"]
        label2 = data2["label"]
        logger.info("label2 is {}".format(label2))

        lam = np.abs(label - label2)
        for index in range(batch_size - 1):
            if np.square(lam[index]).mean() != 0:
                lam_value = 1 - np.sum(lam[index]) / 2
                img_golden = lam_value * image2[index] + (1 - lam_value) * image2[index + 1]
                assert image1[index].all() == img_golden.all()
                logger.info("====test single batch mixup ok====")


def test_mix_up_multi():
    """
    Feature: MixUp Op
    Description: Test Python op, multiple batch mix up scenario
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    logger.info("Test several batch mix up op")

    resize_height = 224
    resize_width = 224

    # Create dataset and define map operations
    ds1 = ds.ImageFolderDataset(DATA_DIR_2)

    num_classes = 3
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), vision.Inter.LINEAR)
    one_hot_encode = data_trans.OneHot(num_classes)  # num_classes is input argument

    ds1 = ds1.map(operations=decode_op, input_columns=["image"])
    ds1 = ds1.map(operations=resize_op, input_columns=["image"])
    ds1 = ds1.map(operations=one_hot_encode, input_columns=["label"])

    # apply batch operations
    batch_size = 3
    ds1 = ds1.batch(batch_size, drop_remainder=True)

    ds2 = ds1
    alpha = 0.2
    transforms = [vision.MixUp(batch_size=batch_size, alpha=alpha, is_single=False)
                  ]
    ds1 = ds1.map(operations=transforms, input_columns=["image", "label"])
    num_iter = 0
    batch1_image1 = 0
    for data1, data2 in zip(ds1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            ds2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = data1["image"]
        label1 = data1["label"]
        logger.info("label: {}".format(label1))

        image2 = data2["image"]
        label2 = data2["label"]
        logger.info("label2: {}".format(label2))

        if num_iter == 0:
            batch1_image1 = image1

        if num_iter == 1:
            lam = np.abs(label2 - label1)
            logger.info("lam value in multi: {}".format(lam))
            for index in range(batch_size):
                if np.square(lam[index]).mean() != 0:
                    lam_value = 1 - np.sum(lam[index]) / 2
                    img_golden = lam_value * image2[index] + (1 - lam_value) * batch1_image1[index]
                    assert image1[index].all() == img_golden.all()
                    logger.info("====test several batch mixup ok====")
            break
        num_iter += 1


if __name__ == "__main__":
    test_mix_up_single()
    test_mix_up_multi()
