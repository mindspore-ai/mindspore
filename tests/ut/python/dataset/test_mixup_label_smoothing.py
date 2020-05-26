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
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c
import mindspore.dataset.transforms.py_transforms as f
import mindspore.dataset.transforms.vision.c_transforms as c_vision
import mindspore.dataset.transforms.vision.py_transforms as py_vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testImageNetData/train"
DATA_DIR_2 = "../data/dataset/testImageNetData2/train"


def test_one_hot_op():
    """
    Test one hot encoding op
    """
    logger.info("Test one hot encoding op")

    # define map operations
    # ds = de.ImageFolderDataset(DATA_DIR, schema=SCHEMA_DIR)
    dataset = ds.ImageFolderDatasetV2(DATA_DIR)
    num_classes = 2
    epsilon_para = 0.1

    transforms = [f.OneHotOp(num_classes=num_classes, smoothing_rate=epsilon_para),
                  ]
    transform_label = py_vision.ComposeOp(transforms)
    dataset = dataset.map(input_columns=["label"], operations=transform_label())

    golden_label = np.ones(num_classes) * epsilon_para / num_classes
    golden_label[1] = 1 - epsilon_para / num_classes

    for data in dataset.create_dict_iterator():
        label = data["label"]
        logger.info("label is {}".format(label))
        logger.info("golden_label is {}".format(golden_label))
        assert label.all() == golden_label.all()
        logger.info("====test one hot op ok====")


def test_mix_up_single():
    """
    Test single batch mix up op
    """
    logger.info("Test single batch mix up op")

    resize_height = 224
    resize_width = 224

    # Create dataset and define map operations
    ds1 = ds.ImageFolderDatasetV2(DATA_DIR_2)

    num_classes = 10
    decode_op = c_vision.Decode()
    resize_op = c_vision.Resize((resize_height, resize_width), c_vision.Inter.LINEAR)
    one_hot_encode = c.OneHot(num_classes)  # num_classes is input argument

    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.map(input_columns=["image"], operations=resize_op)
    ds1 = ds1.map(input_columns=["label"], operations=one_hot_encode)

    # apply batch operations
    batch_size = 3
    ds1 = ds1.batch(batch_size, drop_remainder=True)

    ds2 = ds1
    alpha = 0.2
    transforms = [py_vision.MixUp(batch_size=batch_size, alpha=alpha, is_single=True)
                  ]
    ds1 = ds1.map(input_columns=["image", "label"], operations=transforms)

    for data1, data2 in zip(ds1.create_dict_iterator(), ds2.create_dict_iterator()):
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
    Test multi batch mix up op
    """
    logger.info("Test several batch mix up op")

    resize_height = 224
    resize_width = 224

    # Create dataset and define map operations
    ds1 = ds.ImageFolderDatasetV2(DATA_DIR_2)

    num_classes = 3
    decode_op = c_vision.Decode()
    resize_op = c_vision.Resize((resize_height, resize_width), c_vision.Inter.LINEAR)
    one_hot_encode = c.OneHot(num_classes)  # num_classes is input argument

    ds1 = ds1.map(input_columns=["image"], operations=decode_op)
    ds1 = ds1.map(input_columns=["image"], operations=resize_op)
    ds1 = ds1.map(input_columns=["label"], operations=one_hot_encode)

    # apply batch operations
    batch_size = 3
    ds1 = ds1.batch(batch_size, drop_remainder=True)

    ds2 = ds1
    alpha = 0.2
    transforms = [py_vision.MixUp(batch_size=batch_size, alpha=alpha, is_single=False)
                  ]
    ds1 = ds1.map(input_columns=["image", "label"], operations=transforms)
    num_iter = 0
    batch1_image1 = 0
    for data1, data2 in zip(ds1.create_dict_iterator(), ds2.create_dict_iterator()):
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
        num_iter = num_iter + 1


if __name__ == "__main__":
    test_one_hot_op()
    test_mix_up_single()
    test_mix_up_multi()
