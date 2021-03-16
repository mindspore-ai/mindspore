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
Cifar100 convert tool for MindRecord.
"""

from importlib import import_module
import os
import numpy as np

from mindspore import log as logger
from .cifar100 import Cifar100
from ..common.exceptions import PathNotExistsError
from ..filewriter import FileWriter
from ..shardutils import check_filename, ExceptionThread, SUCCESS

try:
    cv2 = import_module("cv2")
except ModuleNotFoundError:
    cv2 = None

__all__ = ['Cifar100ToMR']


class Cifar100ToMR:
    """
    A class to transform from cifar100 to MindRecord.

    Args:
        source (str): the cifar100 directory to be transformed.
        destination (str): the MindRecord file path to transform into.

    Raises:
        ValueError: If source or destination is invalid.
    """

    def __init__(self, source, destination):
        check_filename(source)
        self.source = source

        files = os.listdir(self.source)
        train_data_flag = False
        test_data_flag = False
        for file in files:
            if file == "train":
                train_data_flag = True
            if file == "test":
                test_data_flag = True
        if not train_data_flag:
            raise PathNotExistsError("train")

        if not test_data_flag:
            raise PathNotExistsError("test")

        check_filename(destination)
        self.destination = destination
        self.writer = None

    def run(self, fields=None):
        """
        Execute transformation from cifar100 to MindRecord.

        Args:
            fields (list[str]): A list of index field, e.g.["fine_label", "coarse_label"].

        Returns:
            MSRStatus, whether cifar100 is successfully transformed to MindRecord.
        """
        if fields and not isinstance(fields, list):
            raise ValueError("The parameter fields should be None or list")

        cifar100_data = Cifar100(self.source, False)
        cifar100_data.load_data()

        images = cifar100_data.images
        logger.info("train images: {}".format(images.shape))
        fine_labels = cifar100_data.fine_labels
        logger.info("train images fine label: {}".format(fine_labels.shape))
        coarse_labels = cifar100_data.coarse_labels
        logger.info("train images coarse label: {}".format(coarse_labels.shape))

        test_images = cifar100_data.Test.images
        logger.info("test images: {}".format(test_images.shape))
        test_fine_labels = cifar100_data.Test.fine_labels
        logger.info("test images fine label: {}".format(fine_labels.shape))
        test_coarse_labels = cifar100_data.Test.coarse_labels
        logger.info("test images coarse label: {}".format(coarse_labels.shape))

        data_list = _construct_raw_data(images, fine_labels, coarse_labels)
        test_data_list = _construct_raw_data(test_images, test_fine_labels, test_coarse_labels)

        if _generate_mindrecord(self.destination, data_list, fields, "img_train") != SUCCESS:
            return FAILED
        if _generate_mindrecord(self.destination + "_test", test_data_list, fields, "img_test") != SUCCESS:
            return FAILED
        return SUCCESS

    def transform(self, fields=None):
        t = ExceptionThread(target=self.run, kwargs={'fields': fields})
        t.daemon = True
        t.start()
        t.join()
        if t.exitcode != 0:
            raise t.exception
        return t.res


def _construct_raw_data(images, fine_labels, coarse_labels):
    """
    Construct raw data from cifar100 data.

    Args:
        images (list): image list from cifar100.
        fine_labels (list): fine label list from cifar100.
        coarse_labels (list): coarse label list from cifar100.

    Returns:
        list[dict], data dictionary constructed from cifar100.
    """
    if not cv2:
        raise ModuleNotFoundError("opencv-python module not found, please use pip install it.")

    raw_data = []
    for i, img in enumerate(images):
        fine_label = np.int(fine_labels[i][0])
        coarse_label = np.int(coarse_labels[i][0])
        _, img = cv2.imencode(".jpeg", img[..., [2, 1, 0]])
        row_data = {"id": int(i),
                    "data": img.tobytes(),
                    "fine_label": int(fine_label),
                    "coarse_label": int(coarse_label)}
        raw_data.append(row_data)
    return raw_data


def _generate_mindrecord(file_name, raw_data, fields, schema_desc):
    """
    Generate MindRecord file from raw data.

    Args:
        file_name (str): File name of MindRecord File.
        fields (list[str]): Fields would be set as index which
          could not belong to blob fields and type could not be 'array' or 'bytes'.
        raw_data (dict): Dict of raw data.
        schema_desc (str): String of schema description.

    Returns:
        MSRStatus, whether successfully written into MindRecord.
    """
    schema = {"id": {"type": "int64"}, "fine_label": {"type": "int64"},
              "coarse_label": {"type": "int64"}, "data": {"type": "bytes"}}

    logger.info("transformed MindRecord schema is: {}".format(schema))

    writer = FileWriter(file_name, 1)
    writer.add_schema(schema, schema_desc)
    if fields and isinstance(fields, list):
        writer.add_index(fields)
    writer.write_raw_data(raw_data)
    return writer.commit()
