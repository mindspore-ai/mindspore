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
Cifar10 convert tool for MindRecord.
"""

from importlib import import_module
import os
import numpy as np

from mindspore import log as logger
from .cifar10 import Cifar10
from ..common.exceptions import PathNotExistsError
from ..filewriter import FileWriter
from ..shardutils import check_filename, ExceptionThread, SUCCESS, FAILED

try:
    cv2 = import_module("cv2")
except ModuleNotFoundError:
    cv2 = None

__all__ = ['Cifar10ToMR']


class Cifar10ToMR:
    """
    A class to transform from cifar10 to MindRecord.

    Args:
        source (str): the cifar10 directory to be transformed.
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
            if file.startswith("data_batch_"):
                train_data_flag = True
            if file.startswith("test_batch"):
                test_data_flag = True
        if not train_data_flag:
            raise PathNotExistsError("data_batch_*")

        if not test_data_flag:
            raise PathNotExistsError("test_batch")

        check_filename(destination)
        self.destination = destination
        self.writer = None

    def run(self, fields=None):
        """
        Execute transformation from cifar10 to MindRecord.

        Args:
            fields (list[str], optional): A list of index fields, e.g.["label"] (default=None).

        Returns:
            MSRStatus, whether cifar10 is successfully transformed to MindRecord.
        """
        if fields and not isinstance(fields, list):
            raise ValueError("The parameter fields should be None or list")

        cifar10_data = Cifar10(self.source, False)
        cifar10_data.load_data()

        images = cifar10_data.images
        logger.info("train images: {}".format(images.shape))
        labels = cifar10_data.labels
        logger.info("train images label: {}".format(labels.shape))

        test_images = cifar10_data.Test.images
        logger.info("test images: {}".format(test_images.shape))
        test_labels = cifar10_data.Test.labels
        logger.info("test images label: {}".format(test_labels.shape))

        data_list = _construct_raw_data(images, labels)
        test_data_list = _construct_raw_data(test_images, test_labels)

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


def _construct_raw_data(images, labels):
    """
    Construct raw data from cifar10 data.

    Args:
        images (list): image list from cifar10.
        labels (list): label list from cifar10.

    Returns:
        list[dict], data dictionary constructed from cifar10.
    """
    if not cv2:
        raise ModuleNotFoundError("opencv-python module not found, please use pip install it.")

    raw_data = []
    for i, img in enumerate(images):
        label = np.int(labels[i][0])
        _, img = cv2.imencode(".jpeg", img[..., [2, 1, 0]])
        row_data = {"id": int(i),
                    "data": img.tobytes(),
                    "label": int(label)}
        raw_data.append(row_data)
    return raw_data


def _generate_mindrecord(file_name, raw_data, fields, schema_desc):
    """
    Generate MindRecord file from raw data.

    Args:
        file_name (str): File name of MindRecord File.
        fields (list[str]): Fields would be set as index which
          could not belong to blob fields and type could not be 'array' or 'bytes'.
        raw_data (dict): dict of raw data.
        schema_desc (str): String of schema description.

    Returns:
        MSRStatus, whether successfully written into MindRecord.
    """
    schema = {"id": {"type": "int64"}, "label": {"type": "int64"},
              "data": {"type": "bytes"}}

    logger.info("transformed MindRecord schema is: {}".format(schema))

    writer = FileWriter(file_name, 1)
    writer.add_schema(schema, schema_desc)
    if fields and isinstance(fields, list):
        writer.add_index(fields)
    writer.write_raw_data(raw_data)
    return writer.commit()
