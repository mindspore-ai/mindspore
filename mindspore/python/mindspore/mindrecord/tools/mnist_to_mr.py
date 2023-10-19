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
"""
Mnist convert tool for MindRecord
"""

from importlib import import_module
import os
import time
import gzip
import numpy as np

from mindspore import log as logger
from ..filewriter import FileWriter
from ..shardutils import check_filename, ExceptionThread, SUCCESS, FAILED

try:
    cv_import = import_module("cv2")
except ModuleNotFoundError:
    cv_import = None

__all__ = ['MnistToMR']


class MnistToMR:
    """
    A class to transform from Mnist to MindRecord.

    Args:
        source (str): Directory that contains t10k-images-idx3-ubyte.gz,
                      train-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz
                      and train-labels-idx1-ubyte.gz.
        destination (str): MindRecord file path to transform into, ensure that the directory is created in advance and
            no file with the same name exists in the directory.
        partition_number (int, optional): The partition size. Default: ``1`` .

    Raises:
        ValueError: If `source` , `destination` , `partition_number` is invalid.

    Examples:
        >>> from mindspore.mindrecord import MnistToMR
        >>>
        >>> mnist_dir = "/path/to/mnist"
        >>> mindrecord_file = "/path/to/mindrecord/file"
        >>> mnist_to_mr = MnistToMR(mnist_dir, mindrecord_file)
        >>> status = mnist_to_mr.transform()
    """

    def __init__(self, source, destination, partition_number=1):
        self.image_size = 28
        self.num_channels = 1

        check_filename(source)

        self.source = source
        self.train_data_filename_ = os.path.join(self.source, 'train-images-idx3-ubyte.gz')
        self.train_labels_filename_ = os.path.join(self.source, 'train-labels-idx1-ubyte.gz')
        self.test_data_filename_ = os.path.join(self.source, 't10k-images-idx3-ubyte.gz')
        self.test_labels_filename_ = os.path.join(self.source, 't10k-labels-idx1-ubyte.gz')

        check_filename(self.train_data_filename_)
        check_filename(self.train_labels_filename_)
        check_filename(self.test_data_filename_)
        check_filename(self.test_labels_filename_)
        check_filename(destination)

        if partition_number is not None:
            if not isinstance(partition_number, int):
                raise ValueError("The parameter partition_number must be int")
            self.partition_number = partition_number
        else:
            raise ValueError("The parameter partition_number must be int")

        self.writer_train = FileWriter("{}_train.mindrecord".format(destination), self.partition_number)
        self.writer_test = FileWriter("{}_test.mindrecord".format(destination), self.partition_number)

        self.mnist_schema_json = {"label": {"type": "int64"}, "data": {"type": "bytes"}}

    # pylint: disable=missing-docstring
    def run(self):
        if not cv_import:
            raise ModuleNotFoundError("opencv-python module not found, please use pip install it.")

        if self._transform_train() == FAILED:
            return FAILED
        if self._transform_test() == FAILED:
            return FAILED

        return SUCCESS

    def transform(self):
        """
        Execute transformation from Mnist to MindRecord.

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.MnistToMR` .

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            ParamTypeError: If index field is invalid.
            MRMOpenError: If failed to open MindRecord file.
            MRMValidateDataError: If data does not match blob fields.
            MRMSetHeaderError: If failed to set header.
            MRMWriteDatasetError: If failed to write dataset.
        """

        t = ExceptionThread(target=self.run)
        t.daemon = True
        t.start()
        t.join()
        if t.exitcode != 0:
            raise t.exception
        return t.res

    def _extract_images(self, filename):
        """Extract the images into a 4D tensor [image index, y, x, channels]."""
        real_file_path = os.path.realpath(filename)
        with gzip.open(real_file_path, "rb") as bytestream:
            bytestream.read(16)
            buf = bytestream.read()
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(-1, self.image_size, self.image_size, self.num_channels)
            return data

    def _extract_labels(self, filename):
        """Extract the labels into a vector of int64 label IDs."""
        real_file_path = os.path.realpath(filename)
        with gzip.open(real_file_path, "rb") as bytestream:
            bytestream.read(8)
            buf = bytestream.read()
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            return labels

    def _mnist_train_iterator(self):
        """
        get data from mnist train data and label file.

        Yields:
            data (dict of list): mnist data list which contains dict.
        """
        train_data = self._extract_images(self.train_data_filename_)
        train_labels = self._extract_labels(self.train_labels_filename_)
        for data, label in zip(train_data, train_labels):
            _, img = cv_import.imencode(".jpeg", data)
            yield {"label": int(label), "data": img.tobytes()}

    def _mnist_test_iterator(self):
        """
        get data from mnist test data and label file.

        Yields:
            data (dict of list): mnist data list which contains dict.
        """
        test_data = self._extract_images(self.test_data_filename_)
        test_labels = self._extract_labels(self.test_labels_filename_)
        for data, label in zip(test_data, test_labels):
            _, img = cv_import.imencode(".jpeg", data)
            yield {"label": int(label), "data": img.tobytes()}

    def _transform_train(self):
        """
        Execute transformation from Mnist train part to MindRecord.

        Returns:
            MSRStatus, whether successfully written into MindRecord.
        """
        t0_total = time.time()

        logger.info("transformed MindRecord schema is: {}".format(self.mnist_schema_json))

        # set the header size
        self.writer_train.set_header_size(1 << 24)

        # set the page size
        self.writer_train.set_page_size(1 << 26)

        # create the schema
        self.writer_train.add_schema(self.mnist_schema_json, "mnist_schema")
        # add the index
        self.writer_train.add_index(["label"])

        train_iter = self._mnist_train_iterator()
        batch_size = 256
        transform_count = 0
        while True:
            data_list = []
            try:
                for _ in range(batch_size):
                    data_list.append(train_iter.__next__())
                    transform_count += 1
                self.writer_train.write_raw_data(data_list)
                logger.info("transformed {} record...".format(transform_count))
            except StopIteration:
                if data_list:
                    self.writer_train.write_raw_data(data_list)
                    logger.info("transformed {} record...".format(transform_count))
                break

        ret = self.writer_train.commit()

        t1_total = time.time()
        logger.info("--------------------------------------------")
        logger.info("Total time [train]: {}".format(t1_total - t0_total))
        logger.info("--------------------------------------------")

        return ret

    def _transform_test(self):
        """
        Execute transformation from Mnist test part to MindRecord.

        Returns:
            MSRStatus, whether Mnist is successfully transformed to MindRecord.
        """
        t0_total = time.time()

        logger.info("transformed MindRecord schema is: {}".format(self.mnist_schema_json))

        # set the header size
        self.writer_test.set_header_size(1 << 24)

        # set the page size
        self.writer_test.set_page_size(1 << 26)

        # create the schema
        self.writer_test.add_schema(self.mnist_schema_json, "mnist_schema")

        # add the index
        self.writer_test.add_index(["label"])

        train_iter = self._mnist_test_iterator()
        batch_size = 256
        transform_count = 0
        while True:
            data_list = []
            try:
                for _ in range(batch_size):
                    data_list.append(train_iter.__next__())
                    transform_count += 1
                self.writer_test.write_raw_data(data_list)
                logger.info("transformed {} record...".format(transform_count))
            except StopIteration:
                if data_list:
                    self.writer_test.write_raw_data(data_list)
                    logger.info("transformed {} record...".format(transform_count))
                break

        ret = self.writer_test.commit()

        t1_total = time.time()
        logger.info("--------------------------------------------")
        logger.info("Total time [test]: {}".format(t1_total - t0_total))
        logger.info("--------------------------------------------")

        return ret
