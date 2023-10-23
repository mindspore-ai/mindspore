# Copyright 2020 Huawei Technologies Co., Ltd
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
Csv format convert tool for MindRecord.
"""
from importlib import import_module
import os

from mindspore import log as logger
from ..filewriter import FileWriter
from ..shardutils import check_filename, ExceptionThread

try:
    pd = import_module("pandas")
except ModuleNotFoundError:
    pd = None

__all__ = ['CsvToMR']


class CsvToMR:
    """
    A class to transform from csv to MindRecord.

    Args:
        source (str): The file path of csv.
        destination (str): The MindRecord file path to transform into, ensure that the directory is created in advance
            and no file with the same name exists in the directory.
        columns_list(list[str], optional): A list of columns to be read. Default: ``None`` .
        partition_number (int, optional): The partition size, Default: ``1`` .

    Raises:
        ValueError: If `source` , `destination` , `partition_number` is invalid.
        RuntimeError: If `columns_list` is invalid.

    Examples:
        >>> from mindspore.mindrecord import CsvToMR
        >>>
        >>> csv_file = "/path/to/csv/file"
        >>> mindrecord_file = "/path/to/mindrecord/file"
        >>> csv_to_mr = CsvToMR(csv_file, mindrecord_file)
        >>> status = csv_to_mr.transform()
    """

    def __init__(self, source, destination, columns_list=None, partition_number=1):
        if not pd:
            raise Exception("Module pandas is not found, please use pip install it.")
        if isinstance(source, str):
            check_filename(source)
            self.source = source
        else:
            raise ValueError("The parameter source must be str.")

        self._check_columns(columns_list, "columns_list")
        self.columns_list = columns_list

        if isinstance(destination, str):
            check_filename(destination)
            self.destination = destination
        else:
            raise ValueError("The parameter destination must be str.")

        if partition_number is not None:
            if not isinstance(partition_number, int):
                raise ValueError("The parameter partition_number must be int")
            self.partition_number = partition_number
        else:
            raise ValueError("The parameter partition_number must be int")

        self.writer = FileWriter(self.destination, self.partition_number)

    def _check_columns(self, columns, columns_name):
        """
        Validate the columns of csv
        """
        if not columns:
            return
        if isinstance(columns, list):
            for col in columns:
                if not isinstance(col, str):
                    raise ValueError("The parameter {} must be list of str.".format(columns_name))
        else:
            raise ValueError("The parameter {} must be list of str.".format(columns_name))

    def _get_schema(self, df):
        """
        Construct schema from df columns
        """
        if self.columns_list:
            for col in self.columns_list:
                if col not in df.columns:
                    raise RuntimeError("The parameter columns_list is illegal, column {} does not exist.".format(col))
        else:
            self.columns_list = df.columns

        schema = {}
        for col in self.columns_list:
            if str(df[col].dtype) == 'int64':
                schema[col] = {"type": "int64"}
            elif str(df[col].dtype) == 'float64':
                schema[col] = {"type": "float64"}
            elif str(df[col].dtype) == 'bool':
                schema[col] = {"type": "int32"}
            else:
                schema[col] = {"type": "string"}
        if not schema:
            raise RuntimeError("Failed to generate schema from csv file.")
        return schema

    def _get_row_of_csv(self, df, columns_list):
        """Get row data from csv file."""
        for _, r in df.iterrows():
            row = {}
            for col in columns_list:
                if str(df[col].dtype) == 'bool':
                    row[col] = int(r[col])
                else:
                    row[col] = r[col]
            yield row

    # pylint: disable=missing-docstring
    def run(self):
        if not os.path.exists(self.source):
            raise IOError("Csv file {} do not exist.".format(self.source))

        pd.set_option('display.max_columns', None)
        df = pd.read_csv(self.source)

        csv_schema = self._get_schema(df)

        logger.info("transformed MindRecord schema is: {}".format(csv_schema))

        # set the header size
        self.writer.set_header_size(1 << 24)

        # set the page size
        self.writer.set_page_size(1 << 26)

        # create the schema
        self.writer.add_schema(csv_schema, "csv_schema")

        # add the index
        for column_name in list(self.columns_list):
            if str(column_name).isdigit():
                raise ValueError("The first line content: {} of the CSV file is used as a column name and "
                                 "does not allow starting with a number.".format(str(column_name)))
        self.writer.add_index(list(self.columns_list))

        csv_iter = self._get_row_of_csv(df, self.columns_list)
        batch_size = 256
        transform_count = 0
        while True:
            data_list = []
            try:
                for _ in range(batch_size):
                    data_list.append(csv_iter.__next__())
                    transform_count += 1
                self.writer.write_raw_data(data_list)
                logger.info("transformed {} record...".format(transform_count))
            except StopIteration:
                if data_list:
                    self.writer.write_raw_data(data_list)
                    logger.info(
                        "transformed {} record...".format(transform_count))
                break

        ret = self.writer.commit()

        return ret

    def transform(self):
        """
        Execute transformation from csv to MindRecord.

        Note:
            Please refer to the Examples of :class:`mindspore.mindrecord.CsvToMR` .

        Returns:
            MSRStatus, SUCCESS or FAILED.

        Raises:
            ParamTypeError: If index field is invalid.
            MRMOpenError: If failed to open MindRecord file.
            MRMValidateDataError: If data does not match blob fields.
            MRMSetHeaderError: If failed to set header.
            MRMWriteDatasetError: If failed to write dataset.
            IOError: Csv file does not exist.
            ValueError: The first line of the CSV file is used as column name and each field cannot start with a number.
        """

        t = ExceptionThread(target=self.run)
        t.daemon = True
        t.start()
        t.join()
        if t.exitcode != 0:
            raise t.exception
        return t.res
