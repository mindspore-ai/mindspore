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
# ============================================================================
"""test csv to mindrecord tool"""
import os
from importlib import import_module
import pytest

from mindspore import log as logger
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import CsvToMR

try:
    pd = import_module('pandas')
except ModuleNotFoundError:
    pd = None

CSV_FILE = "../data/mindrecord/testCsv/data.csv"
CSV_FILE2 = "../data/mindrecord/testCsv/data2.csv"
PARTITION_NUMBER = 4

@pytest.fixture(name="remove_mindrecord_file")
def fixture_remove():
    """add/remove file"""
    def remove_one_file(x):
        if os.path.exists(x):
            os.remove(x)
    def remove_file(file_name):
        x = file_name
        remove_one_file(x)
        x = file_name + ".db"
        remove_one_file(x)
        for i in range(PARTITION_NUMBER):
            x = file_name + str(i)
            remove_one_file(x)
            x = file_name + str(i) + ".db"
            remove_one_file(x)

    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_file(file_name)
    yield "yield_fixture_data"
    remove_file(file_name)

def read(filename, columns, row_num):
    """test file reade"""
    if not pd:
        raise Exception("Module pandas is not found, please use pip install it.")
    df = pd.read_csv(CSV_FILE)
    count = 0
    reader = FileReader(filename)
    for _, x in enumerate(reader.get_next()):
        for  col in columns:
            assert x[col] == df[col].iloc[count]
        assert len(x) == len(columns)
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == row_num
    reader.close()

def test_csv_to_mindrecord(remove_mindrecord_file):
    """test transform csv to mindrecord."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    csv_trans = CsvToMR(CSV_FILE, file_name, partition_number=PARTITION_NUMBER)
    csv_trans.transform()
    for i in range(PARTITION_NUMBER):
        assert os.path.exists(file_name + str(i))
        assert os.path.exists(file_name + str(i) + ".db")
    read(file_name + "0", ["Age", "EmployNumber", "Name", "Sales", "Over18"], 5)

def test_csv_to_mindrecord_with_columns(remove_mindrecord_file):
    """test transform csv to mindrecord."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    csv_trans = CsvToMR(CSV_FILE, file_name, columns_list=['Age', 'Sales'], partition_number=PARTITION_NUMBER)
    csv_trans.transform()
    for i in range(PARTITION_NUMBER):
        assert os.path.exists(file_name + str(i))
        assert os.path.exists(file_name + str(i) + ".db")
    read(file_name + "0", ["Age", "Sales"], 5)

def test_csv_to_mindrecord_with_no_exist_columns(remove_mindrecord_file):
    """test transform csv to mindrecord."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception, match="The parameter columns_list is illegal, column ssales does not exist."):
        csv_trans = CsvToMR(CSV_FILE, file_name, columns_list=['Age', 'ssales'],
                            partition_number=PARTITION_NUMBER)
        csv_trans.transform()

def test_csv_partition_number_with_illegal_columns(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception, match="The parameter columns_list must be list of str."):
        csv_trans = CsvToMR(CSV_FILE, file_name, ["Sales", 2])
        csv_trans.transform()


def test_csv_to_mindrecord_default_partition_number(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    when partition number is default.
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    csv_trans = CsvToMR(CSV_FILE, file_name)
    csv_trans.transform()
    assert os.path.exists(file_name)
    assert os.path.exists(file_name + ".db")
    read(file_name, ["Age", "EmployNumber", "Name", "Sales", "Over18"], 5)

def test_csv_partition_number_0(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    when partition number is 0.
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception, match="Invalid parameter value"):
        csv_trans = CsvToMR(CSV_FILE, file_name, None, 0)
        csv_trans.transform()

def test_csv_to_mindrecord_partition_number_none(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    when partition number is none.
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception,
                       match="The parameter partition_number must be int"):
        csv_trans = CsvToMR(CSV_FILE, file_name, None, None)
        csv_trans.transform()

def test_csv_to_mindrecord_illegal_filename(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    when file name contains illegal character.
    """
    filename = "not_*ok"
    with pytest.raises(Exception, match="File name should not contains"):
        csv_trans = CsvToMR(CSV_FILE, filename)
        csv_trans.transform()


def test_csv_to_mindrecord_illegal_colname(remove_mindrecord_file):
    """
    Feature: test transform csv to mindrecord
    Description: when columna name start with a number.
    Expectation: success
    """
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with pytest.raises(Exception, match="The first line content: 1 of the CSV file is used as a column name and "
                                        "does not allow starting with a number"):
        csv_trans = CsvToMR(CSV_FILE2, file_name)
        csv_trans.transform()
