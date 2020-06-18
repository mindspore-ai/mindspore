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
MINDRECORD_FILE = "../data/mindrecord/testCsv/csv.mindrecord"
PARTITION_NUMBER = 4

@pytest.fixture(name="remove_mindrecord_file")
def fixture_remove():
    """add/remove file"""
    def remove_one_file(x):
        if os.path.exists(x):
            os.remove(x)
    def remove_file():
        x = MINDRECORD_FILE
        remove_one_file(x)
        x = MINDRECORD_FILE + ".db"
        remove_one_file(x)
        for i in range(PARTITION_NUMBER):
            x = MINDRECORD_FILE + str(i)
            remove_one_file(x)
            x = MINDRECORD_FILE + str(i) + ".db"
            remove_one_file(x)

    remove_file()
    yield "yield_fixture_data"
    remove_file()

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
    """test transform csv  to mindrecord."""
    csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE, partition_number=PARTITION_NUMBER)
    csv_trans.transform()
    for i in range(PARTITION_NUMBER):
        assert os.path.exists(MINDRECORD_FILE + str(i))
        assert os.path.exists(MINDRECORD_FILE + str(i) + ".db")
    read(MINDRECORD_FILE + "0", ["Age", "EmployNumber", "Name", "Sales", "Over18"], 5)

def test_csv_to_mindrecord_with_columns(remove_mindrecord_file):
    """test transform csv  to mindrecord."""
    csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE, columns_list=['Age', 'Sales'], partition_number=PARTITION_NUMBER)
    csv_trans.transform()
    for i in range(PARTITION_NUMBER):
        assert os.path.exists(MINDRECORD_FILE + str(i))
        assert os.path.exists(MINDRECORD_FILE + str(i) + ".db")
    read(MINDRECORD_FILE + "0", ["Age", "Sales"], 5)

def test_csv_to_mindrecord_with_no_exist_columns(remove_mindrecord_file):
    """test transform csv  to mindrecord."""
    with pytest.raises(Exception, match="The parameter columns_list is illegal, column ssales does not exist."):
        csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE, columns_list=['Age', 'ssales'],
                            partition_number=PARTITION_NUMBER)
        csv_trans.transform()

def test_csv_partition_number_with_illegal_columns(remove_mindrecord_file):
    """
    test transform csv  to mindrecord
    """
    with pytest.raises(Exception, match="The parameter columns_list must be list of str."):
        csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE, ["Sales", 2])
        csv_trans.transform()


def test_csv_to_mindrecord_default_partition_number(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    when partition number is default.
    """
    csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE)
    csv_trans.transform()
    assert os.path.exists(MINDRECORD_FILE)
    assert os.path.exists(MINDRECORD_FILE + ".db")
    read(MINDRECORD_FILE, ["Age", "EmployNumber", "Name", "Sales", "Over18"], 5)

def test_csv_partition_number_0(remove_mindrecord_file):
    """
    test transform csv  to mindrecord
    when partition number is 0.
    """
    with pytest.raises(Exception, match="Invalid parameter value"):
        csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE, None, 0)
        csv_trans.transform()

def test_csv_to_mindrecord_partition_number_none(remove_mindrecord_file):
    """
    test transform csv to mindrecord
    when partition number is none.
    """
    with pytest.raises(Exception,
                       match="The parameter partition_number must be int"):
        csv_trans = CsvToMR(CSV_FILE, MINDRECORD_FILE, None, None)
        csv_trans.transform()

def test_csv_to_mindrecord_illegal_filename(remove_mindrecord_file):
    """
    test transform csv  to mindrecord
    when file name contains illegal character.
    """
    filename = "not_*ok"
    with pytest.raises(Exception, match="File name should not contains"):
        csv_trans = CsvToMR(CSV_FILE, filename)
        csv_trans.transform()
