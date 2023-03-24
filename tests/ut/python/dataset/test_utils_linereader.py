# Copyright 2023 Huawei Technologies Co., Ltd
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
Test LineReader operations
"""
import pytest
import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR = "../data/dataset/testLineReader"


def test_parameter_exception():
    """
    Feature: LineReader
    Description: Test LineReader with parameter exception
    Expectation: SUCCESS
    """
    logger.info("Test LineReader with parameter exception")
    filename = DATA_DIR + "/null.txt"
    with pytest.raises(RuntimeError) as e:
        ds.LineReader(filename)
    assert "does not exist" in str(e)

    with pytest.raises(RuntimeError) as e:
        ds.LineReader(DATA_DIR)
    assert "is not a regular file" in str(e)

    with pytest.raises(TypeError) as e:
        ds.LineReader(123)
    assert "is not string" in str(e)


def test_empty_file():
    """
    Feature: LineReader
    Description: Test LineReader with empty file
    Expectation: SUCCESS
    """
    logger.info("Test LineReader with empty file")
    filename = DATA_DIR + "/empty.txt"
    fo = ds.LineReader(filename)
    assert fo.len() == 0
    with pytest.raises(ValueError) as e:
        fo.readline(1)
    assert "is not within the required interval" in str(e)
    fo.close()


def test_json_input():
    """
    Feature: LineReader
    Description: Test LineReader with JSON line input in file
    Expectation: SUCCESS
    """
    logger.info("Test LineReader with JSON line input")
    filename = DATA_DIR + "/json.txt"
    fo = ds.LineReader(filename)
    assert fo.len() == 5
    assert len(fo) == 5
    assert fo.readline(1) == '{"bbox": [100, 200, 300, 400], "file_name": "001.jpg"}'
    assert fo[2] == '{"bbox": [200, 200, 400, 400], "file_name": "002.jpg"}'
    assert fo.readline(3) == '{"bbox": [300, 200, 500, 400], "file_name": "003.jpg"}'
    assert fo[4] == '{"bbox": [400, 200, 600, 400], "file_name": "004.jpg"}'
    assert fo.readline(5) == '{"bbox": [500, 200, 700, 400], "file_name": "005.jpg"}'

    with pytest.raises(TypeError) as e:
        fo.readline("1")
    assert "is not of type" in str(e)

    with pytest.raises(ValueError) as e:
        fo.readline(0)
    assert "is not within the required interval of" in str(e)

    with pytest.raises(ValueError) as e:
        fo.readline(-1)
    assert "is not within the required interval of" in str(e)

    with pytest.raises(ValueError) as e:
        fo.readline(6)
    assert "is not within the required interval of" in str(e)

    fo.close()
