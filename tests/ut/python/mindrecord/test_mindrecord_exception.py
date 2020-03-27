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
# ============================================================================
"""test mindrecord exception"""
import os
import pytest
from mindspore.mindrecord import FileWriter, FileReader, MindPage
from mindspore.mindrecord import MRMOpenError, MRMGenerateIndexError, ParamValueError
from mindspore import log as logger
from utils import get_data

CV_FILE_NAME = "./imagenet.mindrecord"
NLP_FILE_NAME = "./aclImdb.mindrecord"
FILES_NUM = 4

def test_cv_file_writer_shard_num_none():
    """test cv file writer when shard num is None."""
    with pytest.raises(Exception, match="Shard num is illegal."):
        FileWriter("/tmp/123454321", None)

def test_cv_file_writer_shard_num_str():
    """test cv file writer when shard num is string."""
    with pytest.raises(Exception, match="Shard num is illegal."):
        FileWriter("/tmp/123454321", "20")

def test_cv_page_reader_consumer_num_none():
    """test cv page reader when consumer number is None."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        MindPage(CV_FILE_NAME + "0", None)

def test_cv_page_reader_consumer_num_str():
    """test cv page reader when consumer number is string."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        MindPage(CV_FILE_NAME + "0", "2")

def test_nlp_file_reader_consumer_num_none():
    """test nlp file reader when consumer number is None."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        FileReader(NLP_FILE_NAME + "0", None)

def test_nlp_file_reader_consumer_num_str():
    """test nlp file reader when consumer number is string."""
    with pytest.raises(Exception, match="Consumer number is illegal."):
        FileReader(NLP_FILE_NAME + "0", "4")

def create_cv_mindrecord(files_num):
    writer = FileWriter(CV_FILE_NAME, files_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int64"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

def test_lack_partition_and_db():
    """test file reader when mindrecord file does not exist."""
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader('dummy.mindrecord')
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)

def test_lack_db():
    """test file reader when db file does not exist."""
    create_cv_mindrecord(1)
    os.remove("{}.db".format(CV_FILE_NAME))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME)
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    os.remove(CV_FILE_NAME)

def test_lack_some_partition_and_db():
    """test file reader when some partition and db do not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[3]))
    os.remove("{}.db".format(paths[3]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

def test_lack_some_partition_first():
    """test file reader when first partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[0]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

def test_lack_some_partition_middle():
    """test file reader when some partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[1]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

def test_lack_some_partition_last():
    """test file reader when last partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[3]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

def test_mindpage_lack_some_partition():
    """test page reader when some partition does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}".format(paths[0]))
    with pytest.raises(MRMOpenError) as err:
        MindPage(CV_FILE_NAME + "0")
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

def test_lack_some_db():
    """test file reader when some db does not exist."""
    create_cv_mindrecord(4)
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    os.remove("{}.db".format(paths[3]))
    with pytest.raises(MRMOpenError) as err:
        reader = FileReader(CV_FILE_NAME + "0")
        reader.close()
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

def test_invalid_mindrecord():
    """test file reader when the content of mindrecord is illegal."""
    with open(CV_FILE_NAME, 'w') as f:
        dummy = 's' * 100
        f.write(dummy)
    with pytest.raises(MRMOpenError) as err:
        FileReader(CV_FILE_NAME)
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    os.remove(CV_FILE_NAME)

def test_invalid_db():
    """test file reader when the content of db is illegal."""
    create_cv_mindrecord(1)
    os.remove("imagenet.mindrecord.db")
    with open('imagenet.mindrecord.db', 'w') as f:
        f.write('just for test')
    with pytest.raises(MRMOpenError) as err:
        FileReader('imagenet.mindrecord')
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    os.remove("imagenet.mindrecord")
    os.remove("imagenet.mindrecord.db")

def test_overwrite_invalid_mindrecord():
    """test file writer when overwrite invalid mindreocrd file."""
    with open(CV_FILE_NAME, 'w') as f:
        f.write('just for test')
    with pytest.raises(MRMOpenError) as err:
        create_cv_mindrecord(1)
    assert '[MRMOpenError]: error_code: 1347690596, ' \
           'error_msg: MindRecord File could not open successfully.'\
           in str(err.value)
    os.remove(CV_FILE_NAME)

def test_overwrite_invalid_db():
    """test file writer when overwrite invalid db file."""
    with open('imagenet.mindrecord.db', 'w') as f:
        f.write('just for test')
    with pytest.raises(MRMGenerateIndexError) as err:
        create_cv_mindrecord(1)
    assert '[MRMGenerateIndexError]: error_code: 1347690612, ' \
           'error_msg: Failed to generate index.' in str(err.value)
    os.remove("imagenet.mindrecord")
    os.remove("imagenet.mindrecord.db")

def test_read_after_close():
    """test file reader when close read."""
    create_cv_mindrecord(1)
    reader = FileReader(CV_FILE_NAME)
    reader.close()
    count = 0
    for index, x in enumerate(reader.get_next()):
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 0
    os.remove(CV_FILE_NAME)
    os.remove("{}.db".format(CV_FILE_NAME))

def test_file_read_after_read():
    """test file reader when finish read."""
    create_cv_mindrecord(1)
    reader = FileReader(CV_FILE_NAME)
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 3
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 10
    reader.close()
    cnt = 0
    for index, x in enumerate(reader.get_next()):
        cnt = cnt + 1
        logger.info("#item{}: {}".format(index, x))
    assert cnt == 0
    os.remove(CV_FILE_NAME)
    os.remove("{}.db".format(CV_FILE_NAME))

def test_cv_file_writer_shard_num_greater_than_1000():
    """test cv file writer shard number greater than 1000."""
    with pytest.raises(ParamValueError) as err:
        FileWriter(CV_FILE_NAME, 1001)
    assert 'Shard number should between' in str(err.value)
