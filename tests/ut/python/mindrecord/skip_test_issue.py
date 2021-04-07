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
"""test issue"""
import os
import pytest
from utils import get_data, get_nlp_data, get_mkv_data

from mindspore import log as logger
from mindspore.mindrecord import FileWriter, FileReader
from mindspore.mindrecord import MRMAddIndexError
from mindspore.mindrecord import ShardHeader, SUCCESS, FAILED, ParamValueError
from mindspore.mindrecord import ShardWriter, ShardIndexGenerator, ShardReader

FILES_NUM = 4
CV_FILE_NAME = "./imagenet.mindrecord"
NLP_FILE_NAME = "./aclImdb.mindrecord"
MKV_FILE_NAME = "./vehPer.mindrecord"


def test_cv_file_writer_default_shard_num():
    """test cv dataset writer when shard_num is default value."""
    writer = FileWriter(CV_FILE_NAME)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()
    reader = FileReader(CV_FILE_NAME)
    for index, x in enumerate(reader.get_next()):
        logger.info("#item{}: {}".format(index, x))
    reader.close()

    os.remove("{}".format(CV_FILE_NAME))
    os.remove("{}.db".format(CV_FILE_NAME))


def test_cv_file_writer_shard_num_10():
    """test cv dataset writer when shard_num equals 10."""
    shard_num = 10
    writer = FileWriter(CV_FILE_NAME, shard_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()
    reader = FileReader(CV_FILE_NAME + "0")
    for index, x in enumerate(reader.get_next()):
        logger.info("#item{}: {}".format(index, x))
    reader.close()

    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(shard_num)]
    for item in paths:
        os.remove("{}".format(item))
        os.remove("{}.db".format(item))


def test_cv_file_writer_file_name_none():
    """test cv dataset writer when file_name is none."""
    with pytest.raises(Exception) as e:
        FileWriter(None)
    assert str(e.value) == "[ParamValueError]: error_code: 1347686402, " \
                           "error_msg: Invalid parameter value." \
                           " File path is not allowed None or empty!"


def test_cv_file_writer_file_name_null():
    """test cv dataset writer when file_name is empty string."""
    with pytest.raises(Exception) as e:
        FileWriter("")
    assert str(e.value) == "[ParamValueError]: error_code: 1347686402, " \
                           "error_msg: Invalid parameter value." \
                           " File path is not allowed None or empty!"


def test_cv_file_writer_shard_number_less_1():
    """test cv dataset writer when shard_num is less than 1."""
    with pytest.raises(Exception) as e:
        FileWriter("test.mindrecord", 0)
    assert "[ParamValueError]: error_code: 1347686402, " \
           "error_msg: Invalid parameter value. " \
           "Shard number should " in str(e.value)


def test_cv_file_writer_shard_number_more_1000():
    """test cv dataset writer when shard_num is greater than 1000."""
    with pytest.raises(Exception) as e:
        FileWriter("test.mindrecord", 1001)
    assert "[ParamValueError]: error_code: 1347686402, " \
           "error_msg: Invalid parameter value. " \
           "Shard number should " in str(e.value)


def test_add_empty_schema():
    """test schema add when schema is empty."""
    header = ShardHeader()
    schema = {}
    desc = "test_schema"
    with pytest.raises(Exception) as e:
        schema = header.build_schema(schema, ["data"], desc)
        schema_id = header.add_schema(schema)  # add schema
        assert schema_id == -1
    assert str(e.value) == "[MRMBuildSchemaError]: error_code: 1347690609, " \
                           "error_msg: Failed to build schema."


def test_add_schema_without_desc():
    """test schema add without desc."""
    header = ShardHeader()
    schema_json = {"id_001": {"type": "number"}, "name_002": {"type": "string"},
                   "data_003": {"type": "string"},
                   "label": {"type": "string"}, "key": {"type": "string"}}
    schema = header.build_schema(schema_json, ["data"])
    schema_id = header.add_schema(schema)  # add schema
    assert schema_id == 0


def test_add_empty_index():
    """test index add when index fields is empty string."""
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "number"}}
    header = ShardHeader()
    schema = header.build_schema(schema_json, ["data"], "img")  # create schema
    header.add_schema(schema)  # add schema
    with pytest.raises(Exception, match="incompatible"):
        header.add_index_fields("")


def test_file_writer_fail_add_index():
    """test file writer, read when failed on adding index."""
    data_raw = get_data("../data/mindrecord/testImageNetData/")
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "number"}}
    header = ShardHeader()
    schema = header.build_schema(schema_json, ["data"], "img")  # create schema
    schema_id = header.add_schema(schema)  # add schema
    with pytest.raises(TypeError, match="missing 1 "):
        ret = header.add_index_fields()
        assert ret == FAILED

    with pytest.raises(MRMAddIndexError):
        index_fields = []
        ret = header.add_index_fields(index_fields)
        assert ret == FAILED

    file_name = os.path.join(os.getcwd(), "test_001.mindrecord")  # set output filename
    writer = ShardWriter()  # test_file_writer
    ret = writer.open([file_name])
    assert ret == SUCCESS, 'failed on opening files.'
    ret = writer.set_shard_header(header)  # write header
    assert ret == SUCCESS, 'failed on setting header.'
    ret = writer.write_raw_cv_data({schema_id: data_raw})
    assert ret == SUCCESS, 'failed on writing raw data.'
    ret = writer.commit()  # commit data
    assert ret == SUCCESS, "commit failed"
    # ShardIndexGenerator
    generator = ShardIndexGenerator(os.path.realpath(file_name))
    generator.build()
    generator.write_to_db()

    reader = ShardReader()
    ret = reader.open(file_name)
    reader.launch()
    index = 0
    _, blob_fields = reader.get_blob_fields()
    iterator = reader.get_next()
    while iterator:
        for blob, raw in iterator:
            raw[blob_fields[0]] = bytes(blob)
            logger.info("#item{}: {}".format(index, raw))
            index += 1
            iterator = reader.get_next()
    reader.finish()
    reader.close()

    os.remove("{}".format(file_name))
    os.remove("{}.db".format(file_name))


def test_add_index_with_incorrect_field():
    """test index add with incorrect field(64)."""
    header = ShardHeader()
    mkv_schema_json = {"file_name": {"type": "string"},
                       "id": {"type": "number"}, "prelabel": {"type": "string"}}
    schema = header.build_schema(mkv_schema_json, ["data"], "mkv_schema")
    header.add_schema(schema)
    with pytest.raises(Exception, match="incompatible function arguments"):
        header.add_index_fields([(-1, "id")])


def test_add_index_with_string_list():
    """test index add with list of string(64)."""
    header = ShardHeader()
    schema_json = {"id": {"type": "number"}, "name": {"type": "string"},
                   "label": {"type": "string"}, "key": {"type": "string"}}
    schema = header.build_schema(schema_json, ["key"], "schema_desc")
    header.add_schema(schema)
    ret = header.add_index_fields(["id", "label"])
    assert ret == SUCCESS


def test_add_index_with_dict():
    """test index add when index fields' datatype is dict(64)."""
    writer = FileWriter(MKV_FILE_NAME, FILES_NUM)
    mkv_schema_json = {"file_name": {"type": "string"},
                       "id": {"type": "number"},
                       "prelabel": {"type": "string"},
                       "data": {"type": "bytes"}}
    writer.add_schema(mkv_schema_json, "mkv_schema")
    with pytest.raises(Exception) as e:
        writer.add_index({"file_name": {"type": "string"}})
    assert str(e.value) == "[ParamTypeError]: error_code: 1347686401, " \
                           "error_msg: Invalid parameter type. " \
                           "'index_fields' expect list type."


def test_mkv_file_reader_with_negative_num_consumer():
    """test mkv file reader when the number of consumer is negative."""
    writer = FileWriter(MKV_FILE_NAME, FILES_NUM)
    data = get_mkv_data("../data/mindrecord/testVehPerData/")
    mkv_schema_json = {"file_name": {"type": "string"},
                       "id": {"type": "number"},
                       "prelabel": {"type": "string"},
                       "data": {"type": "bytes"}}
    writer.add_schema(mkv_schema_json, "mkv_schema")
    writer.add_index(["file_name", "prelabel"])
    writer.write_raw_data(data)
    writer.commit()

    with pytest.raises(Exception) as e:
        FileReader(MKV_FILE_NAME + "1", -1)
    assert "Consumer number should between 1 and" in str(e.value)

    paths = ["{}{}".format(MKV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_write_raw_data_with_empty_list():
    """test write raw data with empty list."""
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    ret = writer.write_raw_data([])
    assert ret == SUCCESS
    writer.commit()

    reader = FileReader(CV_FILE_NAME + "0")
    for index, x in enumerate(reader.get_next()):
        logger.info("#item{}: {}".format(index, x))
    reader.close()

    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_issue_38():
    """test cv dataset writer when schema does not match raw data."""
    writer = FileWriter(CV_FILE_NAME, 1)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}}
    writer.add_schema(cv_schema_json, "img_schema")
    with pytest.raises(Exception) as e:
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()
    assert str(e.value) == "[MRMDefineIndexError]: error_code: 1347694794, " \
                           "error_msg: Failed to define index field. " \
                           "Detail: Could not set blob field " \
                           "'file_name' as index field."


def test_issue_39():
    """test cv dataset writer when schema fields' datatype does not match raw data."""
    writer = FileWriter(CV_FILE_NAME, 1)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "number"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(CV_FILE_NAME)
    index = 0
    for _ in reader.get_next():
        index += 1
    assert index == 0, "failed on reading data!"
    reader.close()
    os.remove("{}".format(CV_FILE_NAME))
    os.remove("{}.db".format(CV_FILE_NAME))


def test_issue_40():
    """test cv dataset when write raw data twice."""
    writer = FileWriter(CV_FILE_NAME, 1)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.write_raw_data(data)
    ret = writer.commit()
    assert ret == SUCCESS, 'failed on writing data!'

    os.remove("{}".format(CV_FILE_NAME))
    os.remove("{}.db".format(CV_FILE_NAME))


def test_issue_73():
    """test file reader by column name."""
    writer = FileWriter(MKV_FILE_NAME, FILES_NUM)
    data = get_mkv_data("../data/mindrecord/testVehPerData/")
    mkv_schema_json = {"file_name": {"type": "string"},
                       "id": {"type": "number"},
                       "prelabel": {"type": "string"},
                       "data": {"type": "bytes"}}
    writer.add_schema(mkv_schema_json, "mkv_schema")
    writer.add_index(["file_name", "prelabel"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(MKV_FILE_NAME + "1", 4, ["file_name"])
    for index, x in enumerate(reader.get_next()):
        logger.info("#item{}: {}".format(index, x))
    reader.close()

    paths = ["{}{}".format(MKV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_issue_117():
    """test add schema when field type is incorrect."""
    writer = FileWriter(__file__, FILES_NUM)
    schema = {"id": {"type": "string"}, "label": {"type": "number"},
              "rating": {"type": "number"},
              "input_ids": {"type": "list", "items": {"type": "number"}},
              "input_mask": {"type": "array", "items": {"type": "number"}},
              "segment_ids": {"type": "array", "items": {"type": "number"}}
              }
    with pytest.raises(Exception,
                       match="Field '{'type': 'list', "
                             "'items': {'type': 'number'}}' "
                             "contains illegal attributes"):
        writer.add_schema(schema, "img_schema")


def test_issue_95():
    """test file reader when failed on file write."""
    writer = FileWriter(__file__, FILES_NUM)
    data_raw = get_data("../data/mindrecord/testImageNetData/")
    schema_json = {"file_name": {"type": "number"},
                   "label": {"type": "number"},
                   "data": {"type": "bytes"}, "data1": {"type": "string"}}
    writer.add_schema(schema_json, "img_schema")
    with pytest.raises(MRMAddIndexError):
        writer.add_index(["key"])
    writer.write_raw_data(data_raw, True)
    writer.commit()

    reader = FileReader(__file__ + "1")
    for index, x in enumerate(reader.get_next()):
        logger.info("#item{}: {}".format(index, x))
    reader.close()

    paths = ["{}{}".format(__file__, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_mindrecord_add_index_016():
    """test index add when index fields are incorrect."""
    schema_json = {"id": {"type": "number"}, "name": {"type": "string"},
                   "label": {"type": "string"}, "key": {"type": "string"}}
    header = ShardHeader()
    schema = header.build_schema(schema_json, ["data"], "img")
    header.add_schema(schema)
    index_fields_list = [(0, "id")]
    with pytest.raises(Exception):
        header.add_index_fields(index_fields_list)


def test_mindrecord_add_index_011():
    """test index add"""
    schema_json = {"id": {"type": "number"}, "name": {"type": "string"},
                   "label": {"type": "string"}, "key": {"type": "string"}}
    header = ShardHeader()
    schema = header.build_schema(schema_json, ["data"], "img")  # create schema
    header.add_schema(schema)  # add schema
    index_fields_list = ["id", "name", "label", "key"]
    ret = header.add_index_fields(index_fields_list)
    assert ret == 0, 'failed on adding index fields.'


def test_issue_118():
    """test file writer when raw data do not match schema."""
    shard_num = 4
    writer = FileWriter(CV_FILE_NAME, shard_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    data.append({"file_name": "abcdefg", "label": 11,
                 "data": str(data[0]["data"])})
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(CV_FILE_NAME + "0")
    for index, _ in enumerate(reader.get_next()):
        logger.info(index)
    reader.close()

    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(shard_num)]
    for item in paths:
        os.remove("{}".format(item))
        os.remove("{}.db".format(item))


def test_issue_87():
    """test file writer when data(bytes) do not match field type(string)."""
    shard_num = 4
    writer = FileWriter(CV_FILE_NAME, shard_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "string"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["label"])
    with pytest.raises(Exception, match="data is wrong"):
        writer.write_raw_data(data, False)
        writer.commit()

    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(shard_num)]
    for item in paths:
        os.remove("{}".format(item))


def test_issue_84():
    """test file reader when db does not match."""
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    writer = FileWriter(NLP_FILE_NAME, FILES_NUM)
    data = list(get_nlp_data("../data/mindrecord/testAclImdbData/pos",
                             "../data/mindrecord/testAclImdbData/vocab.txt",
                             10))
    nlp_schema_json = {"id": {"type": "string"}, "label": {"type": "number"},
                       "rating": {"type": "number"},
                       "input_ids": {"type": "array",
                                     "items": {"type": "number"}},
                       "input_mask": {"type": "array",
                                      "items": {"type": "number"}},
                       "segment_ids": {"type": "array",
                                       "items": {"type": "number"}}
                       }
    writer.set_header_size(1 << 14)
    writer.set_page_size(1 << 15)
    writer.add_schema(nlp_schema_json, "nlp_schema")
    writer.add_index(["id", "rating"])
    writer.write_raw_data(data)
    writer.commit()

    reader = ShardReader()
    os.rename("imagenet.mindrecord1.db", "imagenet.mindrecord1.db.bk")
    os.rename("aclImdb.mindrecord1.db", "imagenet.mindrecord1.db")
    file_name = os.path.join(os.getcwd(), "imagenet.mindrecord1")
    with pytest.raises(Exception) as e:
        reader.open(file_name)
    assert str(e.value) == "[MRMOpenError]: error_code: 1347690596, " \
                           "error_msg: " \
                           "MindRecord File could not open successfully."

    os.rename("imagenet.mindrecord1.db", "aclImdb.mindrecord1.db")
    paths = ["{}{}".format(NLP_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for item in paths:
        os.remove("{}".format(item))
        os.remove("{}.db".format(item))

    os.rename("imagenet.mindrecord1.db.bk", "imagenet.mindrecord1.db")
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for item in paths:
        os.remove("{}".format(item))
        os.remove("{}.db".format(item))


def test_issue_65():
    """test file reader when file name is illegal."""
    reader = ShardReader()
    file_name = os.path.join(os.getcwd(), "imagenet.mindrecord01qwert")
    with pytest.raises(Exception) as e:
        reader.open(file_name)
    assert str(e.value) == "[MRMOpenError]: error_code: 1347690596, " \
                           "error_msg: " \
                           "MindRecord File could not open successfully."


def skip_test_issue_155():
    """test file writer loop."""
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name"])
    for _ in range(1000):
        writer.write_raw_data(data)
    writer.commit()
    reader = FileReader(CV_FILE_NAME + "0")
    count = 0
    for _ in reader.get_next():
        count += 1
    assert count == 10000, "Failed to read multiple writed data."


def test_issue_124():
    """test file writer when data(string) do not match field type(bytes)."""
    shard_num = 4
    writer = FileWriter(CV_FILE_NAME, shard_num)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    data.append({"file_name": "abcdefg", "label": 11,
                 "data": str(data[0]["data"])})
    writer.add_schema(cv_schema_json, "img_schema")
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(CV_FILE_NAME + "0")
    for index, _ in enumerate(reader.get_next()):
        logger.info(index)
    reader.close()

    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(shard_num)]
    for item in paths:
        os.remove("{}".format(item))
        os.remove("{}.db".format(item))


def test_issue_36():
    """test file writer when shard num is illegal."""
    with pytest.raises(ParamValueError, match="Shard number should between "):
        writer = FileWriter(CV_FILE_NAME, -1)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    with pytest.raises(UnboundLocalError,
                       match="local variable "
                             "'writer' referenced before assignment"):
        writer.add_schema(cv_schema_json, "cv_schema")
        writer.add_index(["file_name", "label"])
        writer.write_raw_data(data)
        writer.commit()


def test_issue_34():
    """test file writer"""
    writer = FileWriter(CV_FILE_NAME)
    data = get_data("../data/mindrecord/testImageNetData/")
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "number"}, "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "cv_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(CV_FILE_NAME)
    i = 0
    for index, x in enumerate(reader.get_next()):
        logger.info("#item{}: {}".format(index, x))
        i = i + 1
    logger.info("count: {}".format(i))
    reader.close()
    os.remove(CV_FILE_NAME)
    os.remove("{}.db".format(CV_FILE_NAME))


def test_file_writer_raw_data_038():
    """test write raw data without verify."""
    shard_num = 11
    writer = FileWriter("test_file_writer_raw_data_", shard_num)
    data_raw = get_data("../data/mindrecord/testImageNetData/")
    schema_json = {"file_name": {"type": "string"}, "label": {"type": "number"},
                   "data": {"type": "bytes"}}
    writer.add_schema(schema_json, "img_schema")
    writer.add_index(["file_name"])
    for _ in range(shard_num):
        writer.write_raw_data(data_raw, False)
    writer.commit()

    file_name = ""
    if shard_num > 1:
        file_name = '99' if shard_num > 99 else str(shard_num - 1)
    reader = FileReader("test_file_writer_raw_data_" + file_name)
    i = 0
    for _, _ in enumerate(reader.get_next()):
        i = i + 1
    assert i == shard_num * 10
    reader.close()
    if shard_num == 1:
        os.remove("test_file_writer_raw_data_")
        os.remove("test_file_writer_raw_data_.db")
        return
    for x in range(shard_num):
        n = str(x)
        if shard_num > 10:
            n = '0' + str(x) if x < 10 else str(x)
        if os.path.exists("test_file_writer_raw_data_{}".format(n)):
            os.remove("test_file_writer_raw_data_{}".format(n))
        if os.path.exists("test_file_writer_raw_data_{}.db".format(n)):
            os.remove("test_file_writer_raw_data_{}.db".format(n))


def test_more_than_1_bytes_in_schema():
    """test file writer when schema contains multiple 'bytes' fields."""
    schema_json = {"id": {"type": "string"}, "label": {"type": "number"},
                   "rating": {"type": "number"},
                   "input_ids": {"type": "bytes"},
                   "input_mask": {"type": "bytes"},
                   "segment_ids": {"type": "array",
                                   "items": {"type": "number"}}
                   }
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    writer.add_schema(schema_json, "img_schema")


def test_shard_4_raw_data_1():
    """test file writer when shard_num equals 4 and number of sample equals 1."""
    writer = FileWriter(CV_FILE_NAME, FILES_NUM)
    schema_json = {"file_name": {"type": "string"},
                   "label": {"type": "number"}}
    writer.add_schema(schema_json, "img_schema")
    writer.add_index(["label"])
    data = [{"file_name": "001.jpg", "label": 1}]
    writer.write_raw_data(data)
    writer.commit()

    reader = FileReader(CV_FILE_NAME + "0")
    count = 0
    for index, x in enumerate(reader.get_next()):
        assert len(x) == 2
        count = count + 1
        logger.info("#item{}: {}".format(index, x))
    assert count == 1
    reader.close()
    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))
