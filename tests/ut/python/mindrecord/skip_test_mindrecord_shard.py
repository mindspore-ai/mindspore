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
"""test internal shard api"""
import os
import random
from mindspore.mindrecord import ShardHeader, SUCCESS
from mindspore.mindrecord import ShardWriter, ShardIndexGenerator, ShardReader, ShardSegment
from mindspore import log as logger
from utils import get_data, get_nlp_data, get_mkv_data

FILES_NUM = 4
CV_FILE_NAME = "./imagenet.mindrecord"
NLP_FILE_NAME = "./aclImdb.mindrecord"
MKV_FILE_NAME = "./vehPer.mindrecord"


def test_nlp_file_writer():
    """test nlp file writer using shard api"""
    schema_json = {"id": {"type": "string"}, "label": {"type": "number"},
                   "rating": {"type": "number"},
                   "input_ids": {"type": "array",
                                 "items": {"type": "number"}},
                   "input_mask": {"type": "array",
                                  "items": {"type": "number"}},
                   "segment_ids": {"type": "array",
                                   "items": {"type": "number"}}
                   }
    data = list(get_nlp_data("../data/mindrecord/testAclImdbData/pos",
                             "../data/mindrecord/testAclImdbData/vocab.txt",
                             10))
    header = ShardHeader()
    schema = header.build_schema(schema_json, ["segment_ids"], "nlp_schema")
    schema_id = header.add_schema(schema)
    assert schema_id == 0, 'failed on adding schema'
    index_fields_list = ["id", "rating"]
    ret = header.add_index_fields(index_fields_list)
    assert ret == SUCCESS, 'failed on adding index fields.'
    writer = ShardWriter()
    paths = ["{}{}".format(NLP_FILE_NAME, x) for x in range(FILES_NUM)]
    ret = writer.open(paths)
    assert ret == SUCCESS, 'failed on opening files.'
    writer.set_header_size(1 << 14)
    writer.set_page_size(1 << 15)
    ret = writer.set_shard_header(header)
    assert ret == SUCCESS, 'failed on setting header.'
    ret = writer.write_raw_nlp_data({schema_id: data})
    assert ret == SUCCESS, 'failed on writing raw data.'
    ret = writer.commit()
    assert ret == SUCCESS, 'failed on committing.'
    generator = ShardIndexGenerator(os.path.realpath(paths[0]))
    generator.build()
    generator.write_to_db()


def test_nlp_file_reader():
    """test nlp file reader using shard api"""
    dataset = ShardReader()
    dataset.open(NLP_FILE_NAME + "0")
    dataset.launch()
    index = 0
    iterator = dataset.get_next()
    while iterator:
        for _, raw in iterator:
            logger.info("#item{}: {}".format(index, raw))
            index += 1
            iterator = dataset.get_next()
    dataset.finish()
    dataset.close()


def test_nlp_page_reader():
    """test nlp page reader using shard api"""
    reader = ShardSegment()
    reader.open(NLP_FILE_NAME + "0")

    fields = reader.get_category_fields()
    logger.info("fields: {}".format(fields))

    ret = reader.set_category_field("rating")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    img1 = reader.read_at_page_by_id(0, 0, 1)
    logger.info("img1 len: {}, img1[0] len: {}, img1[0]: {}".format(len(img1), len(img1[0]), img1[0]))

    img2 = reader.read_at_page_by_name("7", 0, 1)
    logger.info("img2 len: {}, img2[0] len: {}, img2[0]: {}".format(len(img2), len(img2[0]), img2[0]))

    paths = ["{}{}".format(NLP_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_cv_file_writer():
    """test cv file reader using shard api"""
    img_schema_json = {"file_name": {"type": "string"},
                       "label": {"type": "number"}}
    data = get_data("../data/mindrecord/testImageNetData/")

    header = ShardHeader()
    img_schema = header.build_schema(img_schema_json, ["data"], "img_schema")
    schema_id = header.add_schema(img_schema)
    assert schema_id == 0, 'failed on building schema.'
    index_fields_list = ["file_name", "label"]
    ret = header.add_index_fields(index_fields_list)
    assert ret == SUCCESS, 'failed on adding index fields.'

    writer = ShardWriter()
    paths = ["{}{}".format(CV_FILE_NAME, x) for x in range(FILES_NUM)]
    ret = writer.open(paths)
    assert ret == SUCCESS, 'failed on opening files.'
    writer.set_header_size(1 << 24)
    writer.set_page_size(1 << 25)
    ret = writer.set_shard_header(header)
    assert ret == SUCCESS, 'failed on setting header.'
    ret = writer.write_raw_cv_data({schema_id: data})
    assert ret == SUCCESS, 'failed on writing raw data.'
    ret = writer.commit()
    assert ret == SUCCESS, 'failed on committing.'
    # ShardIndexGenerator
    generator = ShardIndexGenerator(os.path.abspath(paths[0]))
    generator.build()
    generator.write_to_db()


def test_cv_file_reader():
    """test cv file reader using shard api"""
    dataset = ShardReader()
    dataset.open(CV_FILE_NAME + "0")
    dataset.launch()
    index = 0
    _, blob_fields = dataset.get_blob_fields()
    iterator = dataset.get_next()
    while iterator:
        for blob, raw in iterator:
            raw[blob_fields[0]] = bytes(blob)
            logger.info("#item{}: {}".format(index, raw))
            index += 1
            iterator = dataset.get_next()
    dataset.finish()
    dataset.close()


def test_cv_page_reader():
    """test cv page reader using shard api"""
    reader = ShardSegment()
    reader.open(CV_FILE_NAME + "0")
    fields = reader.get_category_fields()
    logger.info("fields: {}".format(fields))

    ret = reader.set_category_field("label")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    img1 = reader.read_at_page_by_id(0, 0, 1)
    logger.info("img1 len: {}, img1[0] len: {}".format(len(img1), len(img1[0])))

    img2 = reader.read_at_page_by_name("822", 0, 1)
    logger.info("img2 len: {}, img2[0] len: {}".format(len(img2), len(img2[0])))

    paths = ["{}{}".format(CV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_mkv_file_writer():
    """test mkv file writer  using shard api"""
    data = get_mkv_data("../data/mindrecord/testVehPerData/")
    schema_json = {"file_name": {"type": "string"}, "id": {"type": "number"},
                   "prelabel": {"type": "string"}}
    header = ShardHeader()
    img_schema = header.build_schema(schema_json, ["data"], "img_schema")
    schema_id = header.add_schema(img_schema)
    assert schema_id == 0, 'failed on building schema.'
    index_fields_list = ["id", "file_name"]
    ret = header.add_index_fields(index_fields_list)
    assert ret == SUCCESS, 'failed on adding index fields.'

    writer = ShardWriter()
    paths = ["{}{}".format(MKV_FILE_NAME, x) for x in range(FILES_NUM)]
    ret = writer.open(paths)
    assert ret == SUCCESS, 'failed on opening files.'
    writer.set_header_size(1 << 24)
    writer.set_page_size(1 << 25)
    ret = writer.set_shard_header(header)
    assert ret == SUCCESS, 'failed on setting header.'
    ret = writer.write_raw_cv_data({schema_id: data})
    assert ret == SUCCESS, 'failed on writing raw data.'
    ret = writer.commit()
    assert ret == SUCCESS, 'failed on committing.'

    generator = ShardIndexGenerator(os.path.realpath(paths[0]))
    generator.build()
    generator.write_to_db()


def test_mkv_page_reader():
    """test mkv page reader using shard api"""
    reader = ShardSegment()
    reader.open(MKV_FILE_NAME + "0")

    fields = reader.get_category_fields()
    logger.info("fields: {}".format(fields))

    ret = reader.set_category_field("id")
    assert ret == SUCCESS, 'failed on setting category field.'

    info = reader.read_category_info()
    logger.info("category info: {}".format(info))

    img1 = reader.read_at_page_by_id(0, 0, 1)
    logger.info("img1 len: {}, img1[0] len: {}, img1[0]: {}".format(len(img1), len(img1[0]), img1[0]))

    img2 = reader.read_at_page_by_name("2", 0, 1)
    logger.info("img2 len: {}, img2[0] len: {}, img2[0]: {}".format(len(img2), len(img2[0]), img2[0]))


def test_mkv_page_reader_random():
    """test mkv page random reader using shard api"""
    reader = ShardSegment()
    reader.open(MKV_FILE_NAME + "0")

    fields = reader.get_category_fields()
    logger.info("fields: {}".format(fields))

    ret = reader.set_category_field("id")
    assert ret == SUCCESS, 'failed on setting category field.'

    names = random.sample(range(1, 6), 5)
    for name in names:
        img2 = reader.read_at_page_by_name(str(name), 0, 2)
        logger.info("name: {}, img2[0] len: {}".format(str(name), len(img2[0])))

    paths = ["{}{}".format(MKV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(FILES_NUM)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))


def test_mkv_file_writer_with_exactly_schema():
    """test mkv file writer using shard api"""
    header = ShardHeader()
    img_schema_json = {"annotation_name": {"type": "array",
                                           "items": {"type": "string"}},
                       "annotation_pose": {"type": "array",
                                           "items": {"type": "string"}},
                       "annotation_truncated": {"type": "array",
                                                "items": {"type": "string"}},
                       "annotation_difficult": {"type": "array",
                                                "items": {"type": "string"}},
                       "annotation_xmin": {"type": "array",
                                           "items": {"type": "number"}},
                       "annotation_ymin": {"type": "array",
                                           "items": {"type": "number"}},
                       "annotation_xmax": {"type": "array",
                                           "items": {"type": "number"}},
                       "annotation_ymax": {"type": "array",
                                           "items": {"type": "number"}},
                       "metadata_width": {"type": "number"},
                       "metadata_height": {"type": "number"},
                       "metadata_depth": {"type": "number"},
                       "img_path": {"type": "string"},
                       "annotation_path": {"type": "string"}}
    img_schema = header.build_schema(img_schema_json, ["data"], "image_schema")
    schema_id = header.add_schema(img_schema)
    assert schema_id == 0, 'failed on building schema.'

    writer = ShardWriter()
    paths = ["{}{}".format(MKV_FILE_NAME, x) for x in range(1)]
    ret = writer.open(paths)
    assert ret == SUCCESS, 'failed on opening files.'
    writer.set_header_size(1 << 24)
    writer.set_page_size(1 << 25)

    image_bytes = bytes("it's a image picutre", encoding="utf8")
    data = []
    data.append({"annotation_name": ["xxxxxxxxxx.jpg"],
                 "annotation_pose": ["hahahahah"],
                 "annotation_truncated": ["1"], "annotation_difficult": ["0"],
                 "annotation_xmin": [100], "annotation_ymin": [200],
                 "annotation_xmax": [300], "annotation_ymax": [400],
                 "metadata_width": 333, "metadata_height": 222,
                 "metadata_depth": 3,
                 "img_path": "/tmp/", "annotation_path": "/tmp/annotation",
                 "data": image_bytes})
    data.append({"annotation_name": ["xxxxxxxxxx.jpg"],
                 "annotation_pose": ["hahahahah"],
                 "annotation_truncated": ["1"], "annotation_difficult": ["0"],
                 "annotation_xmin": [100], "annotation_ymin": [200],
                 "annotation_xmax": [300], "annotation_ymax": [400],
                 "metadata_width": 333, "metadata_height": 222,
                 "metadata_depth": 3,
                 "img_path": "/tmp/", "annotation_path": "/tmp/annotation",
                 "data": image_bytes})
    ret = writer.set_shard_header(header)
    assert ret == SUCCESS, 'failed on setting header.'
    ret = writer.write_raw_cv_data({schema_id: data})
    assert ret == SUCCESS, 'failed on writing raw data.'
    ret = writer.commit()
    assert ret == SUCCESS, 'failed on committing.'

    generator = ShardIndexGenerator(os.path.realpath(paths[0]))
    generator.build()
    generator.write_to_db()


def test_mkv_file_reader_with_exactly_schema():
    """test mkv file reader using shard api"""
    dataset = ShardReader()
    dataset.open(MKV_FILE_NAME + "0")
    dataset.launch()
    index = 0
    _, blob_fields = dataset.get_blob_fields()
    iterator = dataset.get_next()
    while iterator:
        for blob, raw in iterator:
            raw[blob_fields[0]] = bytes(blob)
            logger.info("#item{}: {}".format(index, raw))
            index += 1
            iterator = dataset.get_next()
    dataset.finish()
    dataset.close()

    paths = ["{}{}".format(MKV_FILE_NAME, str(x).rjust(1, '0'))
             for x in range(1)]
    for x in paths:
        os.remove("{}".format(x))
        os.remove("{}.db".format(x))
