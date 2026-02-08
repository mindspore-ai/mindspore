# Copyright 2025 Huawei Technologies Co., Ltd
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
"""text transform - jiebatokenizer"""

import os
import numpy as np
import platform
import pytest
import mindspore.dataset as ds
from mindspore.dataset.text import JiebaTokenizer
from mindspore.dataset.text import JiebaMode
from mindspore.dataset import text
from mindspore import log as logger


TEST_DATA_DATASET_FUNC ="../data/dataset/"


JIEBATOKENIZER_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                    "testJiebaDataset", "file1.txt")
JIEBATOKENIZER_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                    "testJiebaDataset", "file2.txt")
JIEBATOKENIZER_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                    "testJiebaDataset", "file3.txt")
JIEBATOKENIZER_HMM_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                       "jiebadict", "hmm_model.utf8")
JIEBATOKENIZER_MP_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                      "jiebadict", "jieba.dict.utf8")

DATA_FILE = "../data/dataset/testJiebaDataset/3.txt"
DATA_ALL_FILE = "../data/dataset/testJiebaDataset/*"

HMM_FILE = "../data/dataset/jiebadict/hmm_model.utf8"
MP_FILE = "../data/dataset/jiebadict/jieba.dict.utf8"

DATA_FILE2 = "../data/dataset/testVocab/words.txt"
VOCAB_FILE2 = "../data/dataset/testVocab/vocab_list.txt"
HMM_FILE2 = "../data/dataset/jiebadict/hmm_model.utf8"
MP_FILE2 = "../data/dataset/jiebadict/jieba.dict.utf8"


def test_jiebatokenizer_operation_01():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with different modes and add_word/add_dict functions
    Expectation: Successfully tokenize Chinese text with custom dictionaries
    """
    # Test jieba tokenizer with no mode
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE1)
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE)
    data = data.map(operations=jieba_op,
                    input_columns=["text"], num_parallel_workers=1)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with english
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE2)
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['Hello', ' ', 'welcome', ' ', 'to', ' ', 'the', ' ', 'hotline', ' ', 'of',
              ' ', 'JinTaiLong', ',', ' ', 'we', ' ', 'will', ' ', 'do', ' ', 'our', ' ',
              'best', ' ', 'to', ' ', 'server', ' ', 'you', '!']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with add word,freq is 0
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_word('北京清华大学', freq=0)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with add word,freq is 15
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_word('北京清华大学', freq=150)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with add word,freq is 500090
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_word('北京清华大学', freq=500090)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test add_dict with dict
    user_dict = {
        "北京清华大学": 10
    }
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test add_dict with dict
    user_dict = {
        "北京清华大学": 10,
        "硕士毕业": 10000
    }
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jiebatokenizer_operation_02():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with dict files and with_offsets parameter
    Expectation: Successfully load user dictionaries and provide offset information
    """
    # Test add_dict with empty dict
    user_dict = {}
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京', '清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test add_dict with empty dict
    if platform.system() == "Windows":
        jiebatokenizer_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                           "testJiebaDataset", "user_dict_win.txt")
    else:
        jiebatokenizer_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                           "testJiebaDataset", "user_dict.txt")
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(jiebatokenizer_file)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test with_offsets is True
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE1)
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MP,
                                   with_offsets=True)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"], num_parallel_workers=1)
    data = data.project(columns=["token", "offsets_start", "offsets_limit"])
    expect = ['小', '明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']
    expected_offsets_start = [0, 3, 6, 12, 18, 21, 36, 45, 48, 51, 54, 72]
    expected_offsets_limit = [3, 6, 12, 18, 21, 36, 45, 48, 51, 54, 72, 78]
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # Test jieba tokenizer default value
    data = "我爱我的家乡"
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE)
    data1 = jieba_op(data)

    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX,
                                  with_offsets=False)
    data2 = jieba_op(data)
    assert (data1 == data2).all()

    # Test jieba tokenizer mode is JiebaMode.MP
    data = "我爱我的家乡"
    data1 = ['我', '爱', '我', '的', '家乡']
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MP)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "我爱我的家乡"
    data1 = ['我', '爱', '我', '的', '家', '乡']
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "home 123"
    data1 = ['home', ' ', '123']
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "申猴酉鸡@哈喽"
    data1 = ['申猴', '酉', '鸡', '@', '哈', '喽']
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "今天是元宵节，猜灯谜！"
    data1 = ['今天', '是', '元宵节', '，', '猜灯谜', '！']
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()


def test_jiebatokenizer_operation_03():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with callable and mode
    Expectation: Successfully load user dictionaries and provide offset information
    """

    # one tensor and multiple tensors
    logger.info("test_jieba_callable")
    jieba_op1 = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op2 = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.HMM)

    # test one tensor
    text1 = "今天天气太好了我们一起去外面玩吧"
    text2 = "男默女泪市长江大桥"
    assert np.array_equal(jieba_op1(text1), ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧'])
    assert np.array_equal(jieba_op2(text1), ['今天', '天气', '太', '好', '了', '我们', '一起', '去', '外面', '玩', '吧'])
    jieba_op1.add_word("男默女泪")
    assert np.array_equal(jieba_op1(text2), ['男默女泪', '市', '长江大桥'])

    # test input multiple tensors
    with pytest.raises(RuntimeError) as info:
        _ = jieba_op1(text1, text2)
    assert "JiebaTokenizerOp: input should be one column data." in str(info.value)

    # MP mode
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    ret = []
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # HMM mode
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.HMM)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天', '天气', '太', '好', '了', '我们', '一起', '去', '外面', '玩', '吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # HMM MIX
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MIX)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jiebatokenizer_operation_04():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with add_word
    Expectation: Successfully load user dictionaries and provide offset information
    """

    # add_word op
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_word("男默女泪")
    expect = ['男默女泪', '市', '长江大桥']
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # add_word op with freq
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_word("男默女泪", 10)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['男默女泪', '市', '长江大桥']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # add_word with invalid None input
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    try:
        jieba_op.add_word(None)
    except ValueError:
        pass

    # add_word op with freq where the value of freq affects the result of segmentation
    data_file4 = "../data/dataset/testJiebaDataset/6.txt"
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_word("江大桥", 20000)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['江州', '市长', '江大桥', '参加', '了', '长江大桥', '的', '通车', '仪式']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jiebatokenizer_operation_05():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with add_dict
    Expectation: Successfully load user dictionaries and provide offset information
    """

    # add_dict op with dict
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    user_dict = {
        "男默女泪": 10
    }
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['男默女泪', '市', '长江大桥']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # add_dict op with dict
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    user_dict = {
        "男默女泪": 10,
        "江大桥": 20000
    }
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['男默女泪', '市长', '江大桥']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # add_dict op with valid file path
    data_file4 = "../data/dataset/testJiebaDataset/3.txt"
    dict_file = "../data/dataset/testJiebaDataset/user_dict.txt"

    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_dict(dict_file)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # add_dict op with invalid file path
    dict_file = ""
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    try:
        jieba_op.add_dict(dict_file)
    except ValueError:
        pass

    # add_word op with num_parallel_workers=1
    data_file4 = "../data/dataset/testJiebaDataset/6.txt"

    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    jieba_op.add_word("江大桥", 20000)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['江州', '市长', '江大桥', '参加', '了', '长江大桥', '的', '通车', '仪式']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jiebatokenizer_operation_06():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with MP mode and with_offsets
    Expectation: Successfully load user dictionaries and provide offset information
    """

    # MP mode and with_offsets=True
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    expected_offsets_start = [0, 12, 21, 27, 33, 36, 42]
    expected_offsets_limit = [12, 21, 27, 33, 36, 42, 48]
    ret = []
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # HMM mode and with_offsets=True
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.HMM, with_offsets=True)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['今天', '天气', '太', '好', '了', '我们', '一起', '去', '外面', '玩', '吧']
    expected_offsets_start = [0, 6, 12, 15, 18, 21, 27, 33, 36, 42, 45]
    expected_offsets_limit = [6, 12, 15, 18, 21, 27, 33, 36, 42, 45, 48]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # HMM MIX mode and with_offsets=True
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MIX, with_offsets=True)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    expected_offsets_start = [0, 12, 21, 27, 33, 36, 42]
    expected_offsets_limit = [12, 21, 27, 33, 36, 42, 48]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]


def test_jiebatokenizer_operation_07():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with add_word op with with_offsets
    Expectation: Successfully load user dictionaries and provide offset information
    """

    # add_word op with with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_word("男默女泪")
    expect = ['男默女泪', '市', '长江大桥']
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=2)
    expected_offsets_start = [0, 12, 15]
    expected_offsets_limit = [12, 15, 27]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # freq and with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_word("男默女泪", 10)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=2)
    expect = ['男默女泪', '市', '长江大桥']
    expected_offsets_start = [0, 12, 15]
    expected_offsets_limit = [12, 15, 27]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # freq where freq affects the result of segmentation and with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/6.txt"
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_word("江大桥", 20000)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=2)
    expect = ['江州', '市长', '江大桥', '参加', '了', '长江大桥', '的', '通车', '仪式']
    expected_offsets_start = [0, 6, 12, 21, 27, 30, 42, 45, 51]
    expected_offsets_limit = [6, 12, 21, 27, 30, 42, 45, 51, 57]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]


def test_jiebatokenizer_operation_08():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with add_dict op and with_offsets
    Expectation: Successfully load user dictionaries and provide offset information
    """

    # add_dict op with dict and with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    user_dict = {
        "男默女泪": 10
    }
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['男默女泪', '市', '长江大桥']
    expected_offsets_start = [0, 12, 15]
    expected_offsets_limit = [12, 15, 27]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # add_dict op with dict and with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/4.txt"
    user_dict = {
        "男默女泪": 10,
        "江大桥": 20000
    }
    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['男默女泪', '市长', '江大桥']
    expected_offsets_start = [0, 12, 18]
    expected_offsets_limit = [12, 18, 27]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # add_dict with valid file path and with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/3.txt"
    dict_file = "../data/dataset/testJiebaDataset/user_dict.txt"

    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_dict(dict_file)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    expected_offsets_start = [0, 12, 21, 27, 33, 36, 42]
    expected_offsets_limit = [12, 21, 27, 33, 36, 42, 48]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # add_word op with valid input and with_offsets=True
    data_file4 = "../data/dataset/testJiebaDataset/6.txt"

    data = ds.TextFileDataset(data_file4)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    jieba_op.add_word("江大桥", 20000)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"],
                    num_parallel_workers=1)
    expect = ['江州', '市长', '江大桥', '参加', '了', '长江大桥', '的', '通车', '仪式']
    expected_offsets_start = [0, 6, 12, 21, 27, 30, 42, 45, 51]
    expected_offsets_limit = [6, 12, 21, 27, 30, 42, 45, 51, 57]
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]


def gen():
    text_data = np.array("今天天气太好了我们一起去外面玩吧", dtype=np.str_)
    yield (text_data,)


def pytoken_op(input_data):
    te = input_data.item()
    tokens = [te[:5], te[5:10], te[10:]]
    return np.array(tokens, dtype=np.str_)


def test_jiebatokenizer_operation_09():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with PyToken
    Expectation: Successfully load user dictionaries and provide offset information
    """

    data = ds.GeneratorDataset(gen, column_names=["text"])
    data = data.map(operations=pytoken_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天天气太', '好了我们一', '起去外面玩吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_on_tokenized_line():
    """
    Feature: Python text.Vocab class
    Description: Test Lookup op on tokenized line using JiebaTokenizer with special_tokens
    Expectation: Output is equal to the expected output
    """
    data = ds.TextFileDataset("../data/dataset/testVocab/lines.txt", shuffle=False)
    jieba_op = text.JiebaTokenizer(HMM_FILE2, MP_FILE2, mode=text.JiebaMode.MP)
    with open(VOCAB_FILE2, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split(',')[0]
            jieba_op.add_word(word)
    data = data.map(operations=jieba_op, input_columns=["text"])
    vocab = text.Vocab.from_file(VOCAB_FILE2, ",", special_tokens=["<pad>", "<unk>"])
    lookup = text.Lookup(vocab, "<unk>")
    data = data.map(operations=lookup, input_columns=["text"])
    res = np.array([[10, 1, 11, 1, 12, 1, 15, 1, 13, 1, 14],
                    [11, 1, 12, 1, 10, 1, 14, 1, 13, 1, 15]], dtype=np.int32)
    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["text"], res[i])


    # JiebaTokenizer without special_tokens
    data = ds.TextFileDataset("../data/dataset/testVocab/lines.txt", shuffle=False)
    jieba_op = text.JiebaTokenizer(HMM_FILE2, MP_FILE2, mode=text.JiebaMode.MP)
    with open(VOCAB_FILE2, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split(',')[0]
            jieba_op.add_word(word)

    data = data.map(operations=jieba_op, input_columns=["text"])
    vocab = text.Vocab.from_file(VOCAB_FILE2, ",")
    lookup = text.Lookup(vocab, "not")
    data = data.map(operations=lookup, input_columns=["text"])
    res = np.array([[8, 0, 9, 0, 10, 0, 13, 0, 11, 0, 12],
                    [9, 0, 10, 0, 8, 0, 12, 0, 11, 0, 13]], dtype=np.int32)
    for i, d in enumerate(data.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["text"], res[i])


def test_jiebatokenizer_exception_01():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with invalid parameters and input types
    Expectation: Raise expected exceptions for invalid inputs
    """
    # Test jieba tokenizer with error mode
    with pytest.raises(TypeError, match='Wrong input type for mode, should be JiebaMode'):
        text.JiebaTokenizer(hmm_path=JIEBATOKENIZER_HMM_FILE, mp_path=JIEBATOKENIZER_MP_FILE, mode='HIM')

    # Test jieba tokenizer with no mp_path
    with pytest.raises(TypeError, match="missing a required argument: 'mp_path'"):
        text.JiebaTokenizer(hmm_path=JIEBATOKENIZER_HMM_FILE, mode=JiebaMode.MIX)

    # Test jieba tokenizer with no hmm_path
    with pytest.raises(TypeError, match="missing a required argument: 'hmm_path'"):
        text.JiebaTokenizer(mp_path=JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)

    # Test jieba tokenizer with no hmm_path and mp_path
    with pytest.raises(TypeError, match="missing a required argument: 'hmm_path'"):
        text.JiebaTokenizer(mode=JiebaMode.MIX)

    # Test jieba tokenizer with add word,freq is -1
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    with pytest.raises(ValueError, match=r"Input is not within the required interval of \[0, 4294967295\]"):
        jieba_op.add_word('北京清华大学', freq=-1)

    # Test jieba tokenizer with add word,freq is string
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    with pytest.raises(TypeError, match="Argument  with value test is not of type"):
        jieba_op.add_word('北京清华大学', freq='test')

    # Test add_dict with space
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    with pytest.raises(ValueError, match="user dict file      is not exist"):
        jieba_op.add_dict("    ")

    # Test jieba tokenizer mode is JiebaMode.HMM, data is 1234
    data = 1234
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    with pytest.raises(RuntimeError, match="JiebaTokenizerOp: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = jieba_op(data)

    # Test jieba tokenizer mode is JiebaMode.HMM, data is ["hhh", "哈喽", "world"]
    data = ["hhh", "哈喽", "world"]
    jieba_op = text.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    with pytest.raises(RuntimeError, match="JiebaTokenizerOp: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = jieba_op(data)
