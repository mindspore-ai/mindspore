# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset.text import JiebaTokenizer
from mindspore.dataset.text import JiebaMode
from mindspore import log as logger

DATA_FILE = "../data/dataset/testJiebaDataset/3.txt"
DATA_ALL_FILE = "../data/dataset/testJiebaDataset/*"

HMM_FILE = "../data/dataset/jiebadict/hmm_model.utf8"
MP_FILE = "../data/dataset/jiebadict/jieba.dict.utf8"


def test_jieba_callable():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with one tensor and multiple tensors
    Expectation: Output is equal to the expected output for one tensor and error is raised for multiple tensors
    """
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


def test_jieba_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with MP mode
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_1_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with HMM mode
    Expectation: Output is equal to the expected output
    """
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.HMM)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天', '天气', '太', '好', '了', '我们', '一起', '去', '外面', '玩', '吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jieba_1_2():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with HMM MIX
    Expectation: Output is equal to the expected output
    """
    data = ds.TextFileDataset(DATA_FILE)
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MIX)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天天气', '太好了', '我们', '一起', '去', '外面', '玩吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jieba_2():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_2_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op with freq
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_2_2():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word with invalid None input
    Expectation: Error is raised as expected
    """
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    try:
        jieba_op.add_word(None)
    except ValueError:
        pass


def test_jieba_2_3():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op with freq where the value of freq affects the result of segmentation
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_3():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict op with dict
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_3_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict op with dict
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_4():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict op with valid file path
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_4_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict op with invalid file path
    Expectation: Error is raised as expected
    """
    dict_file = ""
    jieba_op = JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)
    try:
        jieba_op.add_dict(dict_file)
    except ValueError:
        pass


def test_jieba_5():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op with num_parallel_workers=1
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer with MP mode and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_1_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer with HMM mode and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_1_2():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer with HMM MIX mode and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_2():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op with with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_2_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op with freq and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_2_2():
    """
    Feature: JiebaTokenizer op
    Description: Test add_word op with freq where freq affects the result of segmentation and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_3():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict op with dict and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_3_1():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict op with dict and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_4():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_dict with valid file path and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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


def test_jieba_with_offsets_5():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer add_word op with valid input and with_offsets=True
    Expectation: Output is equal to the expected output
    """
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
    text = np.array("今天天气太好了我们一起去外面玩吧", dtype=np.str_)
    yield (text,)


def pytoken_op(input_data):
    te = input_data.item()
    tokens = [te[:5], te[5:10], te[10:]]
    return np.array(tokens, dtype=np.str_)


def test_jieba_6():
    """
    Feature: Pytoken_op
    Description: Test pytoken_op on GeneratorDataset
    Expectation: Output is equal to the expected output
    """
    data = ds.GeneratorDataset(gen, column_names=["text"])
    data = data.map(operations=pytoken_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['今天天气太', '好了我们一', '起去外面玩吧']
    for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


if __name__ == "__main__":
    test_jieba_callable()
    test_jieba_1()
    test_jieba_1_1()
    test_jieba_1_2()
    test_jieba_2()
    test_jieba_2_1()
    test_jieba_2_2()
    test_jieba_3()
    test_jieba_3_1()
    test_jieba_4()
    test_jieba_4_1()
    test_jieba_5()
    test_jieba_5()
    test_jieba_6()
    test_jieba_with_offsets_1()
    test_jieba_with_offsets_1_1()
    test_jieba_with_offsets_1_2()
    test_jieba_with_offsets_2()
    test_jieba_with_offsets_2_1()
    test_jieba_with_offsets_2_2()
    test_jieba_with_offsets_3()
    test_jieba_with_offsets_3_1()
    test_jieba_with_offsets_4()
    test_jieba_with_offsets_5()
