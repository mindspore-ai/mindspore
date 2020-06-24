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
Testing PythonTokenizer op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore import log as logger

DATA_FILE = "../data/dataset/testTokenizerData/1.txt"


def test_whitespace_tokenizer_ch():
    """
    Test PythonTokenizer
    """
    whitespace_strs = [["Welcome", "to", "Beijing!"],
                       ["北京欢迎您！"],
                       ["我喜欢English!"],
                       [""]]

    def my_tokenizer(line):
        words = line.split()
        if not words:
            return [""]
        return words

    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    tokenizer = text.PythonTokenizer(my_tokenizer)
    dataset = dataset.map(operations=tokenizer, num_parallel_workers=1)
    tokens = []
    for i in dataset.create_dict_iterator():
        s = text.to_str(i['text']).tolist()
        tokens.append(s)
    logger.info("The out tokens is : {}".format(tokens))
    assert whitespace_strs == tokens


if __name__ == '__main__':
    test_whitespace_tokenizer_ch()
