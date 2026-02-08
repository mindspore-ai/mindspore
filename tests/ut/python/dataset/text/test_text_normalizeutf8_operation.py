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
"""
Testing UnicodeCharTokenizer op in DE
"""
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset import text

DATA_FILE = "../data/dataset/testTokenizerData/1.txt"
NORMALIZE_FILE = "../data/dataset/testTokenizerData/normalize.txt"
REGEX_REPLACE_FILE = "../data/dataset/testTokenizerData/regex_replace.txt"
REGEX_TOKENIZER_FILE = "../data/dataset/testTokenizerData/regex_tokenizer.txt"


def test_normalize_utf8():
    """
    Feature: NormalizeUTF8 op
    Description: Test NormalizeUTF8 op basic usage
    Expectation: Output is equal to the expected output
    """

    def normalize(normalize_form):
        dataset = ds.TextFileDataset(NORMALIZE_FILE, shuffle=False)
        normalize = text.NormalizeUTF8(normalize_form=normalize_form)
        dataset = dataset.map(operations=normalize)
        out_bytes = []
        out_texts = []
        for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            out_bytes.append(text.to_bytes(i['text']))
            out_texts.append(i['text'].tolist())
        logger.info("The out bytes is : ", out_bytes)
        logger.info("The out texts is: ", out_texts)
        return out_bytes

    expect_normlize_data = [
        # NFC
        [b'\xe1\xb9\xa9', b'\xe1\xb8\x8d\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'\xef\xac\x81', b'2\xe2\x81\xb5', b'\xe1\xba\x9b\xcc\xa3'],
        # NFKC
        [b'\xe1\xb9\xa9', b'\xe1\xb8\x8d\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'fi', b'25', b'\xe1\xb9\xa9'],
        # NFD
        [b's\xcc\xa3\xcc\x87', b'd\xcc\xa3\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'\xef\xac\x81', b'2\xe2\x81\xb5', b'\xc5\xbf\xcc\xa3\xcc\x87'],
        # NFKD
        [b's\xcc\xa3\xcc\x87', b'd\xcc\xa3\xcc\x87', b'q\xcc\xa3\xcc\x87',
         b'fi', b'25', b's\xcc\xa3\xcc\x87']
    ]
    assert normalize(text.utils.NormalizeForm.NFC) == expect_normlize_data[0]
    assert normalize(text.utils.NormalizeForm.NFKC) == expect_normlize_data[1]
    assert normalize(text.utils.NormalizeForm.NFD) == expect_normlize_data[2]
    assert normalize(text.utils.NormalizeForm.NFKD) == expect_normlize_data[3]


if __name__ == '__main__':
    test_normalize_utf8()
