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
Testing from_dataset in mindspore.dataset
"""
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.text as text


def test_demo_basic_from_dataset():
    """ this is a tutorial on how from_dataset should be used in a normal use case"""
    data = ds.TextFileDataset("../data/dataset/testVocab/words.txt", shuffle=False)
    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None, special_tokens=["<pad>", "<unk>"],
                                    special_first=True)
    data = data.map(input_columns=["text"], operations=text.Lookup(vocab))
    res = []
    for d in data.create_dict_iterator():
        res.append(d["text"].item())
    assert res == [4, 5, 3, 6, 7, 2], res


def test_demo_basic_from_dataset_with_tokenizer():
    """ this is a tutorial on how from_dataset should be used in a normal use case with tokenizer"""
    data = ds.TextFileDataset("../data/dataset/testTokenizerData/1.txt", shuffle=False)
    data = data.map(input_columns=["text"], operations=text.UnicodeCharTokenizer())
    vocab = text.Vocab.from_dataset(data, None, freq_range=None, top_k=None, special_tokens=["<pad>", "<unk>"],
                                    special_first=True)
    data = data.map(input_columns=["text"], operations=text.Lookup(vocab))
    res = []
    for d in data.create_dict_iterator():
        res.append(list(d["text"]))
    assert res == [[13, 3, 7, 14, 9, 17, 3, 2, 19, 9, 2, 11, 3, 4, 16, 4, 8, 6, 5], [21, 20, 10, 25, 23, 26],
                   [24, 22, 10, 12, 8, 6, 7, 4, 18, 15, 5], [2, 2]]


def test_from_dataset():
    """ test build vocab with generator dataset """

    def gen_corpus():
        # key: word, value: number of occurrences, reason for using letters is so their order is apparent
        corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
        for k, v in corpus.items():
            yield (np.array([k] * v, dtype='S'),)

    def test_config(freq_range, top_k):
        corpus_dataset = ds.GeneratorDataset(gen_corpus, column_names=["text"])
        vocab = text.Vocab.from_dataset(corpus_dataset, None, freq_range, top_k, special_tokens=["<pad>", "<unk>"],
                                        special_first=True)
        corpus_dataset = corpus_dataset.map(input_columns="text", operations=text.Lookup(vocab))
        res = []
        for d in corpus_dataset.create_dict_iterator():
            res.append(list(d["text"]))
        return res

    # take words whose frequency is with in [3,4] order them alphabetically for words with the same frequency
    test1_res = test_config(freq_range=(3, 4), top_k=4)
    assert test1_res == [[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1], [5, 5, 5], [1, 1], [1]], str(test1_res)

    # test words with frequency range [2,inf], only the last word will be filtered out
    test2_res = test_config((2, None), None)
    assert test2_res == [[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [6, 6, 6], [5, 5, 5], [7, 7], [1]], str(test2_res)

    # test filter only by top_k
    test3_res = test_config(None, 4)
    assert test3_res == [[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1], [5, 5, 5], [1, 1], [1]], str(test3_res)

    # test filtering out the most frequent
    test4_res = test_config((None, 3), 100)
    assert test4_res == [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [3, 3, 3], [2, 2, 2], [4, 4], [5]], str(test4_res)

    # test top_k == 1
    test5_res = test_config(None, 1)
    assert test5_res == [[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1], [1]], str(test5_res)

    # test min_frequency == max_frequency
    test6_res = test_config((4, 4), None)
    assert test6_res == [[4, 4, 4, 4], [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1], [1]], str(test6_res)


def test_from_dataset_special_token():
    """ test build vocab with generator dataset """

    def gen_corpus():
        # key: word, value: number of occurrences, reason for using letters is so their order is apparent
        corpus = {"D": 1, "C": 1, "B": 1, "A": 1}
        for k, v in corpus.items():
            yield (np.array([k] * v, dtype='S'),)

    def gen_input(texts):
        for word in texts.split(" "):
            yield (np.array(word, dtype='S'),)

    def test_config(texts, top_k, special_tokens, special_first):
        corpus_dataset = ds.GeneratorDataset(gen_corpus, column_names=["text"])
        vocab = text.Vocab.from_dataset(corpus_dataset, None, None, top_k, special_tokens, special_first)
        data = ds.GeneratorDataset(gen_input(texts), column_names=["text"])
        data = data.map(input_columns="text", operations=text.Lookup(vocab))
        res = []
        for d in data.create_dict_iterator():
            res.append(d["text"].item())
        return res

    # test special tokens are inserted before
    assert test_config("A B C D <pad> <unk>", 4, ["<pad>", "<unk>"], True) == [2, 3, 4, 5, 0, 1]
    # test special tokens are inserted after
    assert test_config("A B C D <pad> <unk>", 4, ["<pad>", "<unk>"], False) == [0, 1, 2, 3, 4, 5]


def test_from_dataset_exceptions():
    """ test various exceptions during that are checked in validator """

    def test_config(columns, freq_range, top_k, s):
        try:
            data = ds.TextFileDataset("../data/dataset/testVocab/words.txt", shuffle=False)
            vocab = text.Vocab.from_dataset(data, columns, freq_range, top_k)
            assert isinstance(vocab.text.Vocab)
        except ValueError as e:
            assert s in str(e), str(e)

    test_config("text", (), 1, "freq_range needs to be either None or a tuple of 2 integers")
    test_config("text", (2, 3), 1.2345, "top_k needs to be a positive integer")
    test_config(23, (2, 3), 1.2345, "columns need to be a list of strings")
    test_config("text", (100, 1), 12, "frequency range [a,b] should be 0 <= a <= b")
    test_config("text", (2, 3), 0, "top_k needs to be a positive integer")
    test_config([123], (2, 3), 0, "columns need to be a list of strings")


if __name__ == '__main__':
    test_demo_basic_from_dataset()
    test_from_dataset()
    test_from_dataset_exceptions()
    test_demo_basic_from_dataset_with_tokenizer()
    test_from_dataset_special_token()
