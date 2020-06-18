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
Testing Ngram in mindspore.dataset
"""
import mindspore.dataset as ds
import mindspore.dataset.text as text
import numpy as np


def test_multiple_ngrams():
    """ test n-gram where n is a list of integers"""
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(
        ['WildRose', 'Country', '_ WildRose', 'WildRose Country', 'Country _', '_ _ WildRose', '_ WildRose Country',
         'WildRose Country _', 'Country _ _'])
    n_gram_mottos.append(
        ["Canada's", 'Ocean', 'Playground', "_ Canada's", "Canada's Ocean", 'Ocean Playground', 'Playground _',
         "_ _ Canada's", "_ Canada's Ocean", "Canada's Ocean Playground", 'Ocean Playground _', 'Playground _ _'])
    n_gram_mottos.append(
        ['Land', 'of', 'Living', 'Skies', '_ Land', 'Land of', 'of Living', 'Living Skies', 'Skies _', '_ _ Land',
         '_ Land of', 'Land of Living', 'of Living Skies', 'Living Skies _', 'Skies _ _'])

    def gen(texts):
        for line in texts:
            yield (np.array(line.split(" "), dtype='S'),)

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(input_columns=["text"], operations=text.Ngram([1, 2, 3], ("_", 2), ("_", 2), " "))

    i = 0
    for data in dataset.create_dict_iterator():
        assert [d.decode("utf8") for d in data["text"]] == n_gram_mottos[i]
        i += 1


def test_simple_ngram():
    """ test simple gram with only one n value"""
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = [[""]]
    n_gram_mottos.append(["Yours to Discover"])
    n_gram_mottos.append(['Land of Living', 'of Living Skies'])
    n_gram_mottos.append(['Birthplace of the', 'of the Confederation'])

    def gen(texts):
        for line in texts:
            yield (np.array(line.split(" "), dtype='S'),)

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(input_columns=["text"], operations=text.Ngram(3, separator=None))

    i = 0
    for data in dataset.create_dict_iterator():
        assert [d.decode("utf8") for d in data["text"]] == n_gram_mottos[i], i
        i += 1


def test_corner_cases():
    """ testing various corner cases and exceptions"""

    def test_config(input_line, output_line, n, l_pad=None, r_pad=None, sep=None):
        def gen(texts):
            yield (np.array(texts.split(" "), dtype='S'),)

        dataset = ds.GeneratorDataset(gen(input_line), column_names=["text"])
        dataset = dataset.map(input_columns=["text"], operations=text.Ngram(n, l_pad, r_pad, separator=sep))
        for data in dataset.create_dict_iterator():
            assert [d.decode("utf8") for d in data["text"]] == output_line, output_line

    # test tensor length smaller than n
    test_config("Lone Star", ["Lone Star", "", "", ""], [2, 3, 4, 5])
    # test empty separator
    test_config("Beautiful British Columbia", ['BeautifulBritish', 'BritishColumbia'], 2, sep="")
    # test separator with longer length
    test_config("Beautiful British Columbia", ['Beautiful^-^British^-^Columbia'], 3, sep="^-^")
    # test left pad != right pad
    test_config("Lone Star", ['The Lone Star State'], 4, ("The", 1), ("State", 1))
    # test invalid n
    try:
        test_config("Yours to Discover", "", [0, [1]])
    except Exception as e:
        assert "ngram needs to be a positive number" in str(e)
    # test empty n
    try:
        test_config("Yours to Discover", "", [])
    except Exception as e:
        assert "n needs to be a non-empty list" in str(e)
    # test invalid pad
    try:
        test_config("Yours to Discover", "", [1], ("str", -1))
    except Exception as e:
        assert "padding width need to be positive numbers" in str(e)
    # test invalid pad
    try:
        test_config("Yours to Discover", "", [1], ("str", "rts"))
    except Exception as e:
        assert "pad needs to be a tuple of (str, int)" in str(e)


if __name__ == '__main__':
    test_multiple_ngrams()
    test_simple_ngram()
    test_corner_cases()
