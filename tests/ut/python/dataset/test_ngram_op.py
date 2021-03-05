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
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.text as text


def test_ngram_callable():
    """
    Test ngram op is callable
    """
    op = text.Ngram(2, separator="-")

    input1 = " WildRose Country"
    input1 = np.array(input1.split(" "), dtype='S')
    expect1 = ['-WildRose', 'WildRose-Country']
    result1 = op(input1)
    assert np.array_equal(result1, expect1)

    input2 = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    expect2 = ["WildRose Country-Canada's Ocean Playground", "Canada's Ocean Playground-Land of Living Skies"]
    result2 = op(input2)
    assert np.array_equal(result2, expect2)


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
    dataset = dataset.map(operations=text.Ngram([1, 2, 3], ("_", 2), ("_", 2), " "), input_columns="text")

    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
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
    dataset = dataset.map(operations=text.Ngram(3, separator=" "), input_columns="text")

    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert [d.decode("utf8") for d in data["text"]] == n_gram_mottos[i], i
        i += 1


def test_corner_cases():
    """ testing various corner cases and exceptions"""

    def test_config(input_line, n, l_pad=("", 0), r_pad=("", 0), sep=" "):
        def gen(texts):
            yield (np.array(texts.split(" "), dtype='S'),)

        try:
            dataset = ds.GeneratorDataset(gen(input_line), column_names=["text"])
            dataset = dataset.map(operations=text.Ngram(n, l_pad, r_pad, separator=sep), input_columns=["text"])
            for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
                return [d.decode("utf8") for d in data["text"]]
        except (ValueError, TypeError) as e:
            return str(e)

    # test tensor length smaller than n
    assert test_config("Lone Star", [2, 3, 4, 5]) == ["Lone Star", "", "", ""]
    # test empty separator
    assert test_config("Beautiful British Columbia", 2, sep="") == ['BeautifulBritish', 'BritishColumbia']
    # test separator with longer length
    assert test_config("Beautiful British Columbia", 3, sep="^-^") == ['Beautiful^-^British^-^Columbia']
    # test left pad != right pad
    assert test_config("Lone Star", 4, ("The", 1), ("State", 1)) == ['The Lone Star State']
    # test invalid n
    assert "gram[1] with value [1] is not of type (<class 'int'>,)" in test_config("Yours to Discover", [1, [1]])
    assert "n needs to be a non-empty list" in test_config("Yours to Discover", [])
    # test invalid pad
    assert "padding width need to be positive numbers" in test_config("Yours to Discover", [1], ("str", -1))
    assert "pad needs to be a tuple of (str, int)" in test_config("Yours to Discover", [1], ("str", "rts"))
    # test 0 as in valid input
    assert "gram_0 must be greater than 0" in test_config("Yours to Discover", 0)
    assert "gram_0 must be greater than 0" in test_config("Yours to Discover", [0])
    assert "gram_1 must be greater than 0" in test_config("Yours to Discover", [1, 0])


if __name__ == '__main__':
    test_ngram_callable()
    test_multiple_ngrams()
    test_simple_ngram()
    test_corner_cases()
