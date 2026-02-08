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
"""text transform - ngram"""

import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text


def gen(texts):
    for line in texts:
        yield (np.array(line.split(" "), dtype=str),)


def test_ngram_operation_01():
    """
    Feature: Ngram op
    Description: Test Ngram op with different n values
    Expectation: Output matches expected n-gram results
    """
    # test n=1
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = []
    n_gram_mottos.append(["Friendly", "Manitoba"])
    n_gram_mottos.append(["Yours", "to", "Discover"])
    n_gram_mottos.append(['Land', 'of', 'Living', 'Skies'])
    n_gram_mottos.append(['Birthplace', 'of', 'the', 'Confederation'])
    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(1, separator=" "), input_columns=["text"])
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test n=[2, 3]
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['WildRose Country', ''])
    n_gram_mottos.append(["Canada's Ocean", "Ocean Playground", "Canada's Ocean Playground"])
    n_gram_mottos.append(["Land of", "of Living", "Living Skies", "Land of Living", "of Living Skies"])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram([2, 3], separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test n=[1, 1]
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(["WildRose", "Country", "WildRose", "Country"])
    n_gram_mottos.append(["Canada's", "Ocean", "Playground", "Canada's", "Ocean", "Playground"])
    n_gram_mottos.append(["Land", "of", "Living", "Skies", "Land", "of", "Living", "Skies"])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram([1, 1], separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test n=[2]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = []
    n_gram_mottos.append(["Friendly Manitoba"])
    n_gram_mottos.append(["Yours to", "to Discover"])
    n_gram_mottos.append(['Land of', 'of Living', 'Living Skies'])
    n_gram_mottos.append(['Birthplace of', 'of the', 'the Confederation'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram([2], separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test n=[3, 1, 2]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = []
    n_gram_mottos.append(['', "Friendly", "Manitoba", "Friendly Manitoba"])
    n_gram_mottos.append(["Yours to Discover", "Yours", "to", "Discover", "Yours to", "to Discover"])
    n_gram_mottos.append(
        ['Land of Living', 'of Living Skies', 'Land', 'of', 'Living', 'Skies', 'Land of', 'of Living', 'Living Skies'])
    n_gram_mottos.append(
        ['Birthplace of the', 'of the Confederation', 'Birthplace', 'of', 'the', 'Confederation', 'Birthplace of',
         'of the', 'the Confederation'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram([3, 1, 2], separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test left_pad=("_", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['_ _ WildRose', '_ WildRose Country'])
    n_gram_mottos.append(["_ _ Canada's", "_ Canada's Ocean", "Canada's Ocean Playground"])
    n_gram_mottos.append(["_ _ Land", "_ Land of", "Land of Living", "of Living Skies"])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1


def test_ngram_operation_02():
    """
    Feature: Ngram op
    Description: Test Ngram op with different left_pad parameters
    Expectation: Output matches expected padded n-gram results
    """
    # test left_pad=("__", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['__ __ WildRose', '__ WildRose Country'])
    n_gram_mottos.append(["__ __ Canada's", "__ Canada's Ocean", "Canada's Ocean Playground"])
    n_gram_mottos.append(['__ __ Land', '__ Land of', 'Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("__", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test left_pad=(" ", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['    WildRose', '  WildRose Country'])
    n_gram_mottos.append(["    Canada's", "  Canada's Ocean", "Canada's Ocean Playground"])
    n_gram_mottos.append(['    Land', '  Land of', 'Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=(" ", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test left_pad=("", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['  WildRose', ' WildRose Country'])
    n_gram_mottos.append(["  Canada's", " Canada's Ocean", "Canada's Ocean Playground"])
    n_gram_mottos.append(['  Land', ' Land of', 'Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test left_pad=("_", 1)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['_ WildRose Country'])
    n_gram_mottos.append(["_ Canada's Ocean", "Canada's Ocean Playground"])
    n_gram_mottos.append(['_ Land of', 'Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", 1), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test left_pad=("_", 0)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = [['']]
    n_gram_mottos.append(["Canada's Ocean Playground"])
    n_gram_mottos.append(['Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", 0), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test left_pad=("_", 3)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['_ _ WildRose', '_ WildRose Country'])
    n_gram_mottos.append(["_ _ Canada's", "_ Canada's Ocean", "Canada's Ocean Playground"])
    n_gram_mottos.append(["_ _ Land", "_ Land of", "Land of Living", "of Living Skies"])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", 3), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1


def test_ngram_operation_03():
    """
    Feature: Ngram op
    Description: Test Ngram op with different right_pad parameters
    Expectation: Output matches expected padded n-gram results
    """
    # test right_pad=("_", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['WildRose Country _', 'Country _ _'])
    n_gram_mottos.append(["Canada's Ocean Playground", 'Ocean Playground _', 'Playground _ _'])
    n_gram_mottos.append(["Land of Living", "of Living Skies", 'Living Skies _', 'Skies _ _'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("_", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test right_pad=("__", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['WildRose Country __', 'Country __ __'])
    n_gram_mottos.append(["Canada's Ocean Playground", 'Ocean Playground __', 'Playground __ __'])
    n_gram_mottos.append(["Land of Living", "of Living Skies", 'Living Skies __', 'Skies __ __'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("__", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test right_pad=(" ", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['WildRose Country  ', 'Country    '])
    n_gram_mottos.append(["Canada's Ocean Playground", 'Ocean Playground  ', 'Playground    '])
    n_gram_mottos.append(["Land of Living", "of Living Skies", 'Living Skies  ', 'Skies    '])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=(" ", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test right_pad=("", 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['WildRose Country ', 'Country  '])
    n_gram_mottos.append(["Canada's Ocean Playground", 'Ocean Playground ', 'Playground  '])
    n_gram_mottos.append(["Land of Living", "of Living Skies", 'Living Skies ', 'Skies  '])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("", 2), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test right_pad=("_", 1)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = []
    n_gram_mottos.append(['WildRose Country _'])
    n_gram_mottos.append(["Canada's Ocean Playground", 'Ocean Playground _'])
    n_gram_mottos.append(['Land of Living', 'of Living Skies', 'Living Skies _'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("_", 1), separator=" "), input_columns=["text"])

    i = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        i += 1

    # test right_pad=("_", 0)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = [['']]
    n_gram_mottos.append(["Canada's Ocean Playground"])
    n_gram_mottos.append(['Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("_", 0), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1


def test_ngram_operation_04():
    """
    Feature: Ngram op
    Description: Test Ngram op with different separator parameters
    Expectation: Output matches expected n-gram results with specified separators
    """
    # test right_pad=("", 0)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    n_gram_mottos = [['']]
    n_gram_mottos.append(["Canada's Ocean Playground"])
    n_gram_mottos.append(['Land of Living', 'of Living Skies'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("", 0), separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1

    # test separator='-'
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = [['']]
    n_gram_mottos.append(["Yours-to-Discover"])
    n_gram_mottos.append(['Land-of-Living', 'of-Living-Skies'])
    n_gram_mottos.append(['Birthplace-of-the', 'of-the-Confederation'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(3, separator='-'), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test separator=''
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = [['']]
    n_gram_mottos.append(["YourstoDiscover"])
    n_gram_mottos.append(['LandofLiving', 'ofLivingSkies'])
    n_gram_mottos.append(['Birthplaceofthe', 'oftheConfederation'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(3, separator=''), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test separator='--'
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = [['']]
    n_gram_mottos.append(["Yours--to--Discover"])
    n_gram_mottos.append(['Land--of--Living', 'of--Living--Skies'])
    n_gram_mottos.append(['Birthplace--of--the', 'of--the--Confederation'])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(3, separator='--'), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test separator=" "
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = [['']]
    n_gram_mottos.append([''])
    n_gram_mottos.append(["Land of Living Skies"])
    n_gram_mottos.append(["Birthplace of the Confederation"])

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(4, separator=" "), input_columns=["text"])

    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1

    # test separator=" ",n=3
    data = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    op = text.Ngram(3, separator=" ")
    res = op(data)
    assert res == ["WildRose Country Canada's Ocean Playground Land of Living Skies"]

    # test separator=" ",n=[2,2]
    data = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    op = text.Ngram([2, 2], separator=" ")
    res = op(data)
    assert np.array_equal(res, ["WildRose Country Canada's Ocean Playground",
                                "Canada's Ocean Playground Land of Living Skies",
                                "WildRose Country Canada's Ocean Playground",
                                "Canada's Ocean Playground Land of Living Skies"])

    # test separator="==",n=2
    data = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    op = text.Ngram(2, separator="==")
    res = op(data)
    assert np.array_equal(res, ["WildRose Country==Canada's Ocean Playground",
                                "Canada's Ocean Playground==Land of Living Skies"])


def test_ngram_operation_05():
    """
    Feature: Ngram op
    Description: Test Ngram op with Chinese text
    Expectation: Output matches expected Chinese n-gram results
    """
    # test separator=" ",n=[4,1]
    data = ["西安", "欢迎", "您"]
    op = text.Ngram([4, 2], separator="*")
    res = op(data)
    assert np.array_equal(res, ['', '西安*欢迎', '欢迎*您'])


def test_ngram_exception_01():
    """
    Feature: Ngram op
    Description: Test Ngram op with invalid n parameters
    Expectation: Raise expected exceptions for invalid n values
    """
    # test n=0
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="Input gram_0 must be greater than 0"):
        dataset.map(operations=text.Ngram(0, separator=" "), input_columns=["text"])

    # test n=-1
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="Input gram_0 must be greater than 0"):
        dataset.map(operations=text.Ngram(-1, separator=" "), input_columns=["text"])

    # test n=3.0
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="n needs to be a non-empty list of positive integers"):
        dataset.map(operations=text.Ngram(3.0, separator=" "), input_columns=["text"])

    # test n='sjack &%3e1'
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="n needs to be a non-empty list of positive integers"):
        dataset.map(operations=text.Ngram('sjack &%3e1', separator=" "), input_columns=["text"])

    # test n=[0, 1]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="Input gram_0 must be greater than 0"):
        dataset.map(operations=text.Ngram([0, 1], separator=" "), input_columns=["text"])

    # test n=[-2, 3]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="Input gram_0 must be greater than 0"):
        dataset.map(operations=text.Ngram([-2, 3], separator=" "), input_columns=["text"])

    # test n=[2.0, 3.0]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(TypeError, match="Argument gram\\[0\\] with value 2.0 is not of type"):
        dataset.map(operations=text.Ngram([2.0, 3.0], separator=" "), input_columns=["text"])

    # test n=(2, 3)
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="n needs to be a non-empty list of positive integers"):
        dataset.map(operations=text.Ngram((2, 3), separator=" "), input_columns=["text"])

    # test n=None
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="n needs to be a non-empty list of positive integers"):
        dataset.map(operations=text.Ngram(n=None, separator=" "), input_columns=["text"])

    # test n=[]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="n needs to be a non-empty list of positive integers"):
        dataset.map(operations=text.Ngram(n=[], separator=" "), input_columns=["text"])

    # test n=""
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="n needs to be a non-empty list of positive integers"):
        dataset.map(operations=text.Ngram(n="", separator=" "), input_columns=["text"])

    # test n=[""]
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(TypeError, match='Argument gram\\[0\\] with value "" is not of type'):
        dataset.map(operations=text.Ngram(n=[""], separator=" "), input_columns=["text"])

    # test left_pad=(1, 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=(1, 2), separator=" "), input_columns=["text"])


def test_ngram_exception_02():
    """
    Feature: Ngram op
    Description: Test Ngram op with invalid pad parameters
    Expectation: Raise expected exceptions for invalid padding configurations
    """
    # test left_pad=(None, 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=(None, 2), separator=" "), input_columns=["text"])

    # test left_pad=("_", -1)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="padding width need to be positive numbers"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", -1), separator=" "), input_columns=["text"])

    # test left_pad=("_", 2.5)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", 2.5), separator=" "), input_columns=["text"])

    # test left_pad=("_", None)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=("_", None), separator=" "), input_columns=["text"])

    # test left_pad=["_",2]
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="left_pad needs to be a tuple"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=["_", 2], separator=" "), input_columns=["text"])

    # test left_pad=[]
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="left_pad needs to be a tuple"):
        dataset = dataset.map(operations=text.Ngram(n=3, left_pad=[], separator=" "), input_columns=["text"])

    # test right_pad=(1, 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, right_pad=(1, 2), separator=" "), input_columns=["text"])

    # test right_pad=(None, 2)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, right_pad=(None, 2), separator=" "), input_columns=["text"])

    # test right_pad=("_", -1)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="padding width need to be positive numbers"):
        dataset.map(operations=text.Ngram(n=3, right_pad=("_", -1), separator=" "), input_columns=["text"])

    # test right_pad=("_", 2.5)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("_", 2.5), separator=" "), input_columns=["text"])

    # test right_pad=("_", None)
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="str is pad token and int is pad_width"):
        dataset = dataset.map(operations=text.Ngram(n=3, right_pad=("_", None), separator=" "), input_columns=["text"])

    # test right_pad=["_",2]
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="right_pad needs to be a tuple"):
        dataset = dataset.map(operations=text.Ngram(n=3, right_pad=["_", 2], separator=" "), input_columns=["text"])

    # test right_pad=[]
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(ValueError, match="right_pad needs to be a tuple"):
        dataset = dataset.map(operations=text.Ngram(n=3, right_pad=[], separator=" "), input_columns=["text"])

    # test separator=1
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(TypeError, match="Argument separator with value 1 is not of type"):
        dataset = dataset.map(operations=text.Ngram(n=3, separator=1), input_columns=["text"])


def test_ngram_callable():
    """
    Feature: Ngram op
    Description: Test Ngram op basic usage with valid input
    Expectation: Output is the same as expected output
    """
    op = text.Ngram(2, separator="-")

    input1 = " WildRose Country"
    input1 = np.array(input1.split(" "))
    expect1 = ['-WildRose', 'WildRose-Country']
    result1 = op(input1)
    assert np.array_equal(result1, expect1)

    input2 = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]
    expect2 = ["WildRose Country-Canada's Ocean Playground", "Canada's Ocean Playground-Land of Living Skies"]
    result2 = op(input2)
    assert np.array_equal(result2, expect2)


def test_multiple_ngrams():
    """
    Feature: Ngram op
    Description: Test Ngram op where n is a list of integers
    Expectation: Output is the same as expected output
    """
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

    def gen_data(texts):
        for line in texts:
            yield (np.array(line.split(" ")),)

    dataset = ds.GeneratorDataset(gen_data(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram([1, 2, 3], ("_", 2), ("_", 2), " "), input_columns="text")

    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i]
        i += 1


def test_simple_ngram():
    """
    Feature: Ngram op
    Description: Test Ngram op with only one n value
    Expectation: Output is the same as expected output
    """
    plates_mottos = ["Friendly Manitoba", "Yours to Discover", "Land of Living Skies",
                     "Birthplace of the Confederation"]
    n_gram_mottos = [[""]]
    n_gram_mottos.append(["Yours to Discover"])
    n_gram_mottos.append(['Land of Living', 'of Living Skies'])
    n_gram_mottos.append(['Birthplace of the', 'of the Confederation'])

    def gen_data(texts):
        for line in texts:
            yield (np.array(line.split(" ")),)

    dataset = ds.GeneratorDataset(gen_data(plates_mottos), column_names=["text"])
    dataset = dataset.map(operations=text.Ngram(3, separator=" "), input_columns="text")

    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert list(data["text"]) == n_gram_mottos[i], i
        i += 1


def test_corner_cases():
    """
    Feature: Ngram op
    Description: Test Ngram op with various corner cases and exceptions
    Expectation: Output is the same as expected output or error is raised when appropriate
    """

    def test_config(input_line, n, l_pad=("", 0), r_pad=("", 0), sep=" "):
        def gen_data(texts):
            yield (np.array(texts.split(" ")),)

        try:
            dataset = ds.GeneratorDataset(gen_data(input_line), column_names=["text"])
            dataset = dataset.map(operations=text.Ngram(n, l_pad, r_pad, separator=sep), input_columns=["text"])
            for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
                return list(data["text"])
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
    assert "gram[1] with value [1] is not of type [<class 'int'>]" in test_config("Yours to Discover", [1, [1]])
    assert "n needs to be a non-empty list" in test_config("Yours to Discover", [])
    # test invalid pad
    assert "padding width need to be positive numbers" in test_config("Yours to Discover", [1], ("str", -1))
    assert "pad needs to be a tuple of (str, int)" in test_config("Yours to Discover", [1], ("str", "rts"))
    # test 0 as in valid input
    assert "gram_0 must be greater than 0" in test_config("Yours to Discover", 0)
    assert "gram_0 must be greater than 0" in test_config("Yours to Discover", [0])
    assert "gram_1 must be greater than 0" in test_config("Yours to Discover", [1, 0])


def test_ngram_exception_03():
    """
    Feature: Ngram op
    Description: Test Ngram op with missing or invalid parameters
    Expectation: Raise expected exceptions for missing required parameters
    """
    # test no parameters
    plates_mottos = ["WildRose Country", "Canada's Ocean Playground", "Land of Living Skies"]

    dataset = ds.GeneratorDataset(gen(plates_mottos), column_names=["text"])
    with pytest.raises(TypeError, match="missing a required argument: 'n'"):
        dataset = dataset.map(operations=text.Ngram(), input_columns=["text"])

    # test separator="==",n=2,data = [['hello'], ['world']]
    data = [['hello'], ['world']]
    op = text.Ngram(2, separator="==")
    with pytest.raises(RuntimeError, match='Ngram: input is not a 1D data with string datatype.'):
        _ = op(data)
