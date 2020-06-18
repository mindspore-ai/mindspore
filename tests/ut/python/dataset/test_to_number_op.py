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
import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.text as text

np_integral_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                     np.uint32, np.uint64]
ms_integral_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8,
                     mstype.uint16, mstype.uint32, mstype.uint64]

np_non_integral_types = [np.float16, np.float32, np.float64]
ms_non_integral_types = [mstype.float16, mstype.float32, mstype.float64]

def string_dataset_generator(strings):
    for string in strings:
        yield (np.array(string, dtype='S'),)


def test_to_number_typical_case_integral():
    input_strings = [["-121", "14"], ["-2219", "7623"], ["-8162536", "162371864"],
                     ["-1726483716", "98921728421"]]

    for ms_type, inputs in zip(ms_integral_types, input_strings):
        dataset = ds.GeneratorDataset(string_dataset_generator(inputs), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))

        expected_output = [int(string) for string in inputs]
        output = []
        for data in dataset.create_dict_iterator():
            output.append(data["strings"])

        assert output == expected_output


def test_to_number_typical_case_non_integral():
    input_strings = [["-1.1", "1.4"], ["-2219.321", "7623.453"], ["-816256.234282", "162371864.243243"]]
    epsilons = [0.001, 0.001, 0.0001, 0.0001, 0.0000001, 0.0000001]

    for ms_type, inputs in zip(ms_non_integral_types, input_strings):
        dataset = ds.GeneratorDataset(string_dataset_generator(inputs), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))

        expected_output = [float(string) for string in inputs]
        output = []
        for data in dataset.create_dict_iterator():
            output.append(data["strings"])

        for expected, actual, epsilon in zip(expected_output, output, epsilons):
            assert abs(expected - actual) < epsilon


def out_of_bounds_error_message_check(dataset, np_type, value_to_cast):
    type_info = np.iinfo(np_type)
    type_max = str(type_info.max)
    type_min = str(type_info.min)
    type_name = str(np.dtype(np_type))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "String input " + value_to_cast + " will be out of bounds if casted to " + type_name in str(info.value)
    assert "valid range is: [" + type_min + ", " + type_max + "]" in str(info.value)


def test_to_number_out_of_bounds_integral():
    for np_type, ms_type in zip(np_integral_types, ms_integral_types):
        type_info = np.iinfo(np_type)
        input_strings = [str(type_info.max + 10)]
        dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))
        out_of_bounds_error_message_check(dataset, np_type, input_strings[0])

        input_strings = [str(type_info.min - 10)]
        dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))
        out_of_bounds_error_message_check(dataset, np_type, input_strings[0])


def test_to_number_out_of_bounds_non_integral():
    above_range = [str(np.finfo(np.float16).max * 10), str(np.finfo(np.float32).max * 10), "1.8e+308"]

    input_strings = [above_range[0]]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_non_integral_types[0]))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "outside of valid float16 range" in str(info.value)

    input_strings = [above_range[1]]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_non_integral_types[1]))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "String input " + input_strings[0] + " will be out of bounds if casted to float32" in str(info.value)

    input_strings = [above_range[2]]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_non_integral_types[2]))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "String input " + input_strings[0] + " will be out of bounds if casted to float64" in str(info.value)

    below_range = [str(np.finfo(np.float16).min * 10), str(np.finfo(np.float32).min * 10), "-1.8e+308"]

    input_strings = [below_range[0]]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_non_integral_types[0]))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "outside of valid float16 range" in str(info.value)

    input_strings = [below_range[1]]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_non_integral_types[1]))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "String input " + input_strings[0] + " will be out of bounds if casted to float32" in str(info.value)

    input_strings = [below_range[2]]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_non_integral_types[2]))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "String input " + input_strings[0] + " will be out of bounds if casted to float64" in str(info.value)


def test_to_number_boundaries_integral():
    for np_type, ms_type in zip(np_integral_types, ms_integral_types):
        type_info = np.iinfo(np_type)
        input_strings = [str(type_info.max)]
        dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))
        for data in dataset.create_dict_iterator():
            assert data["strings"] == int(input_strings[0])

        input_strings = [str(type_info.min)]
        dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))
        for data in dataset.create_dict_iterator():
            assert data["strings"] == int(input_strings[0])

        input_strings = [str(0)]
        dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
        dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(ms_type))
        for data in dataset.create_dict_iterator():
            assert data["strings"] == int(input_strings[0])


def test_to_number_invalid_input():
    input_strings = ["a8fa9ds8fa"]
    dataset = ds.GeneratorDataset(string_dataset_generator(input_strings), "strings")
    dataset = dataset.map(input_columns=["strings"], operations=text.ToNumber(mstype.int32))

    with pytest.raises(RuntimeError) as info:
        for _ in dataset.create_dict_iterator():
            pass
    assert "It is invalid to convert " + input_strings[0] + " to a number" in str(info.value)


if __name__ == '__main__':
    test_to_number_typical_case_integral()
    test_to_number_typical_case_non_integral()
    test_to_number_boundaries_integral()
    test_to_number_out_of_bounds_integral()
    test_to_number_out_of_bounds_non_integral()
    test_to_number_invalid_input()
