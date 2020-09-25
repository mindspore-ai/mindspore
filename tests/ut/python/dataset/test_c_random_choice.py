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


import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops


def test_random_choice():
    """
    Test RandomChoice op
    """
    ds.config.set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=ops.RandomChoice(op_list), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # Test whether an operation would be randomly chosen.
    # In order to prevent random failure, both results need to be checked.
    res1 = test_config([[0, 1, 2]], [ops.PadEnd([4], 0), ops.Slice([0, 2])])
    assert res1 in [[[0, 1, 2, 0]], [[0, 2]]]

    # Test nested structure
    res2 = test_config([[0, 1, 2]], [ops.Compose([ops.Duplicate(), ops.Concatenate()]),
                                     ops.Compose([ops.Slice([0, 1]), ops.OneHot(2)])])
    assert res2 in [[[[1, 0], [0, 1]]], [[0, 1, 2, 0, 1, 2]]]
    # Test RandomChoice when there is only 1 operation
    assert test_config([[4, 3], [2, 1]], [ops.Slice([0])]) == [[4], [2]]


if __name__ == "__main__":
    test_random_choice()
