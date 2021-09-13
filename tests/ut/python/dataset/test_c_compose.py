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

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops
import mindspore.dataset.transforms.py_transforms as py_ops


def test_compose():
    ds.config.set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=ops.Compose(op_list), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # test simple compose with only 1 op, this would generate a warning
    assert test_config([[1, 0], [3, 4]], [ops.Fill(2)]) == [[2, 2], [2, 2]]
    # test 1 column -> 2columns -> 1 -> 2 -> 1
    assert test_config([[1, 0]], [ops.Duplicate(), ops.Concatenate(), ops.Duplicate(), ops.Concatenate()]) == [
        [1, 0] * 4]
    # test one python transform followed by a C transform. type after oneHot is float (mixed use-case)
    assert test_config([1, 0], [py_ops.OneHotOp(2), ops.TypeCast(mstype.int32)]) == [[0, 1], [1, 0]]
    # test exceptions. compose, randomApply randomChoice use the same validator
    assert "op_list[0] is neither a c_transform op" in test_config([1, 0], [1, ops.TypeCast(mstype.int32)])
    # test empty op list
    assert "op_list can not be empty." in test_config([1, 0], [])


if __name__ == "__main__":
    test_compose()
