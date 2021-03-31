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


def test_random_apply():
    ds.config.set_seed(0)

    def test_config(arr, op_list, prob=0.5):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=ops.RandomApply(op_list, prob), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    res1 = test_config([[0, 1]], [ops.Duplicate(), ops.Concatenate()])
    assert res1 in [[[0, 1]], [[0, 1, 0, 1]]]
    # test single nested compose
    assert test_config([[0, 1, 2]], [ops.Compose([ops.Duplicate(), ops.Concatenate(), ops.Slice([0, 1, 2])])]) == [
        [0, 1, 2]]
    # test exception
    assert "is not of type [<class 'list'>]" in test_config([1, 0], ops.TypeCast(mstype.int32))
    assert "Input prob is not within the required interval" in test_config([0, 1], [ops.Slice([0, 1])], 1.1)
    assert "is not of type [<class 'float'>, <class 'int'>]" in test_config([1, 0], [ops.TypeCast(mstype.int32)], None)
    assert "op_list with value None is not of type [<class 'list'>]" in test_config([1, 0], None)


if __name__ == "__main__":
    test_random_apply()
