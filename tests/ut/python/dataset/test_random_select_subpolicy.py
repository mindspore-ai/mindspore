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
import mindspore.dataset.vision.c_transforms as visions


def test_random_select_subpolicy():
    ds.config.set_seed(0)

    def test_config(arr, policy):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(operations=visions.RandomSelectSubpolicy(policy), input_columns=["col"])
            res = []
            for i in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # 3 possible outcomes
    policy1 = [[(ops.PadEnd([4], 0), 0.5), (ops.Compose([ops.Duplicate(), ops.Concatenate()]), 1)],
               [(ops.Slice([0, 1]), 0.5), (ops.Duplicate(), 1), (ops.Concatenate(), 1)]]
    res1 = test_config([[1, 2, 3]], policy1)
    assert res1 in [[[1, 2, 1, 2]], [[1, 2, 3, 1, 2, 3]], [[1, 2, 3, 0, 1, 2, 3, 0]]]

    # test exceptions
    assert "policy can not be empty." in test_config([[1, 2, 3]], [])
    assert "policy[0] can not be empty." in test_config([[1, 2, 3]], [[]])
    assert "op of (op, prob) in policy[1][0] is neither a c_transform op (TensorOperation) nor a callable pyfunc" \
           in test_config([[1, 2, 3]], [[(ops.PadEnd([4], 0), 0.5)], [(1, 0.4)]])
    assert "prob of (op, prob) policy[1][0] is not within the required interval of [0, 1]" in test_config([[1]], [
        [(ops.Duplicate(), 0)], [(ops.Duplicate(), -0.1)]])


if __name__ == "__main__":
    test_random_select_subpolicy()
