# Copyright 2019 Huawei Technologies Co., Ltd
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

from mindspore import Tensor
from mindspore.parallel._tensor import _reshape_param_data


def test_reshape_param_data():
    expected_tensor = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    dev_mat = [2, 2]
    tensor_map = [0, 1]
    input_tensor = Tensor([[1, 2], [5, 6], [3, 4], [7, 8]])
    tensor = _reshape_param_data(input_tensor, dev_mat, tensor_map)
    if expected_tensor.__str__() != tensor.__str__():
        raise AssertionError

    tensor_map = [1, -1]
    input_tensor = Tensor([[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]])
    tensor = _reshape_param_data(input_tensor, dev_mat, tensor_map)
    if expected_tensor.__str__() != tensor.__str__():
        raise AssertionError

    expected_tensor = Tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                              [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

    input_tensor = Tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], \
                           [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

    dev_mat = [4]
    tensor_map = [-1, -1, -1, -1]
    tensor = _reshape_param_data(input_tensor, dev_mat, tensor_map)
    if expected_tensor.__str__() != tensor.__str__():
        raise AssertionError


if __name__ == '__main__':
    test_reshape_param_data()
