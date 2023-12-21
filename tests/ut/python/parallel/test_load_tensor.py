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

from hccl_test.manage.api import Hccl

from mindspore import Tensor
from mindspore.parallel._tensor import _load_tensor


def test_load_tensor():
    hccl = Hccl()
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])
    dev_mat = [2, 3]
    tensor_map = [1, -1]
    hccl.rank_id = 5
    tensor_slice = _load_tensor(tensor, dev_mat, tensor_map, [2, 3])
    expected_tensor = Tensor([[4, 5, 6]])
    if expected_tensor.__str__() != tensor_slice.__str__():
        raise AssertionError

    hccl.rank_id = 2
    tensor_slice = _load_tensor(tensor, dev_mat, tensor_map, [2, 3])
    expected_tensor = Tensor([[1, 2, 3]])
    if expected_tensor.__str__() != tensor_slice.__str__():
        raise AssertionError

    # set back to the default value
    hccl.rank_id = 0


if __name__ == '__main__':
    test_load_tensor()
