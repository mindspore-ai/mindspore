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

import pytest
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as ops
import mindspore.dataset.transforms.py_transforms as py_ops


def test_compose():
    """
    Test C++ and Python Compose Op
    """
    ds.config.set_seed(0)

    def test_config(arr, op_list):
        try:
            data = ds.NumpySlicesDataset(arr, column_names="col", shuffle=False)
            data = data.map(input_columns=["col"], operations=op_list)
            res = []
            for i in data.create_dict_iterator(output_numpy=True):
                res.append(i["col"].tolist())
            return res
        except (TypeError, ValueError) as e:
            return str(e)

    # Test simple compose with only 1 op, this would generate a warning
    assert test_config([[1, 0], [3, 4]], ops.Compose([ops.Fill(2)])) == [[2, 2], [2, 2]]
    # Test 1 column -> 2 columns -> 1 -> 2 -> 1
    assert test_config([[1, 0]],
                       ops.Compose([ops.Duplicate(), ops.Concatenate(), ops.Duplicate(), ops.Concatenate()])) \
           == [[1, 0] * 4]
    # Test one Python transform followed by a C transform. Type after OneHot is a float (mixed use-case)
    assert test_config([1, 0], ops.Compose([py_ops.OneHotOp(2), ops.TypeCast(mstype.int32)])) == [[[0, 1]], [[1, 0]]]
    # Test exceptions.
    with pytest.raises(TypeError) as error_info:
        ops.Compose([1, ops.TypeCast(mstype.int32)])
    assert "op_list[0] is not a c_transform op (TensorOp) nor a callable pyfunc." in str(error_info.value)
    # Test empty op list
    with pytest.raises(ValueError) as error_info:
        test_config([1, 0], ops.Compose([]))
    assert "op_list can not be empty." in str(error_info.value)

    # Test Python compose op
    assert test_config([1, 0], py_ops.Compose([py_ops.OneHotOp(2)])) == [[[0, 1]], [[1, 0]]]
    assert test_config([1, 0], py_ops.Compose([py_ops.OneHotOp(2), (lambda x: x + x)])) == [[[0, 2]], [[2, 0]]]
    # Test nested Python compose op
    assert test_config([1, 0],
                       py_ops.Compose([py_ops.Compose([py_ops.OneHotOp(2)]), (lambda x: x + x)])) \
           == [[[0, 2]], [[2, 0]]]

    with pytest.raises(TypeError) as error_info:
        py_ops.Compose([(lambda x: x + x)])()
    assert "Compose was called without an image. Fix invocation (avoid it being invoked as Compose([...])())." in str(
        error_info.value)


if __name__ == "__main__":
    test_compose()
