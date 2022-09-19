# Copyright 2022 Huawei Technologies Co., Ltd
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
# ============================================================================
"""test vmap in graph mode"""
import pytest
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops.functional import vmap
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


class ThreeInputsTwoOutputsNet(nn.Cell):
    def construct(self, x, y, z):
        return x + y, z


def test_lambda_fn():
    """
    Feature: vmap
    Description: The first argument of `vmap` is a lambda function.
    Expectation: throw TypeError:"Parse Lambda Function Fail. Node type must be Lambda, but got Call."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(TypeError) as ex:
        vmap(lambda x, y, z: x + y + z, in_axes=(1, 1, None), out_axes=0)(x_hat, y_hat, z_hat)
    assert "Parse Lambda Function Fail. Node type must be Lambda, but got Call." in str(ex.value)


def test_single_op():
    """
    Feature: vmap
    Description: The first argument of `vmap` is a single primitive.
    Expectation: throw RuntimeError:"'VmapOperation' arg0 Prim: S-Prim-Add cast to 'FuncGraphAbstractClosure' failed."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    with pytest.raises(RuntimeError) as ex:
        vmap(P.Add(), in_axes=(1, 1), out_axes=0)(x_hat, y_hat)
    assert "'VmapOperation' arg0 Prim: S-Prim-Add cast to 'FuncGraphAbstractClosure' failed." in str(ex.value)


def test_none_in_axes():
    """
    Feature: vmap
    Description: The `in_axis` argument of `vmap` is a single None, and it's invalid when apply `vmap`.
    Expectation: throw RuntimeError:"The 'in_axes' of 'vmap' cannot be a single None."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=None, out_axes=0)(x_hat, y_hat, z_hat)
    assert "The 'in_axes' of 'vmap' cannot be a single None while 'fn' is not a 'CellList'." in str(ex.value)


def test_none_out_axes():
    """
    Feature: vmap
    Description: The `out_axes` argument of `vmap` is a nested None, and it's invalid when apply `vmap`.
    Expectation: throw RuntimeError:"The 'out_axes' of 'vmap' cannot be all None, but got
        (None, None, None, (None, None))."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1, 1, None),
             out_axes=(None, None, None, (None, None)))(x_hat, y_hat, z_hat)
    assert "The 'out_axes' of 'vmap' cannot be all None while 'fn' is not a 'CellList', " \
           "but got (None, None, None, (None, None))." in str(ex.value)


def test_mismatch_out_axes():
    """
    Feature: vmap
    Description: The `out_axes` of `vmap` sets to (0, 0, 0), but the outputs of `fn` is x + y, z.
    Expectation: throw RuntimeError:"The size of vmap's 'out_axes' should be equal to the number of results of 'fn': 2,
        but got size: 3."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1, 1, None), out_axes=(0, 0, 0))(x_hat, y_hat, z_hat)
    assert "The size of vmap's 'out_axes' should be equal to the number of results of 'fn': 2, but got size: 3." \
           in str(ex.value)


def test_axis_type():
    """
    Feature: vmap
    Description: The `in_axes` of `vmap` contains elements of Float type.
    Expectation: throw RuntimeError:"The axis in vmap's 'in_axes' should be a None or a scalar of type Int64Imm,
        but got a 1."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1., 1., None), out_axes=0)(x_hat, y_hat, z_hat)
    assert "The axis in vmap's 'in_axes' should be a None or a scalar of type Int64Imm, but got a 1." in str(ex.value)


def test_axis_out_of_bounds():
    """
    Feature: vmap
    Description: The dimension of X is 2, but the corresponding axis -3 is set.
    Expectation: throw RuntimeError:"The axis: -3 in 'in_axes' is out of bounds for array of dimension [-2,2)."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(-3, 2, None), out_axes=0)(x_hat, y_hat, z_hat)
    assert "The axis: -3 in 'in_axes' is out of bounds for array of dimension [-2,2)." in str(ex.value)


def test_mismatch_none_axis():
    """
    Feature: vmap
    Description: The source axis of the first output of `fn` is non-None, but the `out_axes` for that is None,
        it's invalid when apply `vmap`.
    Expectation: throw RuntimeError:"It is invalid that source is not None and dst is None."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1, 1, None), out_axes=(None, 0))(x_hat, y_hat, z_hat)
    assert "It is invalid that source is not None and dst is None." in str(ex.value)


def test_mismatch_parameters_number():
    """
    Feature: vmap
    Description: The arguments of the cell is (x, y, z), but the arguments of vmap-ed function is (x_hat, y_hat).
    Expectation: throw TypeError:"The parameters number of the function is 3, but the number of provided arguments
        is 2."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    with pytest.raises(TypeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1, 1, None), out_axes=0)(x_hat, y_hat)
    assert "The parameters number of the function is 3, but the number of provided arguments is 2." in str(ex.value)


def test_mismatch_axis_size():
    """
    Feature: vmap
    Description: The `axis_size` of X is 3, and the `axis_size` of Y is 2, they are not equal, vmap needs to ensure
        that the `axis_size` of all parameters are uniform.
    Expectation: throw RuntimeError:"The 'axis_size' of each argument in the scope of 'vmap' should be equal,
        but got 3 and 2."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1, 0, None), out_axes=0)(x_hat, y_hat, z_hat)
    assert "The 'axis_size' of each argument in the scope of 'vmap' should be equal, but got 3 and 2." in str(ex.value)


def test_vmap_non_input():
    """
    Feature: vmap
    Description: The arguments of the cell is empty, it's invalid when apply `vmap`.
    Expectation: throw RuntimeError:"Failed to get 'axis_size' within the scope of vmap."
    """
    class NonInputSingleOutputNet(nn.Cell):
        def construct(self):
            return 1

    with pytest.raises(RuntimeError) as ex:
        vmap(NonInputSingleOutputNet())()
    assert "Failed to get 'axis_size' within the scope of vmap." in str(ex.value)


def test_non_fn():
    """
    Feature: vmap
    Description: The first argument of `vmap` not provided, which is required positional argument.
    Expectation: throw TypeError:"vmap() missing 1 required positional argument: 'fn'"
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(TypeError) as ex:
        vmap(in_axes=(1, 1, None), out_axes=0)(x_hat, y_hat, z_hat)
    assert "vmap() missing 1 required positional argument: 'fn'" in str(ex.value)


def test_scalar_with_non_zero_axis():
    """
    Feature: vmap
    Description: The second output of `fn` is a scalar with source axis None, but get a destination axis 1, and it's
        invalid when apply `vmap`.
    Expectation: throw RuntimeError:"The axis: 1 in 'out_axes' is out of bounds for array of dimension [-1,1)."
    """
    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1
    with pytest.raises(RuntimeError) as ex:
        vmap(ThreeInputsTwoOutputsNet(), in_axes=(1, 1, None), out_axes=(0, 1))(x_hat, y_hat, z_hat)
    assert "The axis: 1 in 'out_axes' is out of bounds for array of dimension [-1,1)." in str(ex.value)


class AssignNetWithTwoParams(nn.Cell):
    def __init__(self):
        super(AssignNetWithTwoParams, self).__init__()
        self.assign = P.Assign()
        self.ref_a = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_a')
        self.ref_b = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_b')

    def construct(self, replace_tensor):
        out = self.assign(self.ref_a, replace_tensor)
        out = self.ref_b + out
        return out


class AssignNetWithSingleParam(nn.Cell):
    def __init__(self):
        super(AssignNetWithSingleParam, self).__init__()
        self.assign = P.Assign()
        self.ref_a = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_a')

    def construct(self, replace_tensor):
        out = self.assign(self.ref_a, replace_tensor)
        return out


class AssignNetWithTwoArgus(nn.Cell):
    def __init__(self):
        super(AssignNetWithTwoArgus, self).__init__()
        self.assign = P.Assign()
        self.ref_a = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_a')

    def construct(self, replace_tensor, x):
        out = self.assign(self.ref_a, replace_tensor)
        return out, x


def test_celllist_with_one_model():
    """
    Feature: vmap model ensembling scenario
    Description: The `fn` is a CellList with only one Model.
    Expectation: throw RuntimeError:"In the model ensembling parallel training scenario ('VmapOperation'
        arg0 is a 'CellList'), the size of 'CellList' must be greater than 1, but got 1."
    """
    m1 = AssignNetWithSingleParam()
    mm = nn.CellList([m1])
    replace_tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float32)

    with pytest.raises(RuntimeError) as ex:
        vmap(mm, in_axes=0)(replace_tensor)
    assert "In the model ensembling parallel training scenario ('VmapOperation' arg0 is a 'CellList'), " \
           "the size of 'CellList' must be greater than 1, but got 1." in str(ex.value)


def test_celllist_with_inconsistent_inputs():
    """
    Feature: vmap model ensembling scenario
    Description: The `fn` is a CellList with two Model, but they have different input size.
    Expectation: throw RuntimeError:"'VmapOperation' arg0 is a CellList, whose elements's inputs should be consistent."
    """
    m1 = AssignNetWithSingleParam()
    m2 = AssignNetWithTwoArgus()
    mm = nn.CellList([m1, m2])
    replace_tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float32)

    with pytest.raises(RuntimeError) as ex:
        vmap(mm, in_axes=0)(replace_tensor)
    assert "'VmapOperation' arg0 is a CellList, whose elements's inputs should be consistent." in str(ex.value)


def test_celllist_with_inconsistent_params():
    """
    Feature: vmap model ensembling scenario
    Description: The `fn` is a CellList with two Model, but they have different parameter size.
    Expectation: throw ValueError:"Parameter size of each cell should be consistent, but get 1 and 2."
    """
    m1 = AssignNetWithSingleParam()
    m2 = AssignNetWithTwoParams()
    mm = nn.CellList([m1, m2])
    replace_tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float32)

    with pytest.raises(ValueError) as ex:
        vmap(mm, in_axes=0)(replace_tensor)
    assert "Parameter size of each cell should be consistent, but get 1 and 2." in str(ex.value)
