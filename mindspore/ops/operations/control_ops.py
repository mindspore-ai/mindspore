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
# ============================================================================

"""control_ops"""
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register
from ..._checkparam import Rel
from ..._checkparam import Validator as validator
from ...common import dtype as mstype
from ...common._decorator import deprecated


class ControlDepend(Primitive):
    """
    Adds control dependency relation between source and destination operations.

    In many cases, we need to control the execution order of operations. ControlDepend is designed for this.
    ControlDepend will instruct the execution engine to run the operations in a specific order. ControlDepend
    tells the engine that the destination operations must depend on the source operation which means the source
    operations must be executed before the destination.

    Note:
        This operation does not work in `PYNATIVE_MODE`.
        `ControlDepend` is deprecated from version 1.1 and will be removed in a future version, use `Depend` instead.
    Args:
        depend_mode (int): Use 0 for a normal dependency relation and 1 for a user-defined dependency relation.
            Default: 0.

    Inputs:
        - **src** (Any) - The source input. It can be a tuple of operations output or a single operation output. We do
          not concern about the input data, but concern about the operation that generates the input data.
          If `depend_mode` is 1 and the source input is Parameter, we will try to find the operations that
          used the parameter as input.
        - **dst** (Any) - The destination input. It can be a tuple of operations output or a single operation output.
          We do not concern about the input data, but concern about the operation that generates the input data.
          If `depend_mode` is 1 and the source input is Parameter, we will try to find the operations that
          used the parameter as input.

    Outputs:
        This operation has no actual data output, it will be used to setup the order of relative operations.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.control_depend = P.ControlDepend()
        ...         self.softmax = ops.Softmax()
        ...
        ...     def construct(self, x, y):
        ...         mul = x * y
        ...         softmax = self.softmax(x)
        ...         ret = self.control_depend(mul, softmax)
        ...         return ret
        ...
        >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
        >>> print(output)
        [[1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1.]]
    """
    @deprecated("1.1", "Depend")
    @prim_attr_register
    def __init__(self, depend_mode=0):
        """init"""
        validator.check_int_range(depend_mode, 0, 1, Rel.INC_BOTH, "depend_mode", self.name)

    def __call__(self, src, dst):
        return src


class GeSwitch(PrimitiveWithInfer):
    """
    Adds control switch to data.

    Switch data flows into false or true branch depending on the condition. If the condition is true,
    the true branch will be activated, or vise verse.

    Inputs:
        - **data** (Union[Tensor, Number]) - The data to be used for switch control.
        - **pred** (Tensor) - It must be a scalar whose type is bool and shape is `()`, It is used as condition for
          switch control.
    Outputs:
        tuple. Output is tuple(false_output, true_output). The Elements in the tuple has the same shape of input data.
        The false_output connects with the false_branch and the true_output connects with the true_branch.

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.square = ops.Square()
        ...         self.add = ops.Add()
        ...         self.value = Tensor(np.full((1), 3), mindspore.float32)
        ...         self.switch = ops.GeSwitch()
        ...         self.merge = ops.Merge()
        ...         self.less = ops.Less()
        ...
        ...     def construct(self, x, y):
        ...         cond = self.less(x, y)
        ...         st1, sf1 = self.switch(x, cond)
        ...         st2, sf2 = self.switch(y, cond)
        ...         add_ret = self.add(st1, st2)
        ...         st3, sf3 = self.switch(self.value, cond)
        ...         sq_ret = self.square(sf3)
        ...         ret = self.merge((add_ret, sq_ret))
        ...         return ret[0]
        ...
        >>> x = Tensor(10.0, dtype=mindspore.float32)
        >>> y = Tensor(5.0, dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
        >>> print(output)
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, data, pred):
        raise NotImplementedError

    def infer_shape(self, data, pred):
        validator.check_equal_int(len(pred), 0, "pred rank", self.name)
        return (data, data)

    def infer_dtype(self, data_type, pred_type):
        validator.check_subclass(
            "data", data_type, (mstype.tensor,) + mstype.number_type, self.name)
        validator.check_tensor_dtype_valid("pred", pred_type, [mstype.bool_], self.name)
        return (data_type, data_type)


class Merge(PrimitiveWithInfer):
    """
    Merges all input data to one.

    One and only one of the inputs must be selected as the output

    Inputs:
        - **inputs** (Union(Tuple, List)) - The data to be merged. All tuple elements must have the same data type.

    Outputs:
        tuple. Output is tuple(`data`, `output_index`). The `data` has the same shape of `inputs` element.

    Examples:
        >>> merge = ops.Merge()
        >>> input_x = Tensor(np.linspace(0, 8, 8).reshape(2, 4), mindspore.float32)
        >>> input_y = Tensor(np.random.randint(-4, 4, (2, 4)), mindspore.float32)
        >>> result = merge((input_x, input_y))
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, *args):
        raise NotImplementedError

    def infer_shape(self, inputs):
        return (inputs[0], [1])

    def infer_dtype(self, inputs):
        args = {}
        for i, item in enumerate(inputs):
            args['inputs[%d]' % i] = item

        validator.check_scalar_or_tensor_types_same(args, (mstype.bool_,) + mstype.number_type, self.name)
        return (inputs[0], mstype.int32)
