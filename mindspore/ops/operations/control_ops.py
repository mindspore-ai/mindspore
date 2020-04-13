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

from ...common import dtype as mstype
from ..._checkparam import ParamValidator as validator
from ..primitive import Primitive, PrimitiveWithInfer, prim_attr_register


class ControlDepend(Primitive):
    """
    Adds control dependency relation between source and destination operation.

    In many cases, we need to control the execution order of operations. ControlDepend is designed for this.
    ControlDepend will indicate the execution engine to run the operations in specific order. ControlDepend
    tells the engine that the destination operations should depend on the source operation which means the source
    operations should be executed before the destination.

    Args:
        depend_mode (int): Use 0 for normal depend, 1 for depend on operations that used the parameter. Default: 0.

    Inputs:
        - **src** (Any) - The source input. It can be a tuple of operations output or a single operation output. We do
          not concern about the input data, but concern about the operation that generates the input data.
          If `depend_mode = 1` is specified and the source input is parameter, we will try to find the operations that
          used the parameter as input.
        - **dst** (Any) - The destination input. It can be a tuple of operations output or a single operation output.
          We do not concern about the input data, but concern about the operation that generates the input data.
          If `depend_mode = 1` is specified and the source input is parameter, we will try to find the operations that
          used the parameter as input.

    Outputs:
        Bool. This operation has no actual data output, it will be used to setup the order of relative operations.

    Examples:
        >>> # In the following example, the data calculation uses original global_step. After the calculation the global
        >>> # step should be increased, so the add operation should depend on the data calculation operation.
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>        super(Net, self).__init__()
        >>>         self.global_step = mindspore.Parameter(initializer(0, [1]), name="global_step")
        >>>         self.rate = 0.2
        >>>         self.control_depend = P.ControlDepend()
        >>>
        >>>     def construct(self, x):
        >>>         data = self.rate * self.global_step + x
        >>>         added_global_step = self.global_step + 1
        >>>         self.global_step = added_global_step
        >>>         self.control_depend(data, added_global_step)
        >>>         return data
    """

    @prim_attr_register
    def __init__(self, depend_mode=0):
        """init"""

    def __call__(self, src, dst):
        return src


class GeSwitch(PrimitiveWithInfer):
    """
    Adds control switch to data.

    Switch data to flow into false or true branch depend on the condition. If the condition is true,
    the true branch will be activated, or vise verse.

    Inputs:
        - **data** (Tensor) - The data to be used for switch control.
        - **pred** (Tensor) - It should be a scalar whose type is bool and shape is `()`, It is used as condition for
          switch control.
    Outputs:
        tuple. Output is tuple(false_output, true_output). The Elements in the tuple has the same shape of input data.
        The false_output connects with the false_branch and the true_output connects with the true_branch.

    Examples:
        >>> class Net(nn.Cell):
        >>> 	def __init__(self):
        >>>	    super(Net, self).__init__()
        >>>         self.square = P.Square()
        >>>         self.add = P.TensorAdd()
        >>>         self.value = Tensor(np.full((1), 3), mindspore.float32)
        >>>         self.switch = P.GeSwitch()
        >>>         self.merge = P.Merge()
        >>>         self.less = P.Less()
        >>>
        >>>     def construct(self, x, y):
        >>>         cond = self.less(x, y)
        >>>         st1, sf1 = self.switch(x, cond)
        >>>         st2, sf2 = self.switch(y, cond)
        >>>         add_ret = self.add(st1, st2)
        >>>         st3, sf3 = self.switch(self.value, cond)
        >>>         sq_ret = self.square(sf3)
        >>>         ret = self.merge((add_ret, sq_ret))
        >>>         return ret[0]
        >>>
        >>> x = Tensor(10.0, dtype=mindspore.float32)
        >>> y = Tensor(5.0, dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, data, pred):
        raise NotImplementedError

    def infer_shape(self, data, pred):
        validator.check_scalar_shape_input("pred", pred)
        return (data, data)

    def infer_dtype(self, data_type, pred_type):
        validator.check_type("pred", pred_type, [type(mstype.bool_)])
        return (data_type, data_type)


class Merge(PrimitiveWithInfer):
    """
    Merges all input data to one.

    One and only one of the inputs should be selected as the output

    Inputs:
        - **inputs** (Tuple) - The data to be merged. All tuple elements should have same shape.

    Outputs:
        tuple. Output is tuple(`data`, `output_index`). The `data` has the same shape of `inputs` element.
    """

    @prim_attr_register
    def __init__(self):
        """init"""

    def __call__(self, *args):
        raise NotImplementedError

    def infer_shape(self, inputs):
        return (inputs[0], [1])

    def infer_dtype(self, inputs):
        return (inputs[0], mstype.int32)
