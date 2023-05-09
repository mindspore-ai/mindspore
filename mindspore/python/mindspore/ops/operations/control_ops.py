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

from __future__ import absolute_import
from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype


class GeSwitch(PrimitiveWithInfer):
    """
    Adds control switch to data.

    Switch data flows into ``False`` or ``True`` branch depending on the condition. If the condition is ``True`` ,
    the ``True`` branch will be activated, or vise verse.

    Inputs:
        - **data** (Union[Tensor, Number]) - The data to be used for switch control.
        - **pred** (Tensor) - It must be a scalar whose type is bool and shape is `()`, It is used as condition for
          switch control.
    Outputs:
        tuple. Output is tuple(false_output, true_output). The Elements in the tuple has the same shape of input data.
        The false_output connects with the false_branch and the true_output connects with the true_branch.

    Raises:
        TypeError: If `data` is neither a Tensor nor a Number.
        TypeError: If `pred` is not a Tensor.

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
        """Initialize GeSwitch."""

    def __call__(self, data, pred):
        raise NotImplementedError

    def infer_shape(self, data, pred):
        validator.check_int_range(len(pred), 0, 1, validator.INC_BOTH, "pred rank", self.name)
        return data, data

    def infer_dtype(self, data_type, pred_type):
        validator.check_subclass(
            "data", data_type, (mstype.tensor_type,) + mstype.number_type, self.name)
        validator.check_tensor_dtype_valid("pred", pred_type, [mstype.bool_], self.name)
        return data_type, data_type


class Merge(PrimitiveWithInfer):
    """
    Merges all input data to one.

    One and only one of the inputs must be selected as the output

    Inputs:
        - **inputs** (Union(Tuple, List)) - The data to be merged. All tuple elements must have the same data type.

    Outputs:
        tuple. Output is tuple(`data`, `output_index`). The `data` has the same shape of `inputs` element.

    Raises:
        TypeError: If `inputs` is neither Tuple nor list.

    Examples:
        >>> merge = ops.Merge()
        >>> input_x = Tensor(np.linspace(0, 8, 8).reshape(2, 4), mindspore.float32)
        >>> input_y = Tensor(np.random.randint(-4, 4, (2, 4)), mindspore.float32)
        >>> result = merge((input_x, input_y))
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Merge."""

    def __call__(self, *args):
        raise NotImplementedError

    def infer_shape(self, inputs):
        return inputs[0], [1]

    def infer_dtype(self, inputs):
        args = {}
        for i, item in enumerate(inputs):
            args['inputs[%d]' % i] = item

        validator.check_scalar_or_tensor_types_same(args, (mstype.bool_,) + mstype.number_type, self.name)
        return inputs[0], mstype.int32
