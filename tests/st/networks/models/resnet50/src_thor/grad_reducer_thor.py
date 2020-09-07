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
"""grad reducer cell for distributed training"""
from mindspore.nn.cell import Cell
from mindspore.communication.management import GlobalComm, get_group_size
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.ops.operations.comm_ops import AllReduce
import mindspore.common.dtype as mstype

reduce_opt = C.MultitypeFuncGraph("reduce_opt")


def _init_allreduce_operators(length, split_indices):
    """ initialize allreduce communication operators"""
    indices = split_indices[0]
    fusion = split_indices[1]
    op_list = ()
    j = 0
    for i in range(length):
        if j <= len(indices)-1:
            temp = indices[j]
        else:
            temp = length
        if i >= temp:
            j = j + 1
            fusion = fusion + 1
        op = AllReduce('sum', GlobalComm.WORLD_COMM_GROUP)
        op.add_prim_attr('fusion', fusion)
        op_list = op_list + (op,)
    return op_list


@reduce_opt.register("Function", "Number", "Function", "Tensor")
def _tensors_allreduce_mean(mul, degree, allreduce, parameters):
    """
    Apply allreduce on parameters.

    Args:
        mul(Primitive): The mul operator for parameters.
        degree (int): The mean coefficient.
        allreduce (Primitive): The communication operator for parameters.
        parameters (Tensor): The parameters before operation.

    Returns:
        Tensor, the parameters after operation.
    """
    degree = F.scalar_cast(degree, F.dtype(parameters))
    parameters = allreduce(parameters)
    cast_op = P.Cast()
    return mul(parameters, cast_op(F.scalar_to_array(1.0 / degree), F.dtype(parameters)))


_get_datatype = C.MultitypeFuncGraph("_get_datatype")


@_get_datatype.register("Tensor")
def _tensors_get_datatype(parameters):
    """
    Acquire parameters datatype.

    Args:
        parameters (Tensor): The parameters before operation.

    Returns:
        mstype, the datatype of parameters.
    """
    return F.dtype(parameters)


_cast_datatype = C.MultitypeFuncGraph("_cast_datatype")


@_cast_datatype.register("TypeType", "Tensor")
def _tensors_cast_datatype(datatype, parameters):
    """
    Cast parameters to datatype.

    Args:
        datatype (mstype): the destination datatype of parameters.
        parameters (Tensor): The parameters before operation.

    Returns:
        Tensor, the parameters after operation.
    """
    return F.cast(parameters, datatype)


class DistributedGradReducerThor(Cell):
    """
    A distributed optimizer.

    Constructs a parameters reducer Cell, which applies communication and average operations on
    single-process parameters values.

    Args:
        parameter_length (int): length of the parameters to be updated.
        split_indices(tuple): parameter split indices.
        mean (bool): When mean is true, the mean coefficient (degree) would apply on parameters. Default: False.
        degree (int): The mean coefficient. Usually it equals to device number. Default: None.

    Raises:
        ValueError: If degree is not a int or less than 0.
    """

    def __init__(self, parameter_length, split_indices, mean=True, degree=None):
        super(DistributedGradReducerThor, self).__init__(auto_prefix=False)
        self.hyper_map = C.HyperMap()
        self.mul = P.Mul()
        if degree is None:
            self.degree = get_group_size()
        else:
            if not isinstance(degree, int) or degree <= 0:
                raise ValueError("Parameter 'degree' in DistributedGradReducer should large than 0 and be int")
            self.degree = degree
        self.mean = mean
        self.op_list = _init_allreduce_operators(parameter_length, split_indices)

    def construct(self, parameters):
        datatypes = self.hyper_map(F.partial(_get_datatype), parameters)
        parameters = self.hyper_map(F.partial(_cast_datatype, mstype.float32), parameters)
        new_parameters = self.hyper_map(F.partial(reduce_opt, self.mul, self.degree), self.op_list, parameters)
        new_parameters = self.hyper_map(F.partial(_cast_datatype), datatypes, new_parameters)
        return new_parameters
