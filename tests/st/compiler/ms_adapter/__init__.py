from mindspore.common.api import set_adapter_config
from mindspore._extends.parse import trope as T
from mindspore._extends.parse.resources import convert_object_map
from ._register.ms_adapter_api import Tensor, Parameter
from ._register import multitype_ops
from ._register import standard_method as S

# Update convert_object_map
convert_object_map[T.add] = multitype_ops.add
convert_object_map[T.sub] = multitype_ops.sub
convert_object_map[T.mul] = multitype_ops.mul
convert_object_map[T.truediv] = multitype_ops.div
convert_object_map[T.getitem] = multitype_ops.getitem
convert_object_map[T.setitem] = multitype_ops.setitem
convert_object_map[T.floordiv] = multitype_ops.floordiv
convert_object_map[T.mod] = multitype_ops.mod
convert_object_map[T.pow] = multitype_ops.pow_
convert_object_map[T.and_] = multitype_ops.bitwise_and
convert_object_map[T.or_] = multitype_ops.bitwise_or
convert_object_map[T.xor] = multitype_ops.bitwise_xor
convert_object_map[T.neg] = multitype_ops.negative
convert_object_map[T.not_] = multitype_ops.logical_not
convert_object_map[T.eq] = multitype_ops.equal
convert_object_map[T.ne] = multitype_ops.not_equal
convert_object_map[T.lt] = multitype_ops.less
convert_object_map[T.gt] = multitype_ops.greater
convert_object_map[T.le] = multitype_ops.less_equal
convert_object_map[T.ge] = multitype_ops.greater_equal
convert_object_map[T.contains] = multitype_ops.in_
convert_object_map[T.not_contains] = multitype_ops.not_in_
convert_object_map[T.matmul] = S.adapter_matmul
convert_object_map[T.invert] = S.adapter_invert
convert_object_map[T.abs] = S.adapter_abs
convert_object_map[T.round] = S.adapter_round
convert_object_map[T.max] = S.adapter_max
convert_object_map[T.min] = S.adapter_min
convert_object_map[T.sum] = S.adapter_sum


# map for adapter tensor convert
convert_adapter_tensor_map = {}
convert_adapter_tensor_map["Tensor"] = S.create_adapter_tensor


adapter_config = {"Tensor": Tensor, "Parameter": Parameter, "convert_object_map": convert_object_map,
                  "convert_adapter_tensor_map": convert_adapter_tensor_map}
set_adapter_config(adapter_config)


__all__ = ["Tensor", "Parameter"]
