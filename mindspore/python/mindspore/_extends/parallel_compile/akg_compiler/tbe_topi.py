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
"""tbe topi"""
import math
import functools
from enum import Enum


class OpPattern(Enum):
    """Pattern of operator."""

    ELEMWISE = "ELEMWISE"
    BROADCAST = "BROADCAST"
    REDUCE = "REDUCE"
    OPAQUE = "OPAQUE"


def reg_op(op_name, kernel_name=None, pattern=OpPattern.OPAQUE):
    """Register TBE op."""
    if kernel_name is None:
        kernel_name = op_name.lower()
    if not isinstance(kernel_name, str):
        raise TypeError("kernel_name must be str, but got {} with type {}".format(kernel_name, type(kernel_name)))

    def decorator(func):
        registered_ops = getattr(reg_op, "registered_ops", {})
        registered_ops[op_name] = {"func": func, "kernel_name": kernel_name, "pattern": pattern}
        setattr(reg_op, "registered_ops", registered_ops)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_op_reg_info(op_name, key, strict=True):
    """Get op register info."""
    registered_ops = getattr(reg_op, "registered_ops", None)
    if not isinstance(registered_ops, dict) or registered_ops.get(op_name) is None:
        if strict:
            raise ValueError("Op [{}] not found! Please register it first.".format(op_name))
        return None
    return registered_ops[op_name][key]


def _broadcast(lhs, rhs):
    """Broadcast inputs."""
    from te import tvm
    from te.utils import shape_util
    import te.lang.cce as tbe
    if isinstance(lhs, tvm.tensor.Tensor) and isinstance(rhs, tvm.expr.ConstExpr):
        return lhs, tbe.broadcast(rhs, lhs.shape)
    if isinstance(lhs, tvm.expr.ConstExpr) and isinstance(rhs, tvm.tensor.Tensor):
        return tbe.broadcast(lhs, rhs.shape), rhs
    if isinstance(lhs, tvm.expr.ConstExpr) and isinstance(rhs, tvm.expr.ConstExpr):
        shape = [1]
        return tbe.broadcast(lhs, shape), tbe.broadcast(rhs, shape)
    if isinstance(lhs, tvm.tensor.Tensor) and isinstance(rhs, tvm.tensor.Tensor):
        shape1 = shape_util.shape_to_list(lhs.shape)
        shape2 = shape_util.shape_to_list(rhs.shape)
        if shape1 != shape2:
            _, _, shape = shape_util.broadcast_shapes(shape1, shape2, param_name_input1="lhs", param_name_input2="rhs")
            return tbe.broadcast(lhs, shape), tbe.broadcast(rhs, shape)
        return lhs, rhs
    raise TypeError("Broadcast only supports tvm.tensor.Tensor or tvm.expr.ConstExpr, but got {}, {}"
                    .format(type(lhs), type(rhs)))


@reg_op("Abs", pattern=OpPattern.ELEMWISE)
def _abs(x, attrs=None):
    """Abs"""
    from impl.abs import abs_compute
    return abs_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("ACos", pattern=OpPattern.ELEMWISE)
def _acos(x, attrs=None):
    """ACos"""
    from impl.acos import acos_compute
    return acos_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Acosh", pattern=OpPattern.ELEMWISE)
def _acosh(x, attrs=None):
    """Acosh"""
    from impl.acosh import acosh_compute
    return acosh_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Add", pattern=OpPattern.ELEMWISE)
def _add(x0, x1, attrs=None):
    """Add"""
    from te import tvm
    import te.lang.cce as tbe
    is_float = x0.dtype in ["float16", "float32"]
    if is_float and isinstance(x0, tvm.tensor.Tensor) and isinstance(x1, tvm.expr.ConstExpr):
        return tbe.vadds(x0, x1)
    if is_float and isinstance(x0, tvm.expr.ConstExpr) and isinstance(x1, tvm.tensor.Tensor):
        return tbe.vadds(x1, x0)
    x0, x1 = _broadcast(x0, x1)
    from impl.add import add_compute
    return add_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Asin", pattern=OpPattern.ELEMWISE)
def _asin(x, attrs=None):
    """Asin"""
    from impl.asin import asin_compute
    return asin_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Asinh", pattern=OpPattern.ELEMWISE)
def _asinh(x, attrs=None):
    """Asinh"""
    from impl.asinh import asinh_compute
    return asinh_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Atan", pattern=OpPattern.ELEMWISE)
def _atan(x, attrs=None):
    """Atan"""
    from impl.atan import atan_compute
    return atan_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Atan2", pattern=OpPattern.ELEMWISE)
def _atan2(x, attrs=None):
    """Atan2"""
    from impl.atan2 import atan2_compute
    return atan2_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Cast", pattern=OpPattern.ELEMWISE)
def _cast(x, attrs=None):
    """Cast"""
    src_type = x.dtype
    dst_type = attrs["dst_type"]
    if src_type == "int64":
        from te import tvm
        from impl.cast import _kernel_ir
        res = tvm.extern([x.shape], [x], lambda ins, outs: _kernel_ir(outs, ins, dst_type, "int64"), name="res",
                         dtype=dst_type)
    else:
        from impl.cast import cast_compute
        res = cast_compute(x, None, dst_type, kernel_name=attrs["fusion_op_name"])
    return res


@reg_op("Cos", pattern=OpPattern.ELEMWISE)
def _cos(x, attrs=None):
    """Cos"""
    from impl.cos import cos_compute
    return cos_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Div", pattern=OpPattern.ELEMWISE)
def _div(x0, x1, attrs=None):
    """Div"""
    x0, x1 = _broadcast(x0, x1)
    from impl.div import div_compute
    return div_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Equal", pattern=OpPattern.ELEMWISE)
def _equal(x0, x1, attrs=None):
    """Equal"""
    x0, x1 = _broadcast(x0, x1)
    from impl.equal import equal_compute
    return equal_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Erf", pattern=OpPattern.ELEMWISE)
def _erf(x, attrs=None):
    """Erf"""
    from impl.erf import erf_compute
    return erf_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Erfc", pattern=OpPattern.ELEMWISE)
def _erfc(x, attrs=None):
    """Erfc"""
    from impl.erfc import erfc_compute
    return erfc_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Exp", pattern=OpPattern.ELEMWISE)
def _exp(x, attrs=None):
    """Exp"""
    base = attrs.get("base", -1.0)
    scale = attrs.get("scale", 1.0)
    shift = attrs.get("shift", 0.0)
    from impl.exp import exp_compute
    return exp_compute(x, None, base, scale, shift, kernel_name=attrs["fusion_op_name"])


@reg_op("Expm1", pattern=OpPattern.ELEMWISE)
def _expm1(x, attrs=None):
    """Expm1"""
    from impl.expm1 import expm1_compute
    return expm1_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Floor", pattern=OpPattern.ELEMWISE)
def _floor(x, attrs=None):
    """Floor"""
    from impl.floor import floor_compute
    return floor_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("FloorDiv", "floor_div", OpPattern.ELEMWISE)
def _floordiv(x0, x1, attrs=None):
    """FloorDiv"""
    x0, x1 = _broadcast(x0, x1)
    from impl.floor_div import floor_div_compute
    return floor_div_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("FloorMod", "floor_mod", OpPattern.ELEMWISE)
def _floormod(x0, x1, attrs=None):
    """FloorMod"""
    x0, x1 = _broadcast(x0, x1)
    from impl.floor_mod import floor_mod_compute
    return floor_mod_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Gelu", pattern=OpPattern.ELEMWISE)
def _gelu(x, attrs=None):
    """Gelu"""
    from impl.gelu import gelu_compute
    return gelu_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Greater", pattern=OpPattern.ELEMWISE)
def _greater(x0, x1, attrs=None):
    """Greater"""
    x0, x1 = _broadcast(x0, x1)
    from impl.greater import greater_compute
    return greater_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("GreaterEqual", "greater_equal", OpPattern.ELEMWISE)
def _greater_equal(x0, x1, attrs=None):
    """GreaterEqual"""
    x0, x1 = _broadcast(x0, x1)
    from impl.greater_equal import greater_equal_compute
    return greater_equal_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Less", pattern=OpPattern.ELEMWISE)
def _less(x0, x1, attrs=None):
    """Less"""
    x0, x1 = _broadcast(x0, x1)
    from impl.less import less_compute
    return less_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("LessEqual", "less_equal", OpPattern.ELEMWISE)
def _less_equal(x0, x1, attrs=None):
    """LessEqual"""
    x0, x1 = _broadcast(x0, x1)
    from impl.less_equal import less_equal_compute
    return less_equal_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("LogicalAnd", "logical_and", OpPattern.ELEMWISE)
def _logical_and(x0, x1, attrs=None):
    """LogicalAnd"""
    x0, x1 = _broadcast(x0, x1)
    from impl.logical_and import logical_and_compute
    return logical_and_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("LogicalNot", "logical_not", OpPattern.ELEMWISE)
def _logical_not(x0, x1, attrs=None):
    """LogicalNot"""
    x0, x1 = _broadcast(x0, x1)
    from impl.logical_not import logical_not_compute
    return logical_not_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("LogicalOr", "logical_or", OpPattern.ELEMWISE)
def _logical_or(x0, x1, attrs=None):
    """LogicalOr"""
    x0, x1 = _broadcast(x0, x1)
    from impl.logical_or import logical_or_compute
    return logical_or_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Log", pattern=OpPattern.ELEMWISE)
def _log(x, attrs=None):
    """Log"""
    base = attrs.get("base", -1.0)
    scale = attrs.get("scale", 1.0)
    shift = attrs.get("shift", 0.0)
    if base <= 0 and not math.isclose(base, -1.0, rel_tol=1e-8, abs_tol=0.0):
        raise ValueError("base must be strictly positive or -1, but got {}".format(base))
    from impl.log import log_compute
    return log_compute(x, None, base, scale, shift, kernel_name=attrs["fusion_op_name"])


@reg_op("Maximum", pattern=OpPattern.ELEMWISE)
def _maximum(x0, x1, attrs=None):
    """Maximum"""
    x0, x1 = _broadcast(x0, x1)
    from impl.maximum import maximum_compute
    return maximum_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Minimum", pattern=OpPattern.ELEMWISE)
def _minimum(x0, x1, attrs=None):
    """Minimum"""
    x0, x1 = _broadcast(x0, x1)
    from impl.minimum import minimum_compute
    return minimum_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Mod", pattern=OpPattern.ELEMWISE)
def _mod(x0, x1, attrs=None):
    """Mod"""
    x0, x1 = _broadcast(x0, x1)
    from impl.mod import mod_compute
    return mod_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Mul", pattern=OpPattern.ELEMWISE)
def _mul(x0, x1, attrs=None):
    """Mul"""
    from te import tvm
    import te.lang.cce as tbe
    is_float = x0.dtype in ["float16", "float32"]
    if is_float and isinstance(x0, tvm.tensor.Tensor) and isinstance(x1, tvm.expr.ConstExpr):
        return tbe.vmuls(x0, x1)
    if is_float and isinstance(x0, tvm.expr.ConstExpr) and isinstance(x1, tvm.tensor.Tensor):
        return tbe.vmuls(x1, x0)
    x0, x1 = _broadcast(x0, x1)
    from impl.mul import mul_compute
    return mul_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Neg", pattern=OpPattern.ELEMWISE)
def _neg(x, attrs=None):
    """Neg"""
    from impl.neg import neg_compute
    return neg_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("NotEqual", "not_equal", OpPattern.ELEMWISE)
def _not_equal(x0, x1, attrs=None):
    """NotEqual"""
    x0, x1 = _broadcast(x0, x1)
    from impl.not_equal import not_equal_compute
    return not_equal_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Pow", pattern=OpPattern.ELEMWISE)
def _pow(x0, x1, attrs=None):
    """Pow"""
    x0, x1 = _broadcast(x0, x1)
    from impl.pow import pow_compute
    return pow_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("RealDiv", "real_div", OpPattern.ELEMWISE)
def _realdiv(x0, x1, attrs=None):
    """RealDiv"""
    from te import tvm
    import te.lang.cce as tbe
    if x0.dtype in ["float16", "float32"] and isinstance(x0, tvm.tensor.Tensor) and isinstance(x1, tvm.expr.ConstExpr):
        return tbe.vmuls(x0, tvm.const(1.0 / x1.value, x0.dtype))
    x0, x1 = _broadcast(x0, x1)
    from impl.real_div import real_div_compute
    return real_div_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Reciprocal", pattern=OpPattern.ELEMWISE)
def _reciprocal(x, attrs=None):
    """Reciprocal"""
    from impl.reciprocal import reciprocal_compute
    return reciprocal_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Relu", pattern=OpPattern.ELEMWISE)
def _relu(x, attrs=None):
    """Relu"""
    from impl.relu import relu_compute
    return relu_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Rsqrt", pattern=OpPattern.ELEMWISE)
def _rsqrt(x, attrs=None):
    """Rsqrt"""
    from impl.rsqrt import rsqrt_compute
    return rsqrt_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Sign", pattern=OpPattern.ELEMWISE)
def _sign(x, attrs=None):
    """Sign"""
    from impl.sign import sign_compute
    return sign_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Sin", pattern=OpPattern.ELEMWISE)
def _sin(x, attrs=None):
    """Sin"""
    from impl.sin import sin_compute
    return sin_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Sqrt", pattern=OpPattern.ELEMWISE)
def _sqrt(x, attrs=None):
    """Sqrt"""
    from impl.sqrt import sqrt_compute
    return sqrt_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Square", pattern=OpPattern.ELEMWISE)
def _square(x, attrs=None):
    """Square"""
    from impl.square import square_compute
    return square_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Sub", pattern=OpPattern.ELEMWISE)
def _sub(x0, x1, attrs=None):
    """Sub"""
    from te import tvm
    import te.lang.cce as tbe
    is_float = x0.dtype in ["float16", "float32"]
    if is_float and isinstance(x0, tvm.tensor.Tensor) and isinstance(x1, tvm.expr.ConstExpr):
        return tbe.vadds(x0, tvm.const(-1.0 * x1.value, x0.dtype))
    if is_float and isinstance(x0, tvm.expr.ConstExpr) and isinstance(x1, tvm.tensor.Tensor):
        tmp = tbe.vmuls(x1, tvm.const(-1, x1.dtype))
        return tbe.vadds(tmp, tvm.const(x0.value, x0.dtype))
    x0, x1 = _broadcast(x0, x1)
    from impl.sub import sub_compute
    return sub_compute(x0, x1, None, kernel_name=attrs["fusion_op_name"])


@reg_op("Tanh", pattern=OpPattern.ELEMWISE)
def _tanh(x, attrs=None):
    """Tanh"""
    from impl.tanh import tanh_compute
    return tanh_compute(x, None, kernel_name=attrs["fusion_op_name"])


@reg_op("BroadcastTo", "broadcast", OpPattern.BROADCAST)
def _broadcast_to(x, attrs=None):
    """BroadcastTo"""
    from te import tvm
    import te.lang.cce as tbe
    shape = attrs["shape"]
    if isinstance(x, tvm.tensor.Tensor):
        from impl.broadcast_to_d import broadcast_to_compute
        return broadcast_to_compute(x, None, shape, kernel_name=attrs["fusion_op_name"])
    return tbe.broadcast(x, shape, x.dtype)


@reg_op("ReduceSum", "reduce_sum", OpPattern.REDUCE)
def _reduce_sum(x, attrs=None):
    """ReduceSum"""
    from te.utils import shape_util
    axis = attrs["axis"]
    keep_dims = attrs["keep_dims"]
    if not axis:
        axis = [i for i in range(len(x.shape))]
    if not isinstance(axis, list):
        axis = [axis]
    axis = shape_util.axis_check(len(x.shape), axis)
    from impl.reduce_sum_d import reduce_sum_d_compute
    return reduce_sum_d_compute(x, None, axis, keep_dims, kernel_name=attrs["fusion_op_name"])


@reg_op("ReduceMax", "reduce_max", OpPattern.REDUCE)
def _reduce_max(x, attrs=None):
    """ReduceMax"""
    from te.utils import shape_util
    axis = attrs["axis"]
    keep_dims = attrs["keep_dims"]
    if not axis:
        axis = [i for i in range(len(x.shape))]
    if not isinstance(axis, list):
        axis = [axis]
    axis = shape_util.axis_check(len(x.shape), axis)
    from impl.reduce_max_d import reduce_max_d_compute
    return reduce_max_d_compute(x, None, axis, keep_dims, kernel_name=attrs["fusion_op_name"])


@reg_op("ReduceMin", "reduce_min", OpPattern.REDUCE)
def _reduce_min(x, attrs=None):
    """ReduceMin"""
    from te.utils import shape_util
    axis = attrs["axis"]
    keep_dims = attrs["keep_dims"]
    if not axis:
        axis = [i for i in range(len(x.shape))]
    if not isinstance(axis, list):
        axis = [axis]
    axis = shape_util.axis_check(len(x.shape), axis)
    from impl.reduce_min_d import reduce_min_d_compute
    return reduce_min_d_compute(x, None, axis, keep_dims, kernel_name=attrs["fusion_op_name"])


@reg_op("BatchMatMul", "batch_matmul")
def _batch_matmul(x0, x1, bias=None, attrs=None):
    """BatchMatMul"""
    dst_type = attrs["dst_type"]
    dst_ori_shape = attrs["dst_ori_shape"]
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    from impl.batch_matmul import batch_matmul_compute
    return batch_matmul_compute(x0, x1, bias=bias, output_z={"dtype": dst_type, "ori_shape": dst_ori_shape},
                                trans_a=transpose_a, trans_b=transpose_b, kernel_name=attrs["fusion_op_name"])


@reg_op("MatMul", "mat_mul")
def _matmul(x0, x1, bias=None, attrs=None):
    """MatMul"""
    dst_type = attrs["dst_type"]
    dst_format = attrs["dst_format"]
    transpose_a = attrs["transpose_a"]
    transpose_b = attrs["transpose_b"]
    from impl.mat_mul import mat_mul_compute
    return mat_mul_compute(x0, x1, bias=bias, offset_w=None, output_y={"dtype": dst_type, "format": dst_format},
                           trans_a=transpose_a, trans_b=transpose_b, kernel_name=attrs["fusion_op_name"])


@reg_op("Conv2D")
def _conv2d(x0, x1, bias=None, offset_w=None, attrs=None):
    """Conv2D"""
    strides = attrs["stride"]
    pads = attrs["pad"]
    dilations = attrs["dilation"]
    groups = attrs["groups"]
    data_format = attrs["format"]
    from impl.conv2d import conv2d_compute
    return conv2d_compute(x0, x1, bias, offset_w, None, strides=strides, pads=pads, dilations=dilations,
                          groups=groups, data_format=data_format, kernel_name=attrs["fusion_op_name"])
