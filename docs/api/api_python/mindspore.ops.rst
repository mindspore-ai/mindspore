mindspore.ops
=============

可用于Cell的构造函数的算子。

.. code-block::

    import mindspore.ops as ops

MindSpore中 `mindspore.ops` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `API Updates <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/ops_api_updates.md>`_ 。

.. include:: operations.rst

functional
-----------

functional算子是经过初始化后的Primitive，可以直接作为函数使用。functional算子的使用示例如下：

.. code-block:: python

    from mindspore import Tensor, ops
    from mindspore import dtype as mstype

    input_x = Tensor(-1, mstype.int32)
    input_dict = {'x':1, 'y':2}

    result_abs = ops.absolute(input_x)
    print(result_abs)

    result_in_dict = ops.in_dict('x', input_dict)
    print(result_in_dict)

    result_not_in_dict = ops.not_in_dict('x', input_dict)
    print(result_not_in_dict)

    result_isconstant = ops.isconstant(input_x)
    print(result_isconstant)

    result_typeof = ops.typeof(input_x)
    print(result_typeof)

    # outputs:
    # 1
    # True
    # False
    # True
    # Tensor[Int32]

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.absolute
     - Refer to :class:`mindspore.ops.Abs`.
   * - mindspore.ops.acos
     - Refer to :class:`mindspore.ops.ACos`.
   * - mindspore.ops.acosh
     - Refer to :class:`mindspore.ops.Acosh`.
   * - mindspore.ops.add
     - Refer to :class:`mindspore.ops.Add`.
   * - mindspore.ops.addn
     - Refer to :class:`mindspore.ops.AddN`.
   * - mindspore.ops.asin
     - Refer to :class:`mindspore.ops.Asin`.
   * - mindspore.ops.asinh
     - Refer to :class:`mindspore.ops.Asinh`.
   * - mindspore.ops.assign
     - Refer to :class:`mindspore.ops.Assign`.
   * - mindspore.ops.assign_add
     - Refer to :class:`mindspore.ops.AssignAdd`.
   * - mindspore.ops.assign_sub
     - Refer to :class:`mindspore.ops.AssignSub`.
   * - mindspore.ops.atan
     - Refer to :class:`mindspore.ops.Atan`.
   * - mindspore.ops.atan2
     - Refer to :class:`mindspore.ops.Atan2`.
   * - mindspore.ops.atanh
     - Refer to :class:`mindspore.ops.Atanh`.
   * - mindspore.ops.bitwise_and
     - Refer to :class:`mindspore.ops.BitwiseAnd`.
   * - mindspore.ops.bitwise_or
     - Refer to :class:`mindspore.ops.BitwiseOr`.
   * - mindspore.ops.bitwise_xor
     - Refer to :class:`mindspore.ops.BitwiseXor`.
   * - mindspore.ops.bool_and
     - Calculate the result of logical AND operation. (Usage is the same as "and" in Python)
   * - mindspore.ops.bool_eq
     - Determine whether the Boolean values are equal. (Usage is the same as "==" in Python)
   * - mindspore.ops.bool_not
     - Calculate the result of logical NOT operation. (Usage is the same as "not" in Python)
   * - mindspore.ops.bool_or
     - Calculate the result of logical OR operation. (Usage is the same as "or" in Python)
   * - mindspore.ops.cast
     - Refer to :class:`mindspore.ops.Cast`.
   * - mindspore.ops.check_bprop
     - Refer to :class:`mindspore.ops.CheckBprop`.
   * - mindspore.ops.cos
     - Refer to :class:`mindspore.ops.Cos`.
   * - mindspore.ops.cosh
     - Refer to :class:`mindspore.ops.Cosh`.
   * - mindspore.ops.cumprod
     - Refer to :class:`mindspore.ops.CumProd`.
   * - mindspore.ops.cumsum
     - Refer to :class:`mindspore.ops.CumSum`.
   * - mindspore.ops.div
     - Refer to :class:`mindspore.ops.RealDiv`.
   * - mindspore.ops.depend
     - Refer to :class:`mindspore.ops.Depend`.
   * - mindspore.ops.dtype
     - Refer to :class:`mindspore.ops.DType`.
   * - mindspore.ops.erf
     - Refer to :class:`mindspore.ops.Erf`.
   * - mindspore.ops.erfc
     - Refer to :class:`mindspore.ops.Erfc`.
   * - mindspore.ops.eye
     - Refer to :class:`mindspore.ops.Eye`.
   * - mindspore.ops.equal
     - Refer to :class:`mindspore.ops.Equal`.
   * - mindspore.ops.expand_dims
     - Refer to :class:`mindspore.ops.ExpandDims`.
   * - mindspore.ops.exp
     - Refer to :class:`mindspore.ops.Exp`.
   * - mindspore.ops.fill
     - Refer to :class:`mindspore.ops.Fill`.
   * - mindspore.ops.floor
     - Refer to :class:`mindspore.ops.Floor`.
   * - mindspore.ops.floordiv
     - Refer to :class:`mindspore.ops.FloorDiv`.
   * - mindspore.ops.floormod
     - Refer to :class:`mindspore.ops.FloorMod`.
   * - mindspore.ops.gather
     - Refer to :class:`mindspore.ops.Gather`.
   * - mindspore.ops.gather_d
     - Refer to :class:`mindspore.ops.GatherD`.
   * - mindspore.ops.gather_nd
     - Refer to :class:`mindspore.ops.GatherNd`.
   * - mindspore.ops.ge
     - Refer to :class:`mindspore.ops.GreaterEqual`.
   * - mindspore.ops.gt
     - Refer to :class:`mindspore.ops.Greater`.
   * - mindspore.ops.invert
     - Refer to :class:`mindspore.ops.Invert`.
   * - mindspore.ops.in_dict
     - Determine if a str in dict.
   * - mindspore.ops.is_not
     - Determine whether the input is not the same as the other one. (Usage is the same as "is not" in Python)
   * - mindspore.ops.is\_
     - Determine whether the input is the same as the other one. (Usage is the same as "is" in Python)
   * - mindspore.ops.isconstant
     - Determine whether the object is constant.
   * - mindspore.ops.isfinite
     - Refer to :class:`mindspore.ops.IsFinite`.
   * - mindspore.ops.isinstance\_
     - Refer to :class:`mindspore.ops.IsInstance`.
   * - mindspore.ops.isnan
     - Refer to :class:`mindspore.ops.IsNan`.
   * - mindspore.ops.issubclass\_
     - Refer to :class:`mindspore.ops.IsSubClass`.
   * - mindspore.ops.log
     - Refer to :class:`mindspore.ops.Log`.
   * - mindspore.ops.logical_and
     - Refer to :class:`mindspore.ops.LogicalAnd`.
   * - mindspore.ops.le
     - Refer to :class:`mindspore.ops.LessEqual`.
   * - mindspore.ops.less
     - Refer to :class:`mindspore.ops.Less`.
   * - mindspore.ops.logical_and
     - Refer to :class:`mindspore.ops.LogicalAnd`.
   * - mindspore.ops.logical_not
     - Refer to :class:`mindspore.ops.LogicalNot`.
   * - mindspore.ops.logical_or
     - Refer to :class:`mindspore.ops.LogicalOr`.
   * - mindspore.ops.maximum
     - Refer to :class:`mindspore.ops.Maximum`.
   * - mindspore.ops.minimum
     - Refer to :class:`mindspore.ops.Minimum`.
   * - mindspore.ops.mul
     - Refer to :class:`mindspore.ops.Mul`.
   * - mindspore.ops.neg_tensor
     - Refer to :class:`mindspore.ops.Neg`.
   * - mindspore.ops.not_equal
     - Refer to :class:`mindspore.ops.NotEqual`.
   * - mindspore.ops.not_in_dict
     - Determine whether the object is not in the dict.
   * - mindspore.ops.ones_like
     - Refer to :class:`mindspore.ops.OnesLike`.
   * - mindspore.ops.partial
     - Refer to :class:`mindspore.ops.Partial`.
   * - mindspore.ops.pows
     - Refer to :class:`mindspore.ops.Pow`.
   * - mindspore.ops.print\_
     - Refer to :class:`mindspore.ops.Print`.
   * - mindspore.ops.rank
     - Refer to :class:`mindspore.ops.Rank`.
   * - mindspore.ops.reduce_max
     - Refer to :class:`mindspore.ops.ReduceMax`.
   * - mindspore.ops.reduce_mean
     - Refer to :class:`mindspore.ops.ReduceMean`.
   * - mindspore.ops.reduce_min
     - Refer to :class:`mindspore.ops.ReduceMin`.
   * - mindspore.ops.reduce_prod
     - Refer to :class:`mindspore.ops.ReduceProd`.
   * - mindspore.ops.reduce_sum
     - Refer to :class:`mindspore.ops.ReduceSum`.
   * - mindspore.ops.reshape
     - Refer to :class:`mindspore.ops.Reshape`.
   * - mindspore.ops.same_type_shape
     - Refer to :class:`mindspore.ops.SameTypeShape`.
   * - mindspore.ops.scalar_add
     - Get the sum of two numbers. (Usage is the same as "+" in Python)
   * - mindspore.ops.scalar_cast
     - Refer to :class:`mindspore.ops.ScalarCast`.
   * - mindspore.ops.scalar_div
     - Get the quotient of dividing the first input number by the second input number. (Usage is the same as "/" in Python)
   * - mindspore.ops.scalar_eq
     - Determine whether two numbers are equal. (Usage is the same as "==" in Python)
   * - mindspore.ops.scalar_floordiv
     - Divide the first input number by the second input number and round down to the closest integer. (Usage is the same as "//" in Python)
   * - mindspore.ops.scalar_ge
     - Determine whether the number is greater than or equal to another number. (Usage is the same as ">=" in Python)
   * - mindspore.ops.scalar_gt
     - Determine whether the number is greater than another number. (Usage is the same as ">" in Python)
   * - mindspore.ops.scalar_le
     - Determine whether the number is less than or equal to another number. (Usage is the same as "<=" in Python)
   * - mindspore.ops.scalar_log
     - Get the natural logarithm of the input number.
   * - mindspore.ops.scalar_lt
     - Determine whether the number is less than another number. (Usage is the same as "<" in Python)
   * - mindspore.ops.scalar_mod
     - Get the remainder of dividing the first input number by the second input number. (Usage is the same as "%" in Python)
   * - mindspore.ops.scalar_mul
     - Get the product of the input two numbers. (Usage is the same as "*" in Python)
   * - mindspore.ops.scalar_ne
     - Determine whether two numbers are not equal. (Usage is the same as "!=" in Python)
   * - mindspore.ops.scalar_pow
     - Compute a number to the power of the second input number.
   * - mindspore.ops.scalar_sub
     - Subtract the second input number from the first input number. (Usage is the same as "-" in Python)
   * - mindspore.ops.scalar_to_array
     - Refer to :class:`mindspore.ops.ScalarToArray`.
   * - mindspore.ops.scalar_to_tensor
     - Refer to :class:`mindspore.ops.ScalarToTensor`.
   * - mindspore.ops.scalar_uadd
     - Get the positive value of the input number.
   * - mindspore.ops.scalar_usub
     - Get the negative value of the input number.
   * - mindspore.ops.scatter_nd
     - Refer to :class:`mindspore.ops.ScatterNd`.
   * - mindspore.ops.scatter_nd_update
     - Refer to :class:`mindspore.ops.ScatterNdUpdate`.
   * - mindspore.ops.scatter_update
     - Refer to :class:`mindspore.ops.ScatterUpdate`.
   * - mindspore.ops.shape
     - Refer to :class:`mindspore.ops.Shape`.
   * - mindspore.ops.shape_mul
     - The input of shape_mul must be shape multiply elements in tuple(shape).
   * - mindspore.ops.sin
     - Refer to :class:`mindspore.ops.Sin`.
   * - mindspore.ops.sinh
     - Refer to :class:`mindspore.ops.Sinh`.
   * - mindspore.ops.size
     - Refer to :class:`mindspore.ops.Size`.
   * - mindspore.ops.sort
     - Refer to :class:`mindspore.ops.Sort`.
   * - mindspore.ops.sqrt
     - Refer to :class:`mindspore.ops.Sqrt`.
   * - mindspore.ops.square
     - Refer to :class:`mindspore.ops.Square`.
   * - mindspore.ops.squeeze
     - Refer to :class:`mindspore.ops.Squeeze`.
   * - mindspore.ops.stack
     - Refer to :class:`mindspore.ops.Stack`.
   * - mindspore.ops.stop_gradient
     - Disable update during back propagation. (`stop_gradient <https://www.mindspore.cn/tutorials/en/master/autograd.html#stop-gradient>`_)
   * - mindspore.ops.strided_slice
     - Refer to :class:`mindspore.ops.StridedSlice`.
   * - mindspore.ops.string_concat
     - Concatenate two strings.
   * - mindspore.ops.string_eq
     - Determine if two strings are equal.
   * - mindspore.ops.sub
     - Refer to :class:`mindspore.ops.Sub`.
   * - mindspore.ops.tan
     - Refer to :class:`mindspore.ops.Tan`.
   * - mindspore.ops.tanh
     - Refer to :class:`mindspore.ops.Tanh`.
   * - mindspore.ops.tensor_add
     - Refer to :class:`mindspore.ops.Add`.
   * - mindspore.ops.tensor_div
     - Refer to :class:`mindspore.ops.RealDiv`.
   * - mindspore.ops.tensor_exp
     - Refer to :class:`mindspore.ops.Exp`.
   * - mindspore.ops.tensor_expm1
     - Refer to :class:`mindspore.ops.Expm1`.
   * - mindspore.ops.tensor_floordiv
     - Refer to :class:`mindspore.ops.FloorDiv`.
   * - mindspore.ops.tensor_ge
     - Refer to :class:`mindspore.ops.GreaterEqual`.
   * - mindspore.ops.tensor_gt
     - Refer to :class:`mindspore.ops.Greater`.
   * - mindspore.ops.tensor_le
     - Refer to :class:`mindspore.ops.LessEqual`.
   * - mindspore.ops.tensor_lt
     - Refer to :class:`mindspore.ops.Less`.
   * - mindspore.ops.tensor_mod
     - Refer to :class:`mindspore.ops.FloorMod`.
   * - mindspore.ops.tensor_mul
     - Refer to :class:`mindspore.ops.Mul`.
   * - mindspore.ops.tensor_pow
     - Refer to :class:`mindspore.ops.Pow`.
   * - mindspore.ops.tensor_scatter_add
     - Refer to :class:`mindspore.ops.TensorScatterAdd`.
   * - mindspore.ops.tensor_scatter_update
     - Refer to :class:`mindspore.ops.TensorScatterUpdate`.
   * - mindspore.ops.tensor_slice
     - Refer to :class:`mindspore.ops.Slice`.
   * - mindspore.ops.tensor_sub
     - Refer to :class:`mindspore.ops.Sub`.
   * - mindspore.ops.tile
     - Refer to :class:`mindspore.ops.Tile`.
   * - mindspore.ops.transpose
     - Refer to :class:`mindspore.ops.Transpose`.
   * - mindspore.ops.tuple_to_array
     - Refer to :class:`mindspore.ops.TupleToArray`.
   * - mindspore.ops.typeof
     - Get type of object.
   * - mindspore.ops.zeros_like
     - Refer to :class:`mindspore.ops.ZerosLike`.

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.batch_dot
    mindspore.ops.clip_by_global_norm
    mindspore.ops.clip_by_value
    mindspore.ops.core
    mindspore.ops.count_nonzero
    mindspore.ops.cummin
    mindspore.ops.dot
    mindspore.ops.gamma
    mindspore.ops.grad
    mindspore.ops.GradOperation
    mindspore.ops.HyperMap
    mindspore.ops.jvp
    mindspore.ops.laplace
    mindspore.ops.Map
    mindspore.ops.matmul
    mindspore.ops.multinomial
    mindspore.ops.MultitypeFuncGraph
    mindspore.ops.narrow
    mindspore.ops.normal
    mindspore.ops.poisson
    mindspore.ops.repeat_elements
    mindspore.ops.select
    mindspore.ops.sequence_mask
    mindspore.ops.tensor_dot
    mindspore.ops.uniform
    mindspore.ops.vjp

原语
-----

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.constexpr
    mindspore.ops.prim_attr_register
    mindspore.ops.Primitive
    mindspore.ops.PrimitiveWithCheck
    mindspore.ops.PrimitiveWithInfer

函数实现注册
----------------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.get_vm_impl_fn

算子信息注册
----------------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.AiCPURegOp
    mindspore.ops.custom_info_register
    mindspore.ops.CustomRegOp
    mindspore.ops.DataType
    mindspore.ops.op_info_register
    mindspore.ops.TBERegOp

自定义算子
----------------

.. mscnautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.Custom
    mindspore.ops.ms_hybrid
