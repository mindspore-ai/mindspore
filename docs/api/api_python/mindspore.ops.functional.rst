mindspore.ops.functional
=============================

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

神经网络层算子
----------------

激活函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.fast_gelu
    mindspore.ops.hardshrink
    mindspore.ops.tanh

数学运算算子
----------------

逐元素运算
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.abs
    mindspore.ops.acos
    mindspore.ops.acosh
    mindspore.ops.add
    mindspore.ops.addn
    mindspore.ops.asin
    mindspore.ops.asinh
    mindspore.ops.atan
    mindspore.ops.atan2
    mindspore.ops.atanh
    mindspore.ops.bessel_i0
    mindspore.ops.bessel_i0e
    mindspore.ops.bessel_j0
    mindspore.ops.bessel_j1
    mindspore.ops.bessel_k0
    mindspore.ops.bessel_k0e
    mindspore.ops.bessel_y0
    mindspore.ops.bessel_y1
    mindspore.ops.bessel_i1
    mindspore.ops.bessel_i1e
    mindspore.ops.bessel_k1
    mindspore.ops.bessel_k1e
    mindspore.ops.bitwise_and
    mindspore.ops.bitwise_or
    mindspore.ops.bitwise_xor
    mindspore.ops.cos
    mindspore.ops.cosh
    mindspore.ops.div
    mindspore.ops.erf
    mindspore.ops.erfc
    mindspore.ops.exp
    mindspore.ops.expm1
    mindspore.ops.floor
    mindspore.ops.floor_div
    mindspore.ops.floor_mod
    mindspore.ops.invert
    mindspore.ops.lerp
    mindspore.ops.log
    mindspore.ops.logical_and
    mindspore.ops.logical_not
    mindspore.ops.logical_or
    mindspore.ops.mul
    mindspore.ops.neg
    mindspore.ops.pow
    mindspore.ops.round
    mindspore.ops.sin
    mindspore.ops.sinh
    mindspore.ops.sub
    mindspore.ops.svd
    mindspore.ops.tan

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.absolute
     - `absolute` will be deprecated in the future. Please use `mindspore.ops.abs` instead.
   * - mindspore.ops.floordiv
     - `floordiv` will be deprecated in the future. Please use `mindspore.ops.floor_div` instead.
   * - mindspore.ops.floormod
     - `floormod` will be deprecated in the future. Please use `mindspore.ops.floor_mod` instead.
   * - mindspore.ops.neg_tensor
     - `neg_tensor` will be deprecated in the future. Please use `mindspore.ops.neg` instead.
   * - mindspore.ops.pows
     - `pows` will be deprecated in the future. Please use `mindspore.ops.pow` instead.
   * - mindspore.ops.sqrt
     - Refer to :class:`mindspore.ops.Sqrt`.
   * - mindspore.ops.square
     - Refer to :class:`mindspore.ops.Square`.
   * - mindspore.ops.tensor_add
     - `tensor_add` will be deprecated in the future. Please use `mindspore.ops.add` instead.
   * - mindspore.ops.tensor_div
     - `tensor_div` will be deprecated in the future. Please use `mindspore.ops.div` instead.
   * - mindspore.ops.tensor_exp
     - `tensor_exp` will be deprecated in the future. Please use `mindspore.ops.exp` instead.
   * - mindspore.ops.tensor_expm1
     - `tensor_expm1` will be deprecated in the future. Please use `mindspore.ops.expm1` instead.
   * - mindspore.ops.tensor_floordiv
     - `tensor_floordiv` will be deprecated in the future. Please use `mindspore.ops.floor_div` instead.
   * - mindspore.ops.tensor_mod
     - `tensor_mod` will be deprecated in the future. Please use `mindspore.ops.floor_mod` instead.
   * - mindspore.ops.tensor_mul
     - `tensor_mul` will be deprecated in the future. Please use `mindspore.ops.mul` instead.
   * - mindspore.ops.tensor_pow
     - `tensor_pow` will be deprecated in the future. Please use `mindspore.ops.pow` instead.
   * - mindspore.ops.tensor_sub
     - `tensor_sub` will be deprecated in the future. Please use `mindspore.ops.sub` instead.

Reduction算子
^^^^^^^^^^^^^

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
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
   
比较算子
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.equal
    mindspore.ops.ge
    mindspore.ops.gt
    mindspore.ops.isfinite
    mindspore.ops.isnan
    mindspore.ops.le
    mindspore.ops.less
    mindspore.ops.maximum
    mindspore.ops.minimum
    mindspore.ops.same_type_shape

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.check_bprop
     - Refer to :class:`mindspore.ops.CheckBprop`.
   * - mindspore.ops.isinstance\_
     - Refer to :class:`mindspore.ops.IsInstance`.
   * - mindspore.ops.issubclass\_
     - Refer to :class:`mindspore.ops.IsSubClass`.
   * - mindspore.ops.not_equal
     - `not_equal` will be deprecated in the future. Please use `mindspore.ops.ne` instead.
   * - mindspore.ops.tensor_ge
     - `tensor_ge` will be deprecated in the future. Please use `mindspore.ops.ge` instead.
   * - mindspore.ops.tensor_gt
     - `tensor_gt` will be deprecated in the future. Please use `mindspore.ops.gt` instead.
   * - mindspore.ops.tensor_le
     - `tensor_le` will be deprecated in the future. Please use `mindspore.ops.le` instead.
   * - mindspore.ops.tensor_lt
     - `tensor_lt` will be deprecated in the future. Please use `mindspore.ops.less` instead.
   
线性代数算子
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.matmul
    mindspore.ops.cdist
    mindspore.ops.ger

Tensor操作算子
----------------

Tensor创建
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.eye
    mindspore.ops.fill
    mindspore.ops.fills
    mindspore.ops.ones
    mindspore.ops.ones_like
    mindspore.ops.zeros_like

随机生成算子
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.gamma
    mindspore.ops.multinomial
    mindspore.ops.poisson

Array操作
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.diag
    mindspore.ops.expand_dims
    mindspore.ops.gather
    mindspore.ops.gather_d
    mindspore.ops.gather_nd
    mindspore.ops.masked_select
    mindspore.ops.matrix_diag
    mindspore.ops.meshgrid
    mindspore.ops.nonzero
    mindspore.ops.one_hot
    mindspore.ops.range
    mindspore.ops.rank
    mindspore.ops.reshape
    mindspore.ops.scatter_nd
    mindspore.ops.select
    mindspore.ops.shape
    mindspore.ops.size
    mindspore.ops.tensor_scatter_add
    mindspore.ops.tensor_scatter_sub
    mindspore.ops.tensor_scatter_div
    mindspore.ops.space_to_batch_nd
    mindspore.ops.tile
    mindspore.ops.transpose
    mindspore.ops.unique
    mindspore.ops.unique_consecutive

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.cast
     - Refer to :class:`mindspore.ops.Cast`.
   * - mindspore.ops.cumprod
     - Refer to :class:`mindspore.ops.CumProd`.
   * - mindspore.ops.cumsum
     - Refer to :class:`mindspore.ops.CumSum`.
   * - mindspore.ops.dtype
     - Refer to :class:`mindspore.ops.DType`.
   * - mindspore.ops.sort
     - Refer to :class:`mindspore.ops.Sort`.
   * - mindspore.ops.squeeze
     - Refer to :class:`mindspore.ops.Squeeze`.
   * - mindspore.ops.stack
     - Refer to :class:`mindspore.ops.Stack`.
   * - mindspore.ops.strided_slice
     - Refer to :class:`mindspore.ops.StridedSlice`.
   * - mindspore.ops.tensor_scatter_update
     - Refer to :class:`mindspore.ops.TensorScatterUpdate`.
   * - mindspore.ops.tensor_slice
     - `tensor_slice` will be deprecated in the future. Please use `mindspore.ops.slice` instead.

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.scalar_cast
    mindspore.ops.scalar_to_array
    mindspore.ops.scalar_to_tensor
    mindspore.ops.tuple_to_array

Parameter操作算子
--------------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.assign
    mindspore.ops.assign_add
    mindspore.ops.assign_sub
    mindspore.ops.index_add
    mindspore.ops.scatter_min
    mindspore.ops.scatter_max
    mindspore.ops.scatter_nd_add
    mindspore.ops.scatter_nd_sub

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.scatter_nd_update
     - Refer to :class:`mindspore.ops.ScatterNdUpdate`.
   * - mindspore.ops.scatter_update
     - Refer to :class:`mindspore.ops.ScatterUpdate`.

调试算子
----------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.print\_
     - Refer to :class:`mindspore.ops.Print`.
   
其他算子
----------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - Description
   * - mindspore.ops.bool_and
     - Calculate the result of logical AND operation. (Usage is the same as "and" in Python)
   * - mindspore.ops.bool_eq
     - Determine whether the Boolean values are equal. (Usage is the same as "==" in Python)
   * - mindspore.ops.bool_not
     - Calculate the result of logical NOT operation. (Usage is the same as "not" in Python)
   * - mindspore.ops.bool_or
     - Calculate the result of logical OR operation. (Usage is the same as "or" in Python)
   * - mindspore.ops.depend
     - Refer to :class:`mindspore.ops.Depend`.
   * - mindspore.ops.in_dict
     - Determine if a str in dict.
   * - mindspore.ops.is_not
     - Determine whether the input is not the same as the other one. (Usage is the same as "is not" in Python)
   * - mindspore.ops.is\_
     - Determine whether the input is the same as the other one. (Usage is the same as "is" in Python)
   * - mindspore.ops.isconstant
     - Determine whether the object is constant.
   * - mindspore.ops.not_in_dict
     - Determine whether the object is not in the dict.
   * - mindspore.ops.partial
     - Refer to :class:`mindspore.ops.Partial`.
   * - mindspore.ops.scalar_add
     - Get the sum of two numbers. (Usage is the same as "+" in Python)
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
   * - mindspore.ops.scalar_uadd
     - Get the positive value of the input number.
   * - mindspore.ops.scalar_usub
     - Get the negative value of the input number.
   * - mindspore.ops.shape_mul
     - The input of shape_mul must be shape multiply elements in tuple(shape).
   * - mindspore.ops.stop_gradient
     - Disable update during back propagation. (`stop_gradient <https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html#stopping-gradient-calculation>`_)
   * - mindspore.ops.string_concat
     - Concatenate two strings.
   * - mindspore.ops.string_eq
     - Determine if two strings are equal.
   * - mindspore.ops.typeof
     - Get type of object.

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.arange
    mindspore.ops.batch_dot
    mindspore.ops.clip_by_global_norm
    mindspore.ops.clip_by_value
    mindspore.ops.core
    mindspore.ops.count_nonzero
    mindspore.ops.cummin
    mindspore.ops.derivative
    mindspore.ops.dot
    mindspore.ops.grad
    mindspore.ops.jet
    mindspore.ops.jvp
    mindspore.ops.laplace
    mindspore.ops.narrow
    mindspore.ops.normal
    mindspore.ops.repeat_elements
    mindspore.ops.sequence_mask
    mindspore.ops.tensor_dot
    mindspore.ops.uniform
    mindspore.ops.vjp
