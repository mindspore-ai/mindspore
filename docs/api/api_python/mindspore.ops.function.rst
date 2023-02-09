mindspore.ops.function
=============================

MindSpore中 `mindspore.ops.function` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `mindspore.ops.function API接口变更 <https://gitee.com/mindspore/docs/blob/r1.10/resource/api_updates/func_api_updates_cn.md>`_ 。

神经网络层函数
----------------

神经网络
^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.adaptive_avg_pool2d
    mindspore.ops.adaptive_avg_pool3d
    mindspore.ops.adaptive_max_pool2d
    mindspore.ops.avg_pool2d
    mindspore.ops.bias_add
    mindspore.ops.conv2d
    mindspore.ops.ctc_greedy_decoder
    mindspore.ops.deformable_conv2d
    mindspore.ops.dropout2d
    mindspore.ops.dropout3d
    mindspore.ops.flatten
    mindspore.ops.interpolate
    mindspore.ops.lrn
    mindspore.ops.max_pool3d
    mindspore.ops.kl_div
    mindspore.ops.pad
    mindspore.ops.padding

损失函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.binary_cross_entropy_with_logits
    mindspore.ops.cross_entropy
    mindspore.ops.nll_loss
    mindspore.ops.smooth_l1_loss

激活函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.dropout
    mindspore.ops.fast_gelu
    mindspore.ops.gumbel_softmax
    mindspore.ops.hardshrink
    mindspore.ops.hardswish
    mindspore.ops.log_softmax
    mindspore.ops.mish
    mindspore.ops.selu
    mindspore.ops.sigmoid
    mindspore.ops.softsign
    mindspore.ops.soft_shrink
    mindspore.ops.softmax
    mindspore.ops.tanh

采样函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.grid_sample
    mindspore.ops.uniform_candidate_sampler

距离函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.cdist

图像函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.iou

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
    mindspore.ops.addcdiv
    mindspore.ops.addcmul
    mindspore.ops.addn
    mindspore.ops.asin
    mindspore.ops.asinh
    mindspore.ops.atan
    mindspore.ops.atan2
    mindspore.ops.atanh
    mindspore.ops.bernoulli
    mindspore.ops.bessel_i0
    mindspore.ops.bessel_i0e
    mindspore.ops.bessel_i1
    mindspore.ops.bessel_i1e
    mindspore.ops.bessel_j0
    mindspore.ops.bessel_j1
    mindspore.ops.bessel_k0
    mindspore.ops.bessel_k0e
    mindspore.ops.bessel_k1
    mindspore.ops.bessel_k1e
    mindspore.ops.bessel_y0
    mindspore.ops.bessel_y1
    mindspore.ops.bitwise_and
    mindspore.ops.bitwise_or
    mindspore.ops.bitwise_xor
    mindspore.ops.ceil
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
    mindspore.ops.inv
    mindspore.ops.invert
    mindspore.ops.lerp
    mindspore.ops.log
    mindspore.ops.log1p
    mindspore.ops.logical_and
    mindspore.ops.logical_not
    mindspore.ops.logical_or
    mindspore.ops.mul
    mindspore.ops.neg
    mindspore.ops.pow
    mindspore.ops.round
    mindspore.ops.sin
    mindspore.ops.sinh
    mindspore.ops.square
    mindspore.ops.sub
    mindspore.ops.tan
    mindspore.ops.trunc
    mindspore.ops.truncate_div
    mindspore.ops.truncate_mod
    mindspore.ops.xdivy
    mindspore.ops.xlogy

Reduction函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.amax
    mindspore.ops.amin
    mindspore.ops.argmin
    mindspore.ops.cummax
    mindspore.ops.cummin
    mindspore.ops.logsumexp
    mindspore.ops.max
    mindspore.ops.mean
    mindspore.ops.min
    mindspore.ops.norm
    mindspore.ops.prod
    mindspore.ops.std

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - function
     - Description
   * - mindspore.ops.reduce_sum
     - Refer to :class:`mindspore.ops.ReduceSum`.

比较函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.approximate_equal
    mindspore.ops.equal
    mindspore.ops.ge
    mindspore.ops.gt
    mindspore.ops.intopk
    mindspore.ops.isclose
    mindspore.ops.isfinite
    mindspore.ops.isnan
    mindspore.ops.le
    mindspore.ops.less
    mindspore.ops.maximum
    mindspore.ops.minimum
    mindspore.ops.ne
    mindspore.ops.same_type_shape

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - function
     - Description
   * - mindspore.ops.check_bprop
     - Refer to :class:`mindspore.ops.CheckBprop`.
   * - mindspore.ops.isinstance\_
     - Refer to :class:`mindspore.ops.IsInstance`.
   * - mindspore.ops.issubclass\_
     - Refer to :class:`mindspore.ops.IsSubClass`.

线性代数函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.batch_dot
    mindspore.ops.dot
    mindspore.ops.matmul
    mindspore.ops.matrix_solve
    mindspore.ops.ger
    mindspore.ops.renorm
    mindspore.ops.tensor_dot

Tensor操作函数
----------------

Tensor创建
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.eye
    mindspore.ops.fill
    mindspore.ops.linspace
    mindspore.ops.narrow
    mindspore.ops.one_hot
    mindspore.ops.ones
    mindspore.ops.ones_like

随机生成函数
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.gamma
    mindspore.ops.laplace
    mindspore.ops.multinomial
    mindspore.ops.poisson
    mindspore.ops.random_poisson
    mindspore.ops.random_categorical
    mindspore.ops.random_gamma
    mindspore.ops.standard_laplace
    mindspore.ops.standard_normal
    mindspore.ops.uniform

Array操作
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.batch_to_space_nd
    mindspore.ops.broadcast_to
    mindspore.ops.col2im
    mindspore.ops.concat
    mindspore.ops.count_nonzero
    mindspore.ops.diag
    mindspore.ops.expand_dims
    mindspore.ops.gather
    mindspore.ops.gather_d
    mindspore.ops.gather_elements
    mindspore.ops.gather_nd
    mindspore.ops.index_add
    mindspore.ops.index_fill
    mindspore.ops.inplace_add
    mindspore.ops.inplace_sub
    mindspore.ops.inplace_update
    mindspore.ops.masked_fill
    mindspore.ops.masked_select
    mindspore.ops.matrix_band_part
    mindspore.ops.matrix_diag
    mindspore.ops.matrix_diag_part
    mindspore.ops.matrix_set_diag
    mindspore.ops.meshgrid
    mindspore.ops.normal
    mindspore.ops.nonzero
    mindspore.ops.population_count
    mindspore.ops.range
    mindspore.ops.rank
    mindspore.ops.repeat_elements
    mindspore.ops.reshape
    mindspore.ops.reverse_sequence
    mindspore.ops.scatter_nd
    mindspore.ops.select
    mindspore.ops.sequence_mask
    mindspore.ops.shape
    mindspore.ops.size
    mindspore.ops.slice
    mindspore.ops.space_to_batch_nd
    mindspore.ops.sparse_segment_mean
    mindspore.ops.split
    mindspore.ops.squeeze
    mindspore.ops.stack
    mindspore.ops.tensor_scatter_add
    mindspore.ops.tensor_scatter_div
    mindspore.ops.tensor_scatter_mul
    mindspore.ops.tensor_scatter_sub
    mindspore.ops.tensor_scatter_elements
    mindspore.ops.tile
    mindspore.ops.top_k
    mindspore.ops.transpose
    mindspore.ops.unique
    mindspore.ops.unique_consecutive
    mindspore.ops.unique_with_pad
    mindspore.ops.unsorted_segment_max
    mindspore.ops.unsorted_segment_min
    mindspore.ops.unsorted_segment_prod
    mindspore.ops.unsorted_segment_sum
    mindspore.ops.unstack

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - function
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
   * - mindspore.ops.strided_slice
     - Refer to :class:`mindspore.ops.StridedSlice`.
   * - mindspore.ops.tensor_scatter_update
     - Refer to :class:`mindspore.ops.TensorScatterUpdate`.

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

稀疏函数
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.dense_to_sparse_coo
    mindspore.ops.dense_to_sparse_csr
    mindspore.ops.csr_to_coo

梯度剪裁
^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.clip_by_global_norm
    mindspore.ops.clip_by_value

Parameter操作函数
--------------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.assign
    mindspore.ops.assign_add
    mindspore.ops.assign_sub
    mindspore.ops.scatter_add
    mindspore.ops.scatter_div
    mindspore.ops.scatter_max
    mindspore.ops.scatter_min
    mindspore.ops.scatter_mul
    mindspore.ops.scatter_nd_add
    mindspore.ops.scatter_nd_div
    mindspore.ops.scatter_nd_max
    mindspore.ops.scatter_nd_min
    mindspore.ops.scatter_nd_mul
    mindspore.ops.scatter_nd_sub
    mindspore.ops.scatter_update

微分函数
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.derivative
    mindspore.ops.grad
    mindspore.ops.value_and_grad
    mindspore.ops.jet
    mindspore.ops.jvp
    mindspore.ops.vjp
    mindspore.ops.vmap

其他函数
----------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - function
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
     - Disable update during back propagation. (`stop_gradient <https://www.mindspore.cn/tutorials/en/r1.10/beginner/autograd.html#stopping-gradient-calculation>`_)
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

    mindspore.ops.core
