mindspore.Tensor
================

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None, internal=False, const_arg=False)

    张量，即存储多维数组（n-dimensional array）的数据结构。

    参数：
        - **input_data** (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]) - 被存储的数据，可以是其它Tensor，也可以是Python基本数据（如int，float，bool等），或是一个NumPy对象。默认值：None。
        - **dtype** (:class:`mindspore.dtype`) - 用于定义该Tensor的数据类型，必须是 *mindspore.dtype* 中定义的类型。如果该参数为None，则数据类型与 `input_data` 一致，默认值：None。
        - **shape** (Union[tuple, list, int]) - 用于定义该Tensor的形状。如果指定了 `input_data` ，则无需设置该参数。默认值：None。
        - **init** (Initializer) - 用于在并行模式中延迟Tensor的数据的初始化，如果指定该参数，则 `dtype` 和 `shape` 也必须被指定。不推荐在非自动并行之外的场景下使用该接口。只有当调用 `Tensor.init_data` 时，才会使用指定的 `init` 来初始化Tensor数据。默认值：None。
        - **internal** (bool) - Tensor是否由框架创建。如果为True，表示Tensor是由框架创建的，如果为False，表示Tensor是由用户创建的。默认值：False。
        - **const_arg** (bool) - 指定该Tensor作为网络输入时是否为常量。默认值：False。

    输出：
        Tensor。


神经网络层方法
----------------

神经网络
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.flatten

激活函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.hardshrink
    mindspore.Tensor.soft_shrink

数学运算方法
^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.bmm
    mindspore.Tensor.conj
    mindspore.Tensor.cross
    mindspore.Tensor.cumprod
    mindspore.Tensor.div
    mindspore.Tensor.erfinv    
    mindspore.Tensor.equal
    mindspore.Tensor.expm1
    mindspore.Tensor.less_equal

逐元素运算
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.abs
    mindspore.Tensor.add
    mindspore.Tensor.addr
    mindspore.Tensor.addcdiv
    mindspore.Tensor.addcmul
    mindspore.Tensor.asin
    mindspore.Tensor.addmv
    mindspore.Tensor.arcsinh
    mindspore.Tensor.arctanh
    mindspore.Tensor.asinh
    mindspore.Tensor.atan
    mindspore.Tensor.atanh
    mindspore.Tensor.atan2
    mindspore.Tensor.bernoulli
    mindspore.Tensor.bitwise_and
    mindspore.Tensor.bitwise_or
    mindspore.Tensor.bitwise_xor
    mindspore.Tensor.ceil
    mindspore.Tensor.cholesky
    mindspore.Tensor.cholesky_inverse
    mindspore.Tensor.cosh
    mindspore.Tensor.erf
    mindspore.Tensor.erfc
    mindspore.Tensor.exp
    mindspore.Tensor.floor
    mindspore.Tensor.inv
    mindspore.Tensor.invert
    mindspore.Tensor.lerp
    mindspore.Tensor.log
    mindspore.Tensor.log1p
    mindspore.Tensor.logit
    mindspore.Tensor.pow
    mindspore.Tensor.round
    mindspore.Tensor.sigmoid
    mindspore.Tensor.sqrt
    mindspore.Tensor.square
    mindspore.Tensor.std
    mindspore.Tensor.sub
    mindspore.Tensor.svd
    mindspore.Tensor.tan
    mindspore.Tensor.tanh
    mindspore.Tensor.var
    mindspore.Tensor.xdivy
    mindspore.Tensor.xlogy

Reduction方法
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.amax
    mindspore.Tensor.amin
    mindspore.Tensor.argmax
    mindspore.Tensor.argmin
    mindspore.Tensor.argmax_with_value
    mindspore.Tensor.argmin_with_value
    mindspore.Tensor.max
    mindspore.Tensor.mean
    mindspore.Tensor.median
    mindspore.Tensor.min
    mindspore.Tensor.norm
    mindspore.Tensor.prod
    mindspore.Tensor.renorm

比较方法
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.all
    mindspore.Tensor.any
    mindspore.Tensor.approximate_equal
    mindspore.Tensor.ge
    mindspore.Tensor.gt
    mindspore.Tensor.has_init
    mindspore.Tensor.isclose
    mindspore.Tensor.isfinite
    mindspore.Tensor.top_k

线性代数方法
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.ger
    mindspore.Tensor.log_matrix_determinant
    mindspore.Tensor.matrix_determinant
    mindspore.Tensor.det

Tensor操作方法
----------------

Tensor创建
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.choose
    mindspore.Tensor.fill
    mindspore.Tensor.fills
    mindspore.Tensor.view

随机生成方法
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.random_categorical

Array操作
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.broadcast_to
    mindspore.Tensor.col2im
    mindspore.Tensor.copy
    mindspore.Tensor.cummax
    mindspore.Tensor.cummin
    mindspore.Tensor.cumsum
    mindspore.Tensor.diag
    mindspore.Tensor.diagonal
    mindspore.Tensor.dtype
    mindspore.Tensor.expand
    mindspore.Tensor.expand_as
    mindspore.Tensor.expand_dims
    mindspore.Tensor.fold
    mindspore.Tensor.gather
    mindspore.Tensor.gather_elements
    mindspore.Tensor.gather_nd
    mindspore.Tensor.index_fill
    mindspore.Tensor.init_data
    mindspore.Tensor.inplace_update
    mindspore.Tensor.item
    mindspore.Tensor.itemset
    mindspore.Tensor.itemsize
    mindspore.Tensor.masked_fill
    mindspore.Tensor.masked_select
    mindspore.Tensor.minimum
    mindspore.Tensor.nbytes
    mindspore.Tensor.ndim
    mindspore.Tensor.ndimension
    mindspore.Tensor.nonzero
    mindspore.Tensor.narrow
    mindspore.Tensor.ptp
    mindspore.Tensor.ravel
    mindspore.Tensor.repeat
    mindspore.Tensor.reshape
    mindspore.Tensor.resize
    mindspore.Tensor.reverse_sequence
    mindspore.Tensor.reverse
    mindspore.Tensor.scatter_add
    mindspore.Tensor.scatter_div
    mindspore.Tensor.scatter_max
    mindspore.Tensor.scatter_min
    mindspore.Tensor.scatter_mul
    mindspore.Tensor.scatter_sub
    mindspore.Tensor.searchsorted
    mindspore.Tensor.select
    mindspore.Tensor.shape
    mindspore.Tensor.size
    mindspore.Tensor.split
    mindspore.Tensor.squeeze
    mindspore.Tensor.strides
    mindspore.Tensor.sum
    mindspore.Tensor.swapaxes
    mindspore.Tensor.T
    mindspore.Tensor.take
    mindspore.Tensor.tile
    mindspore.Tensor.to_tensor
    mindspore.Tensor.trace
    mindspore.Tensor.transpose
    mindspore.Tensor.unfold
    mindspore.Tensor.unique_consecutive
    mindspore.Tensor.unique_with_pad
    mindspore.Tensor.unsorted_segment_max
    mindspore.Tensor.unsorted_segment_min
    mindspore.Tensor.unsorted_segment_prod

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.asnumpy
    mindspore.Tensor.astype
    mindspore.Tensor.bool
    mindspore.Tensor.float
    mindspore.Tensor.from_numpy
    mindspore.Tensor.half
    mindspore.Tensor.int
    mindspore.Tensor.long
    mindspore.Tensor.to
    mindspore.Tensor.to_coo
    mindspore.Tensor.to_csr

梯度剪裁
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.clip

Parameter操作方法
--------------------

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.assign_value

其他方法
--------------------

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.flush_from_cache
    mindspore.Tensor.set_const_arg
