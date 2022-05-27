mindspore.Tensor
================

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None, internal=False)

    张量，即存储多维数组（n-dimensional array）的数据结构。

    参数：
        - **input_data** (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]) - 被存储的数据，可以是其它Tensor，也可以是Python基本数据（如int，float，bool等），或是一个NumPy对象。默认值：None。
        - **dtype** (:class:`mindspore.dtype`) - 用于定义该Tensor的数据类型，必须是 *mindspore.dtype* 中定义的类型。如果该参数为None，则数据类型与 `input_data` 一致，默认值：None。
        - **shape** (Union[tuple, list, int]) - 用于定义该Tensor的形状。如果指定了 `input_data` ，则无需设置该参数。默认值：None。
        - **init** (Initializer) - 用于在并行模式中延迟Tensor的数据的初始化，如果指定该参数，则 `dtype` 和 `shape` 也必须被指定。不推荐在非自动并行之外的场景下使用该接口。只有当调用 `Tensor.init_data` 时，才会使用指定的 `init` 来初始化Tensor数据。默认值：None。
        - **internal** (bool) - Tensor是否由框架创建。 如果为True，表示Tensor是由框架创建的，如果为False，表示Tensor是由用户创建的。默认值：False。

    输出：
        Tensor。

.. mscnplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.abs
    mindspore.Tensor.addcdiv
    mindspore.Tensor.addcmul
    mindspore.Tensor.all
    mindspore.Tensor.any
    mindspore.Tensor.approximate_equal
    mindspore.Tensor.argmax
    mindspore.Tensor.argmin
    mindspore.Tensor.argmin_with_value
    mindspore.Tensor.asnumpy
    mindspore.Tensor.assign_value
    mindspore.Tensor.astype
    mindspore.Tensor.atan2
    mindspore.Tensor.bernoulli
    mindspore.Tensor.bitwise_and
    mindspore.Tensor.bitwise_or
    mindspore.Tensor.bitwise_xor
    mindspore.Tensor.broadcast_to
    mindspore.Tensor.ceil
    mindspore.Tensor.choose
    mindspore.Tensor.clip
    mindspore.Tensor.col2im
    mindspore.Tensor.copy
    mindspore.Tensor.cosh
    mindspore.Tensor.cummax
    mindspore.Tensor.cummin
    mindspore.Tensor.cumsum
    mindspore.Tensor.diag
    mindspore.Tensor.diagonal
    mindspore.Tensor.dtype
    mindspore.Tensor.erf
    mindspore.Tensor.erfc
    mindspore.Tensor.expand_as
    mindspore.Tensor.expand_dims
    mindspore.Tensor.fill
    mindspore.Tensor.fills
    mindspore.Tensor.flatten
    mindspore.Tensor.flush_from_cache
    mindspore.Tensor.from_numpy
    mindspore.Tensor.gather
    mindspore.Tensor.gather_elements
    mindspore.Tensor.gather_nd
    mindspore.Tensor.ger
    mindspore.Tensor.hardshrink
    mindspore.Tensor.has_init
    mindspore.Tensor.index_fill
    mindspore.Tensor.init_data
    mindspore.Tensor.inplace_update
    mindspore.Tensor.inv
    mindspore.Tensor.invert
    mindspore.Tensor.isclose
    mindspore.Tensor.isfinite
    mindspore.Tensor.item
    mindspore.Tensor.itemset
    mindspore.Tensor.itemsize
    mindspore.Tensor.lerp
    mindspore.Tensor.log1p
    mindspore.Tensor.log_matrix_determinant
    mindspore.Tensor.logit
    mindspore.Tensor.masked_fill
    mindspore.Tensor.masked_select
    mindspore.Tensor.matrix_determinant
    mindspore.Tensor.max
    mindspore.Tensor.mean
    mindspore.Tensor.median
    mindspore.Tensor.min
    mindspore.Tensor.narrow
    mindspore.Tensor.nbytes
    mindspore.Tensor.ndim
    mindspore.Tensor.nonzero
    mindspore.Tensor.norm
    mindspore.Tensor.pow
    mindspore.Tensor.prod
    mindspore.Tensor.ptp
    mindspore.Tensor.random_categorical
    mindspore.Tensor.ravel
    mindspore.Tensor.renorm
    mindspore.Tensor.repeat
    mindspore.Tensor.reshape
    mindspore.Tensor.resize
    mindspore.Tensor.round
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
    mindspore.Tensor.soft_shrink
    mindspore.Tensor.split
    mindspore.Tensor.squeeze
    mindspore.Tensor.std
    mindspore.Tensor.strides
    mindspore.Tensor.sum
    mindspore.Tensor.svd
    mindspore.Tensor.swapaxes
    mindspore.Tensor.T
    mindspore.Tensor.take
    mindspore.Tensor.tan
    mindspore.Tensor.top_k
    mindspore.Tensor.to_coo
    mindspore.Tensor.to_csr
    mindspore.Tensor.to_tensor
    mindspore.Tensor.trace
    mindspore.Tensor.transpose
    mindspore.Tensor.unique_consecutive
    mindspore.Tensor.unique_with_pad
    mindspore.Tensor.unsorted_segment_max
    mindspore.Tensor.unsorted_segment_min
    mindspore.Tensor.unsorted_segment_prod
    mindspore.Tensor.var
    mindspore.Tensor.view
    mindspore.Tensor.xdivy
    mindspore.Tensor.xlogy
