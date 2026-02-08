mindspore.Tensor
================

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None, const_arg=False, device=None)

    Tensor is a data structure that stores an n-dimensional array.

    .. note::
        - If `init` interface is used to initialize `Tensor`, the `Tensor.init_data` API needs to be called to load the
          actual data to `Tensor`.
        - All modes of CPU and GPU, and Atlas training series with `graph mode (mode=mindspore.GRAPH_MODE)
          <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_  do not supported
          in-place operations yet.
        - The default value ``None`` of `input_data` works as a placeholder,
          it does not mean that we can create a NoneType Tensor.
        - Tensor with `shape` contains 0 is not fully tested and supported.

    .. warning::
        To convert dtype of a `Tensor`, it is recommended to use `Tensor.astype()` rather than
        `Tensor(sourceTensor, dtype=newDtype)`.

    Parameters
        - **input_data** (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]) - The data to be stored. It can be
          another Tensor, Python number or NumPy ndarray. Default: ``None`` .
        - **dtype** (:class:`mindspore.dtype`) - Used to indicate the data type of the output Tensor. The argument should
          be defined in `mindspore.dtype`. If it is ``None`` , the data type of the output Tensor will be the same
          as the `input_data`. Default: ``None`` .
        - **shape** (Union[tuple, list, int, :class:`mindspore.Symbol`]) - Used to indicate the shape of the output Tensor.
          If `input_data` is available, `shape` doesn't need to be set. If ``None`` or `Symbol` exists in `shape` ,
          a tensor of dynamic shape is created, `input_data` doesn't need to be set; if only integers exist in
          `shape`, a tensor of static shape is created, `input_data` or `init` must be set. Default: ``None`` .
        - **init** (Initializer) - The information of init data.
          `init` is used for delayed initialization in parallel mode, when using init, `dtype` and `shape` must be set. Default: ``None`` .
        - **const_arg** (bool) - Whether the tensor is a constant when it is used for the argument of a network. Default: ``False`` .
        - **device** (str) - This parameter is reserved and does not need to be configured. Default: ``None`` .

    Outputs:
        Tensor.

    **Examples**:

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.common.initializer import One
        >>> # initialize a tensor with numpy.ndarray
        >>> t1 = Tensor(np.zeros([1, 2, 3]), ms.float32)
        >>> print(t1)
        [[[0. 0. 0.]
        [0. 0. 0.]]]
        >>> print(type(t1))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t1.shape)
        (1, 2, 3)
        >>> print(t1.dtype)
        Float32
        >>>
        >>> # initialize a tensor with a float scalar
        >>> t2 = Tensor(0.1)
        >>> print(t2)
        0.1
        >>> print(type(t2))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t2.shape)
        ()
        >>> print(t2.dtype)
        Float32
        >>>
        >>> # initialize a tensor with a tuple
        >>> t3 = Tensor((1, 2))
        >>> print(t3)
        [1 2]
        >>> print(type(t3))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t3.shape)
        (2,)
        >>> print(t3.dtype)
        Int64
        ...
        >>> # initialize a tensor with init
        >>> t4 = Tensor(shape = (1, 3), dtype=ms.float32, init=One())
        >>> t4.init_data()
        >>> print(t4)
        [[1. 1. 1.]]
        >>> print(type(t4))
        <class 'mindspore.common.tensor.Tensor'>
        >>> print(t4.shape)
        (1, 3)
        >>> print(t4.dtype)
        Float32

.. autosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.abs
    mindspore.Tensor.absolute
    mindspore.Tensor.acos
    mindspore.Tensor.acosh
    mindspore.Tensor.add_
    mindspore.Tensor.add
    mindspore.Tensor.addbmm
    mindspore.Tensor.addcdiv
    mindspore.Tensor.addcmul
    mindspore.Tensor.addmm
    mindspore.Tensor.addmm_
    mindspore.Tensor.addmv
    mindspore.Tensor.addr
    mindspore.Tensor.adjoint
    mindspore.Tensor.all
    mindspore.Tensor.allclose
    mindspore.Tensor.amax
    mindspore.Tensor.amin
    mindspore.Tensor.aminmax
    mindspore.Tensor.any
    mindspore.Tensor.angle
    mindspore.Tensor.approximate_equal
    mindspore.Tensor.arccos
    mindspore.Tensor.arccosh
    mindspore.Tensor.arcsin
    mindspore.Tensor.arcsinh
    mindspore.Tensor.arctan
    mindspore.Tensor.arctan2
    mindspore.Tensor.arctanh
    mindspore.Tensor.argmax
    mindspore.Tensor.argmax_with_value
    mindspore.Tensor.argmin
    mindspore.Tensor.argmin_with_value
    mindspore.Tensor.argsort
    mindspore.Tensor.argwhere
    mindspore.Tensor.asin
    mindspore.Tensor.asinh
    mindspore.Tensor.asnumpy
    mindspore.Tensor.assign_value
    mindspore.Tensor.astype
    mindspore.Tensor.atan
    mindspore.Tensor.atan2
    mindspore.Tensor.atanh
    mindspore.Tensor.baddbmm
    mindspore.Tensor.bernoulli_
    mindspore.Tensor.bernoulli
    mindspore.Tensor.bfloat16
    mindspore.Tensor.bincount
    mindspore.Tensor.bitwise_and
    mindspore.Tensor.bitwise_not
    mindspore.Tensor.bitwise_left_shift
    mindspore.Tensor.bitwise_or
    mindspore.Tensor.bitwise_right_shift
    mindspore.Tensor.bitwise_xor
    mindspore.Tensor.bmm
    mindspore.Tensor.bool
    mindspore.Tensor.broadcast_to
    mindspore.Tensor.byte
    mindspore.Tensor.cauchy
    mindspore.Tensor.ceil
    mindspore.Tensor.cholesky
    mindspore.Tensor.cholesky_solve
    mindspore.Tensor.choose
    mindspore.Tensor.chunk
    mindspore.Tensor.clamp
    mindspore.Tensor.clamp_
    mindspore.Tensor.clip
    mindspore.Tensor.clone
    mindspore.Tensor.col2im
    mindspore.Tensor.conj
    mindspore.Tensor.contiguous
    mindspore.Tensor.copy
    mindspore.Tensor.copy_
    mindspore.Tensor.copysign
    mindspore.Tensor.cos
    mindspore.Tensor.cosh
    mindspore.Tensor.count_nonzero
    mindspore.Tensor.cov
    mindspore.Tensor.cross
    mindspore.Tensor.cummax
    mindspore.Tensor.cummin
    mindspore.Tensor.cumprod
    mindspore.Tensor.cumsum
    mindspore.Tensor.data
    mindspore.Tensor.deg2rad
    mindspore.Tensor.delete_
    mindspore.Tensor.diag
    mindspore.Tensor.diagflat
    mindspore.Tensor.diagonal
    mindspore.Tensor.diagonal_scatter
    mindspore.Tensor.diff
    mindspore.Tensor.digamma
    mindspore.Tensor.div
    mindspore.Tensor.div_
    mindspore.Tensor.divide
    mindspore.Tensor.dot
    mindspore.Tensor.double
    mindspore.Tensor.dsplit
    mindspore.Tensor.dtype
    mindspore.Tensor.eigvals
    mindspore.Tensor.eq
    mindspore.Tensor.equal
    mindspore.Tensor.erf
    mindspore.Tensor.erfc
    mindspore.Tensor.erfinv
    mindspore.Tensor.exp
    mindspore.Tensor.exp_
    mindspore.Tensor.expand
    mindspore.Tensor.expand_as
    mindspore.Tensor.expand_dims
    mindspore.Tensor.expm1
    mindspore.Tensor.exponential_
    mindspore.Tensor.fill_
    mindspore.Tensor.fill_diagonal
    mindspore.Tensor.flatten
    mindspore.Tensor.flip
    mindspore.Tensor.fliplr
    mindspore.Tensor.flipud
    mindspore.Tensor.float
    mindspore.Tensor.float_power
    mindspore.Tensor.floor
    mindspore.Tensor.floor_
    mindspore.Tensor.floor_divide
    mindspore.Tensor.floor_divide_
    mindspore.Tensor.flush_from_cache
    mindspore.Tensor.fmax
    mindspore.Tensor.fmod
    mindspore.Tensor.fold
    mindspore.Tensor.frac
    mindspore.Tensor.from_numpy
    mindspore.Tensor.gather
    mindspore.Tensor.gather_elements
    mindspore.Tensor.gather_nd
    mindspore.Tensor.gcd
    mindspore.Tensor.ge
    mindspore.Tensor.geqrf
    mindspore.Tensor.ger
    mindspore.Tensor.greater
    mindspore.Tensor.greater_equal
    mindspore.Tensor.gt
    mindspore.Tensor.H
    mindspore.Tensor.half
    mindspore.Tensor.hardshrink
    mindspore.Tensor.has_init
    mindspore.Tensor.heaviside
    mindspore.Tensor.histc
    mindspore.Tensor.hsplit
    mindspore.Tensor.hypot
    mindspore.Tensor.i0
    mindspore.Tensor.igamma
    mindspore.Tensor.igammac
    mindspore.Tensor.imag
    mindspore.Tensor.index_add
    mindspore.Tensor.index_add_
    mindspore.Tensor.index_copy_
    mindspore.Tensor.index_fill
    mindspore.Tensor.index_fill_
    mindspore.Tensor.index_put
    mindspore.Tensor.index_put_
    mindspore.Tensor.index_select
    mindspore.Tensor.init_data
    mindspore.Tensor.inner
    mindspore.Tensor.inplace_update
    mindspore.Tensor.int
    mindspore.Tensor.inv
    mindspore.Tensor.inverse
    mindspore.Tensor.invert
    mindspore.Tensor.isclose
    mindspore.Tensor.isfinite
    mindspore.Tensor.is_complex
    mindspore.Tensor.is_contiguous
    mindspore.Tensor.is_floating_point
    mindspore.Tensor.isinf
    mindspore.Tensor.isnan
    mindspore.Tensor.isneginf
    mindspore.Tensor.isposinf
    mindspore.Tensor.isreal
    mindspore.Tensor.is_pinned
    mindspore.Tensor.is_shared
    mindspore.Tensor.is_signed
    mindspore.Tensor.item
    mindspore.Tensor.itemset
    mindspore.Tensor.itemsize
    mindspore.Tensor.lcm
    mindspore.Tensor.ldexp
    mindspore.Tensor.le
    mindspore.Tensor.lerp
    mindspore.Tensor.less
    mindspore.Tensor.less_equal
    mindspore.Tensor.log
    mindspore.Tensor.log10
    mindspore.Tensor.log1p
    mindspore.Tensor.log2
    mindspore.Tensor.logaddexp
    mindspore.Tensor.logaddexp2
    mindspore.Tensor.logcumsumexp
    mindspore.Tensor.logdet
    mindspore.Tensor.logical_and
    mindspore.Tensor.logical_not
    mindspore.Tensor.logical_or
    mindspore.Tensor.logical_xor
    mindspore.Tensor.logit
    mindspore.Tensor.logsumexp
    mindspore.Tensor.log_normal
    mindspore.Tensor.long
    mindspore.Tensor.lt
    mindspore.Tensor.lu_solve
    mindspore.Tensor.masked_fill
    mindspore.Tensor.masked_fill_
    mindspore.Tensor.masked_scatter
    mindspore.Tensor.masked_scatter_
    mindspore.Tensor.masked_select
    mindspore.Tensor.matmul
    mindspore.Tensor.max
    mindspore.Tensor.maximum
    mindspore.Tensor.mean
    mindspore.Tensor.median
    mindspore.Tensor.mH
    mindspore.Tensor.min
    mindspore.Tensor.minimum
    mindspore.Tensor.mm
    mindspore.Tensor.moveaxis
    mindspore.Tensor.movedim
    mindspore.Tensor.move_to
    mindspore.Tensor.msort
    mindspore.Tensor.mT
    mindspore.Tensor.mul
    mindspore.Tensor.mul_
    mindspore.Tensor.multinomial
    mindspore.Tensor.multiply
    mindspore.Tensor.mvlgamma
    mindspore.Tensor.nan_to_num
    mindspore.Tensor.nanmean
    mindspore.Tensor.nanmedian
    mindspore.Tensor.nansum
    mindspore.Tensor.narrow
    mindspore.Tensor.nbytes
    mindspore.Tensor.ndim
    mindspore.Tensor.ndimension
    mindspore.Tensor.ne
    mindspore.Tensor.neg
    mindspore.Tensor.negative
    mindspore.Tensor.nelement
    mindspore.Tensor.new_empty
    mindspore.Tensor.new_ones
    mindspore.Tensor.new_zeros
    mindspore.Tensor.new_full
    mindspore.Tensor.nextafter
    mindspore.Tensor.nonzero
    mindspore.Tensor.norm
    mindspore.Tensor.normal_
    mindspore.Tensor.not_equal
    mindspore.Tensor.numel
    mindspore.Tensor.numpy
    mindspore.Tensor.orgqr
    mindspore.Tensor.ormqr
    mindspore.Tensor.outer
    mindspore.Tensor.permute
    mindspore.Tensor.pin_memory
    mindspore.Tensor.positive
    mindspore.Tensor.pow
    mindspore.Tensor.prod
    mindspore.Tensor.ptp
    mindspore.Tensor.rad2deg
    mindspore.Tensor.random_
    mindspore.Tensor.random_categorical
    mindspore.Tensor.ravel
    mindspore.Tensor.real
    mindspore.Tensor.reciprocal
    mindspore.Tensor.register_hook
    mindspore.Tensor.remainder
    mindspore.Tensor.remainder_
    mindspore.Tensor.renorm
    mindspore.Tensor.repeat
    mindspore.Tensor.repeat_interleave
    mindspore.Tensor.reshape
    mindspore.Tensor.reshape_as
    mindspore.Tensor.resize
    mindspore.Tensor.reverse
    mindspore.Tensor.reverse_sequence
    mindspore.Tensor.roll
    mindspore.Tensor.rot90
    mindspore.Tensor.round
    mindspore.Tensor.rsqrt
    mindspore.Tensor.scatter
    mindspore.Tensor.scatter_
    mindspore.Tensor.scatter_add
    mindspore.Tensor.scatter_add_
    mindspore.Tensor.scatter_div
    mindspore.Tensor.scatter_max
    mindspore.Tensor.scatter_min
    mindspore.Tensor.scatter_mul
    mindspore.Tensor.scatter_sub
    mindspore.Tensor.searchsorted
    mindspore.Tensor.select
    mindspore.Tensor.select_scatter
    mindspore.Tensor.set_
    mindspore.Tensor.set_const_arg
    mindspore.Tensor.sgn
    mindspore.Tensor.shape
    mindspore.Tensor.short
    mindspore.Tensor.sigmoid
    mindspore.Tensor.sigmoid_
    mindspore.Tensor.sign
    mindspore.Tensor.sign_
    mindspore.Tensor.signbit
    mindspore.Tensor.sin
    mindspore.Tensor.sinc
    mindspore.Tensor.sinh
    mindspore.Tensor.size
    mindspore.Tensor.slice_scatter
    mindspore.Tensor.slogdet
    mindspore.Tensor.softmax
    mindspore.Tensor.sort
    mindspore.Tensor.split
    mindspore.Tensor.sqrt
    mindspore.Tensor.square
    mindspore.Tensor.squeeze
    mindspore.Tensor.std
    mindspore.Tensor.storage
    mindspore.Tensor.storage_offset
    mindspore.Tensor.stride
    mindspore.Tensor.strides
    mindspore.Tensor.sub
    mindspore.Tensor.sub_
    mindspore.Tensor.subtract
    mindspore.Tensor.sum
    mindspore.Tensor.sum_to_size
    mindspore.Tensor.svd
    mindspore.Tensor.swapaxes
    mindspore.Tensor.swapdims
    mindspore.Tensor.T
    mindspore.Tensor.t
    mindspore.Tensor.take
    mindspore.Tensor.tan
    mindspore.Tensor.tanh
    mindspore.Tensor.tensor_split
    mindspore.Tensor.tile
    mindspore.Tensor.to
    mindspore.Tensor.to_
    mindspore.Tensor.to_coo
    mindspore.Tensor.to_csr
    mindspore.Tensor.tolist
    mindspore.Tensor.topk
    mindspore.Tensor.trace
    mindspore.Tensor.transpose
    mindspore.Tensor.triangular_solve
    mindspore.Tensor.tril
    mindspore.Tensor.triu
    mindspore.Tensor.true_divide
    mindspore.Tensor.trunc
    mindspore.Tensor.type
    mindspore.Tensor.type_as
    mindspore.Tensor.unbind
    mindspore.Tensor.unfold
    mindspore.Tensor.uniform
    mindspore.Tensor.uniform_
    mindspore.Tensor.unique
    mindspore.Tensor.unique_consecutive
    mindspore.Tensor.unsorted_segment_max
    mindspore.Tensor.unsorted_segment_min
    mindspore.Tensor.unsorted_segment_prod
    mindspore.Tensor.unsqueeze
    mindspore.Tensor.untyped_storage
    mindspore.Tensor.var
    mindspore.Tensor.view
    mindspore.Tensor.view_as
    mindspore.Tensor.vsplit
    mindspore.Tensor.where
    mindspore.Tensor.xdivy
    mindspore.Tensor.xlogy
    mindspore.Tensor.zero_