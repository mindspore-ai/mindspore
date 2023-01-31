mindspore.ops
=============================

神经网络层函数
----------------

神经网络
^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.adaptive_avg_pool1d
    mindspore.ops.adaptive_avg_pool2d
    mindspore.ops.adaptive_avg_pool3d
    mindspore.ops.adaptive_max_pool1d
    mindspore.ops.adaptive_max_pool2d
    mindspore.ops.adaptive_max_pool3d
    mindspore.ops.avg_pool1d
    mindspore.ops.avg_pool2d
    mindspore.ops.avg_pool3d
    mindspore.ops.batch_norm
    mindspore.ops.bias_add
    mindspore.ops.ctc_greedy_decoder
    mindspore.ops.conv2d
    mindspore.ops.conv3d
    mindspore.ops.deformable_conv2d
    mindspore.ops.dropout
    mindspore.ops.dropout1d
    mindspore.ops.dropout2d
    mindspore.ops.dropout3d
    mindspore.ops.flatten
    mindspore.ops.unfold
    mindspore.ops.fold
    mindspore.ops.fractional_max_pool2d
    mindspore.ops.fractional_max_pool3d
    mindspore.ops.lp_pool1d
    mindspore.ops.lp_pool2d
    mindspore.ops.lrn
    mindspore.ops.max_pool3d
    mindspore.ops.max_unpool1d
    mindspore.ops.max_unpool2d
    mindspore.ops.max_unpool3d


损失函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.binary_cross_entropy
    mindspore.ops.binary_cross_entropy_with_logits
    mindspore.ops.cosine_embedding_loss
    mindspore.ops.cross_entropy
    mindspore.ops.gaussian_nll_loss
    mindspore.ops.hinge_embedding_loss
    mindspore.ops.huber_loss
    mindspore.ops.kl_div
    mindspore.ops.l1_loss
    mindspore.ops.margin_ranking_loss
    mindspore.ops.mse_loss
    mindspore.ops.multi_margin_loss
    mindspore.ops.multi_label_margin_loss
    mindspore.ops.multilabel_soft_margin_loss
    mindspore.ops.nll_loss
    mindspore.ops.smooth_l1_loss
    mindspore.ops.triplet_margin_loss

激活函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.celu
    mindspore.ops.elu
    mindspore.ops.fast_gelu
    mindspore.ops.gelu
    mindspore.ops.glu
    mindspore.ops.gumbel_softmax
    mindspore.ops.hardshrink
    mindspore.ops.hardswish
    mindspore.ops.hardtanh
    mindspore.ops.leaky_relu
    mindspore.ops.log_softmax
    mindspore.ops.logsigmoid
    mindspore.ops.mish
    mindspore.ops.prelu
    mindspore.ops.relu
    mindspore.ops.relu6
    mindspore.ops.rrelu
    mindspore.ops.selu
    mindspore.ops.sigmoid
    mindspore.ops.silu
    mindspore.ops.soft_shrink
    mindspore.ops.softmax
    mindspore.ops.softmin
    mindspore.ops.softsign
    mindspore.ops.tanh
    mindspore.ops.threshold

距离函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.cdist
    mindspore.ops.dist
    mindspore.ops.pdist

采样函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.choice_with_mask
    mindspore.ops.random_categorical
    mindspore.ops.log_uniform_candidate_sampler
    mindspore.ops.uniform_candidate_sampler

图像函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.affine_grid
    mindspore.ops.bounding_box_decode
    mindspore.ops.bounding_box_encode
    mindspore.ops.col2im
    mindspore.ops.check_valid
    mindspore.ops.crop_and_resize
    mindspore.ops.grid_sample
    mindspore.ops.interpolate
    mindspore.ops.iou
    mindspore.ops.pad
    mindspore.ops.padding
    mindspore.ops.pixel_shuffle
    mindspore.ops.pixel_unshuffle

数学运算函数
----------------

逐元素运算
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.abs
    mindspore.ops.absolute
    mindspore.ops.accumulate_n
    mindspore.ops.acos
    mindspore.ops.arccos
    mindspore.ops.acosh
    mindspore.ops.add
    mindspore.ops.addcdiv
    mindspore.ops.addcmul
    mindspore.ops.addmv
    mindspore.ops.addn
    mindspore.ops.angle
    mindspore.ops.arccosh
    mindspore.ops.arcsin
    mindspore.ops.arcsinh
    mindspore.ops.arctan
    mindspore.ops.arctanh
    mindspore.ops.arctan2
    mindspore.ops.asin
    mindspore.ops.asinh
    mindspore.ops.atan
    mindspore.ops.atan2
    mindspore.ops.atanh
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
    mindspore.ops.bitwise_left_shift
    mindspore.ops.bitwise_or
    mindspore.ops.bitwise_right_shift
    mindspore.ops.bitwise_xor
    mindspore.ops.ceil
    mindspore.ops.clamp
    mindspore.ops.clip
    mindspore.ops.copysign
    mindspore.ops.cos
    mindspore.ops.cosh
    mindspore.ops.cosine_similarity
    mindspore.ops.cov
    mindspore.ops.diff
    mindspore.ops.deg2rad
    mindspore.ops.digamma
    mindspore.ops.div
    mindspore.ops.divide
    mindspore.ops.erf
    mindspore.ops.erfc
    mindspore.ops.erfinv
    mindspore.ops.exp
    mindspore.ops.exp2
    mindspore.ops.expm1
    mindspore.ops.floor
    mindspore.ops.floor_div
    mindspore.ops.floor_mod
    mindspore.ops.float_power
    mindspore.ops.fmod
    mindspore.ops.frac
    mindspore.ops.hypot
    mindspore.ops.igamma
    mindspore.ops.igammac
    mindspore.ops.i0
    mindspore.ops.inv
    mindspore.ops.invert
    mindspore.ops.lcm
    mindspore.ops.ldexp
    mindspore.ops.lerp
    mindspore.ops.lgamma
    mindspore.ops.log
    mindspore.ops.log2
    mindspore.ops.log10
    mindspore.ops.log1p
    mindspore.ops.logaddexp
    mindspore.ops.logaddexp2
    mindspore.ops.logical_and
    mindspore.ops.logical_not
    mindspore.ops.logical_or
    mindspore.ops.logical_xor
    mindspore.ops.logit
    mindspore.ops.log_matrix_determinant
    mindspore.ops.matrix_determinant
    mindspore.ops.mul
    mindspore.ops.multiply
    mindspore.ops.mvlgamma
    mindspore.ops.neg
    mindspore.ops.negative
    mindspore.ops.positive
    mindspore.ops.pow
    mindspore.ops.rad2deg
    mindspore.ops.ravel
    mindspore.ops.real
    mindspore.ops.reciprocal
    mindspore.ops.remainder
    mindspore.ops.roll
    mindspore.ops.rot90
    mindspore.ops.round
    mindspore.ops.rsqrt
    mindspore.ops.sgn
    mindspore.ops.sign
    mindspore.ops.signbit
    mindspore.ops.sin
    mindspore.ops.sinc
    mindspore.ops.sinh
    mindspore.ops.sqrt
    mindspore.ops.square
    mindspore.ops.sub
    mindspore.ops.subtract
    mindspore.ops.t
    mindspore.ops.tan
    mindspore.ops.tanhshrink
    mindspore.ops.trapz
    mindspore.ops.true_divide
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
    mindspore.ops.argmax
    mindspore.ops.argmin
    mindspore.ops.cummax
    mindspore.ops.cummin
    mindspore.ops.cumsum
    mindspore.ops.logsumexp
    mindspore.ops.max
    mindspore.ops.mean
    mindspore.ops.median
    mindspore.ops.min
    mindspore.ops.norm
    mindspore.ops.prod
    mindspore.ops.std

比较函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.argsort
    mindspore.ops.approximate_equal
    mindspore.ops.equal
    mindspore.ops.ge
    mindspore.ops.greater
    mindspore.ops.greater_equal
    mindspore.ops.gt
    mindspore.ops.intopk
    mindspore.ops.isclose
    mindspore.ops.isfinite
    mindspore.ops.isinf
    mindspore.ops.isnan
    mindspore.ops.isneginf
    mindspore.ops.isposinf
    mindspore.ops.isreal
    mindspore.ops.is_complex
    mindspore.ops.is_floating_point
    mindspore.ops.le
    mindspore.ops.less
    mindspore.ops.less_equal
    mindspore.ops.lt
    mindspore.ops.maximum
    mindspore.ops.minimum
    mindspore.ops.ne
    mindspore.ops.not_equal
    mindspore.ops.msort
    mindspore.ops.topk

线性代数函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bmm
    mindspore.ops.addbmm
    mindspore.ops.addmm
    mindspore.ops.baddbmm
    mindspore.ops.addr
    mindspore.ops.adjoint
    mindspore.ops.cholesky
    mindspore.ops.cholesky_inverse
    mindspore.ops.batch_dot
    mindspore.ops.dot
    mindspore.ops.inner
    mindspore.ops.inverse
    mindspore.ops.ger
    mindspore.ops.kron
    mindspore.ops.matmul
    mindspore.ops.matrix_solve
    mindspore.ops.matrix_exp
    mindspore.ops.matrix_band_part
    mindspore.ops.matrix_diag
    mindspore.ops.matrix_diag_part
    mindspore.ops.matrix_set_diag
    mindspore.ops.mm
    mindspore.ops.mv
    mindspore.ops.outer
    mindspore.ops.orgqr
    mindspore.ops.pinv
    mindspore.ops.svd
    mindspore.ops.tensor_dot
    mindspore.ops.logdet
    mindspore.ops.slogdet

谱函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bartlett_window
    mindspore.ops.blackman_window
    mindspore.ops.hamming_window
    mindspore.ops.hann_window
    mindspore.ops.kaiser_window

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
    mindspore.ops.full
    mindspore.ops.full_like
    mindspore.ops.linspace
    mindspore.ops.one_hot
    mindspore.ops.ones
    mindspore.ops.ones_like
    mindspore.ops.arange
    mindspore.ops.range
    mindspore.ops.zeros
    mindspore.ops.zeros_like
    mindspore.ops.heaviside

随机生成函数
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bernoulli
    mindspore.ops.gamma
    mindspore.ops.laplace
    mindspore.ops.multinomial
    mindspore.ops.rand
    mindspore.ops.rand_like
    mindspore.ops.randint
    mindspore.ops.randint_like
    mindspore.ops.randn
    mindspore.ops.randn_like
    mindspore.ops.random_poisson
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
    mindspore.ops.bincount
    mindspore.ops.block_diag
    mindspore.ops.broadcast_to
    mindspore.ops.cartesian_prod
    mindspore.ops.cat
    mindspore.ops.chunk
    mindspore.ops.concat
    mindspore.ops.conj
    mindspore.ops.count_nonzero
    mindspore.ops.diag
    mindspore.ops.diagflat
    mindspore.ops.diagonal
    mindspore.ops.dyn_shape
    mindspore.ops.dsplit
    mindspore.ops.einsum
    mindspore.ops.expand
    mindspore.ops.expand_dims
    mindspore.ops.flip
    mindspore.ops.fliplr
    mindspore.ops.flipud
    mindspore.ops.gather
    mindspore.ops.gather_d
    mindspore.ops.gather_elements
    mindspore.ops.gather_nd
    mindspore.ops.hsplit
    mindspore.ops.index_add
    mindspore.ops.index_fill
    mindspore.ops.index_select
    mindspore.ops.inplace_add
    mindspore.ops.inplace_sub
    mindspore.ops.inplace_update
    mindspore.ops.masked_fill
    mindspore.ops.masked_select
    mindspore.ops.meshgrid
    mindspore.ops.narrow
    mindspore.ops.nan_to_num
    mindspore.ops.normal
    mindspore.ops.nonzero
    mindspore.ops.numel
    mindspore.ops.permute
    mindspore.ops.population_count
    mindspore.ops.rank
    mindspore.ops.repeat_elements
    mindspore.ops.repeat_interleave
    mindspore.ops.reshape
    mindspore.ops.reverse
    mindspore.ops.reverse_sequence
    mindspore.ops.scatter
    mindspore.ops.scatter_nd
    mindspore.ops.select
    mindspore.ops.sequence_mask
    mindspore.ops.shape
    mindspore.ops.shuffle
    mindspore.ops.size
    mindspore.ops.slice
    mindspore.ops.space_to_batch_nd
    mindspore.ops.sparse_segment_mean
    mindspore.ops.split
    mindspore.ops.squeeze
    mindspore.ops.stack
    mindspore.ops.strided_slice
    mindspore.ops.sum
    mindspore.ops.swapaxes
    mindspore.ops.swapdims
    mindspore.ops.tensor_scatter_add
    mindspore.ops.tensor_scatter_div
    mindspore.ops.tensor_scatter_max
    mindspore.ops.tensor_scatter_min
    mindspore.ops.tensor_scatter_mul
    mindspore.ops.tensor_scatter_sub
    mindspore.ops.tensor_scatter_elements
    mindspore.ops.tensor_split
    mindspore.ops.tile
    mindspore.ops.tril
    mindspore.ops.transpose
    mindspore.ops.unbind
    mindspore.ops.unique
    mindspore.ops.unique_consecutive
    mindspore.ops.unique_with_pad
    mindspore.ops.unsorted_segment_max
    mindspore.ops.unsorted_segment_min
    mindspore.ops.unsorted_segment_prod
    mindspore.ops.unsorted_segment_sum
    mindspore.ops.unsqueeze
    mindspore.ops.unstack
    mindspore.ops.view_as_real
    mindspore.ops.vsplit
    mindspore.ops.where
    mindspore.ops.cross
    mindspore.ops.cumprod
    mindspore.ops.renorm

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.is_tensor
    mindspore.ops.scalar_cast
    mindspore.ops.scalar_to_tensor
    mindspore.ops.tuple_to_array

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
    mindspore.ops.jet
    mindspore.ops.stop_gradient

调试函数
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.print_

稀疏函数
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.dense_to_sparse_coo
    mindspore.ops.dense_to_sparse_csr
    mindspore.ops.csr_to_coo

COO函数
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.coo_abs
    mindspore.ops.coo_acos
    mindspore.ops.coo_acosh
    mindspore.ops.coo_add
    mindspore.ops.coo_asin
    mindspore.ops.coo_asinh
    mindspore.ops.coo_atan
    mindspore.ops.coo_atanh
    mindspore.ops.coo_ceil
    mindspore.ops.coo_concat
    mindspore.ops.coo_cos
    mindspore.ops.coo_cosh
    mindspore.ops.coo_exp
    mindspore.ops.coo_expm1
    mindspore.ops.coo_floor
    mindspore.ops.coo_inv
    mindspore.ops.coo_isfinite
    mindspore.ops.coo_isinf
    mindspore.ops.coo_isnan
    mindspore.ops.coo_log
    mindspore.ops.coo_log1p
    mindspore.ops.coo_neg
    mindspore.ops.coo_relu
    mindspore.ops.coo_relu6
    mindspore.ops.coo_round
    mindspore.ops.coo_sigmoid
    mindspore.ops.coo_sin
    mindspore.ops.coo_sinh
    mindspore.ops.coo_softsign
    mindspore.ops.coo_sqrt
    mindspore.ops.coo_square
    mindspore.ops.coo_tan
    mindspore.ops.coo_tanh

CSR函数
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.csr_abs
    mindspore.ops.csr_acos
    mindspore.ops.csr_acosh
    mindspore.ops.csr_add
    mindspore.ops.csr_asin
    mindspore.ops.csr_asinh
    mindspore.ops.csr_atan
    mindspore.ops.csr_atanh
    mindspore.ops.csr_ceil
    mindspore.ops.csr_cos
    mindspore.ops.csr_cosh
    mindspore.ops.csr_exp
    mindspore.ops.csr_expm1
    mindspore.ops.csr_floor
    mindspore.ops.csr_inv
    mindspore.ops.csr_isfinite
    mindspore.ops.csr_isinf
    mindspore.ops.csr_isnan
    mindspore.ops.csr_log
    mindspore.ops.csr_log1p
    mindspore.ops.csr_mm
    mindspore.ops.csr_neg
    mindspore.ops.csr_relu
    mindspore.ops.csr_relu6
    mindspore.ops.csr_round
    mindspore.ops.csr_sigmoid
    mindspore.ops.csr_sin
    mindspore.ops.csr_sinh
    mindspore.ops.csr_softmax
    mindspore.ops.csr_softsign
    mindspore.ops.csr_sqrt
    mindspore.ops.csr_square
    mindspore.ops.csr_tan
    mindspore.ops.csr_tanh
