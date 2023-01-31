mindspore.ops
================

神经网络层函数
----------------

神经网络
^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.adaptive_avg_pool2d
    mindspore.ops.adaptive_max_pool3d
    mindspore.ops.avg_pool3d
    mindspore.ops.batch_norm
    mindspore.ops.bias_add
    mindspore.ops.conv2d
    mindspore.ops.conv3d
    mindspore.ops.ctc_greedy_decoder
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
    mindspore.ops.gaussian_nll_loss
    mindspore.ops.hinge_embedding_loss
    mindspore.ops.kl_div
    mindspore.ops.margin_ranking_loss
    mindspore.ops.multi_label_margin_loss
    mindspore.ops.nll_loss
    mindspore.ops.smooth_l1_loss

激活函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.elu
    mindspore.ops.fast_gelu
    mindspore.ops.gelu
    mindspore.ops.glu
    mindspore.ops.gumbel_softmax
    mindspore.ops.hardshrink
    mindspore.ops.hardswish
    mindspore.ops.log_softmax
    mindspore.ops.mish
    mindspore.ops.prelu
    mindspore.ops.relu
    mindspore.ops.relu6
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

    mindspore.ops.choice_with_mask
    mindspore.ops.random_categorical
    mindspore.ops.log_uniform_candidate_sampler
    mindspore.ops.uniform_candidate_sampler

距离函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.cdist
    mindspore.ops.pdist

逐元素运算
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.abs
    mindspore.ops.absolute
    mindspore.ops.acos
    mindspore.ops.acosh
    mindspore.ops.add
    mindspore.ops.addcdiv
    mindspore.ops.addcmul
    mindspore.ops.addn
    mindspore.ops.angle
    mindspore.ops.arccos
    mindspore.ops.arccosh
    mindspore.ops.arcsin
    mindspore.ops.arctan
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
    mindspore.ops.bitwise_or
    mindspore.ops.bitwise_xor
    mindspore.ops.ceil
    mindspore.ops.cos
    mindspore.ops.cosh
    mindspore.ops.deg2rad
    mindspore.ops.div
    mindspore.ops.divide
    mindspore.ops.erf
    mindspore.ops.erfc
    mindspore.ops.erfinv
    mindspore.ops.exp
    mindspore.ops.expm1
    mindspore.ops.floor
    mindspore.ops.floor_div
    mindspore.ops.floor_mod
    mindspore.ops.i0
    mindspore.ops.inv
    mindspore.ops.invert
    mindspore.ops.lcm
    mindspore.ops.lerp
    mindspore.ops.log
    mindspore.ops.log2
    mindspore.ops.log10
    mindspore.ops.log1p
    mindspore.ops.logical_and
    mindspore.ops.logical_not
    mindspore.ops.logical_or
    mindspore.ops.logit
    mindspore.ops.mul
    mindspore.ops.multiply
    mindspore.ops.neg
    mindspore.ops.negative
    mindspore.ops.positive
    mindspore.ops.pow
    mindspore.ops.rad2deg
    mindspore.ops.roll
    mindspore.ops.round
    mindspore.ops.sin
    mindspore.ops.sinh
    mindspore.ops.sqrt
    mindspore.ops.square
    mindspore.ops.sub
    mindspore.ops.subtract
    mindspore.ops.tan
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
    mindspore.ops.cumsum
    mindspore.ops.logsumexp
    mindspore.ops.max
    mindspore.ops.median
    mindspore.ops.prod
    mindspore.ops.std

比较函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.approximate_equal
    mindspore.ops.equal
    mindspore.ops.ge
    mindspore.ops.greater
    mindspore.ops.greater_equal
    mindspore.ops.gt
    mindspore.ops.intopk
    mindspore.ops.isfinite
    mindspore.ops.isinf
    mindspore.ops.isnan
    mindspore.ops.is_floating_point
    mindspore.ops.le
    mindspore.ops.less
    mindspore.ops.less_equal
    mindspore.ops.maximum
    mindspore.ops.ne
    mindspore.ops.top_k

线性代数函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.addbmm
    mindspore.ops.addmm
    mindspore.ops.baddbmm
    mindspore.ops.adjoint
    mindspore.ops.batch_dot
    mindspore.ops.dot
    mindspore.ops.ger
    mindspore.ops.matmul
    mindspore.ops.matrix_exp
    mindspore.ops.matrix_diag
    mindspore.ops.pinv
    mindspore.ops.tensor_dot

谱函数
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bartlett_window
    mindspore.ops.blackman_window

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
    mindspore.ops.one_hot
    mindspore.ops.ones
    mindspore.ops.ones_like
    mindspore.ops.arange
    mindspore.ops.range

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
    mindspore.ops.concat
    mindspore.ops.conj
    mindspore.ops.count_nonzero
    mindspore.ops.expand
    mindspore.ops.expand_dims
    mindspore.ops.flip
    mindspore.ops.fliplr
    mindspore.ops.flipud
    mindspore.ops.gather
    mindspore.ops.gather_d
    mindspore.ops.gather_elements
    mindspore.ops.gather_nd
    mindspore.ops.index_add
    mindspore.ops.inplace_add
    mindspore.ops.inplace_sub
    mindspore.ops.inplace_update
    mindspore.ops.masked_fill
    mindspore.ops.masked_select
    mindspore.ops.meshgrid
    mindspore.ops.narrow
    mindspore.ops.numel
    mindspore.ops.permute
    mindspore.ops.population_count
    mindspore.ops.rank
    mindspore.ops.reshape
    mindspore.ops.reverse
    mindspore.ops.reverse_sequence
    mindspore.ops.scatter_nd
    mindspore.ops.select
    mindspore.ops.shape
    mindspore.ops.shuffle
    mindspore.ops.size
    mindspore.ops.slice
    mindspore.ops.space_to_batch_nd
    mindspore.ops.split
    mindspore.ops.squeeze
    mindspore.ops.stack
    mindspore.ops.strided_slice
    mindspore.ops.tensor_scatter_add
    mindspore.ops.tensor_scatter_div
    mindspore.ops.tensor_scatter_max
    mindspore.ops.tensor_scatter_min
    mindspore.ops.tensor_scatter_mul
    mindspore.ops.tensor_scatter_sub
    mindspore.ops.tile
    mindspore.ops.transpose
    mindspore.ops.unbind
    mindspore.ops.unique
    mindspore.ops.unique_with_pad
    mindspore.ops.unsorted_segment_max
    mindspore.ops.unsorted_segment_min
    mindspore.ops.unsorted_segment_prod
    mindspore.ops.unsorted_segment_sum
    mindspore.ops.unsqueeze
    mindspore.ops.unstack
    mindspore.ops.cumprod

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.scalar_cast
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

COO函数
++++++++++++++++

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.coo_abs
    mindspore.ops.coo_acos
    mindspore.ops.coo_acosh
    mindspore.ops.coo_asin
    mindspore.ops.coo_asinh
    mindspore.ops.coo_atan
    mindspore.ops.coo_atanh
    mindspore.ops.coo_ceil
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
	mindspore.ops.sparse_add

CSR函数
++++++++++++++++

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

图像函数
----------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bounding_box_decode
    mindspore.ops.bounding_box_encode
    mindspore.ops.check_valid
    mindspore.ops.crop_and_resize
    mindspore.ops.grid_sample
    mindspore.ops.iou
    mindspore.ops.pad
    mindspore.ops.padding
    mindspore.ops.pixel_shuffle
    mindspore.ops.pixel_unshuffle
