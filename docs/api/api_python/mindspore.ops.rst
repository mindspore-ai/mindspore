mindspore.ops
=============================

mindspore.ops提供了大量的function接口，包括神经网络函数、数学运算函数、Tensor操作函数、Parameter操作函数、微分函数等。

动态shape的支持情况详见 `ops接口动态shape支持情况 <https://mindspore.cn/docs/zh-CN/master/api_python/dynamic_shape_func.html>`_ 。

模块导入方法如下：

.. code-block::

    from mindspore import ops

MindSpore中 `mindspore.ops` 接口与上一版本相比，新增、删除和支持平台的变化信息请参考 `mindspore.ops API接口变更 <https://atomgit.com/mindspore/docs/blob/master/resource/api_updates/func_api_updates_cn.md>`_ 。

Tensor操作函数
----------------

Tensor创建
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.arange
    mindspore.ops.eps
    mindspore.ops.eye
    mindspore.ops.fill
    mindspore.ops.full
    mindspore.ops.full_like
    mindspore.ops.heaviside
    mindspore.ops.linspace
    mindspore.ops.logspace
    mindspore.ops.move_to
    mindspore.ops.one_hot
    mindspore.ops.ones
    mindspore.ops.ones_like
    mindspore.ops.range
    mindspore.ops.zeros
    mindspore.ops.zeros_like

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
    mindspore.ops.randperm
    mindspore.ops.standard_laplace
    mindspore.ops.uniform

Array操作
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.argwhere
    mindspore.ops.bincount
    mindspore.ops.block_diag
    mindspore.ops.broadcast_to
    mindspore.ops.cat
    mindspore.ops.channel_shuffle
    mindspore.ops.chunk
    mindspore.ops.column_stack
    mindspore.ops.concat
    mindspore.ops.conj
    mindspore.ops.count_nonzero
    mindspore.ops.deepcopy
    mindspore.ops.diag
    mindspore.ops.diagflat
    mindspore.ops.diagonal
    mindspore.ops.diagonal_scatter
    mindspore.ops.diag_embed
    mindspore.ops.dsplit
    mindspore.ops.dstack
    mindspore.ops.einsum
    mindspore.ops.expand_dims
    mindspore.ops.flip
    mindspore.ops.fliplr
    mindspore.ops.flipud
    mindspore.ops.gather
    mindspore.ops.gather_d
    mindspore.ops.gather_elements
    mindspore.ops.gather_nd
    mindspore.ops.hsplit
    mindspore.ops.hstack
    mindspore.ops.index_add
    mindspore.ops.index_fill
    mindspore.ops.index_select
    mindspore.ops.inplace_update
    mindspore.ops.is_nonzero
    mindspore.ops.masked_fill
    mindspore.ops.masked_select
    mindspore.ops.meshgrid
    mindspore.ops.moveaxis
    mindspore.ops.movedim
    mindspore.ops.narrow
    mindspore.ops.nan_to_num
    mindspore.ops.nanmean
    mindspore.ops.nanmedian
    mindspore.ops.nansum
    mindspore.ops.normal
    mindspore.ops.nonzero
    mindspore.ops.nsa_compress
    mindspore.ops.nsa_compress_attention
    mindspore.ops.numel
    mindspore.ops.permute
    mindspore.ops.rank
    mindspore.ops.repeat_elements
    mindspore.ops.repeat_interleave
    mindspore.ops.reshape
    mindspore.ops.reverse_sequence
    mindspore.ops.roll
    mindspore.ops.row_stack
    mindspore.ops.scatter
    mindspore.ops.scatter_nd
    mindspore.ops.select
    mindspore.ops.select_scatter
    mindspore.ops.shape
    mindspore.ops.size
    mindspore.ops.slice
    mindspore.ops.slice_scatter
    mindspore.ops.sort
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
    mindspore.ops.triu
    mindspore.ops.transpose
    mindspore.ops.unbind
    mindspore.ops.unique
    mindspore.ops.unique_consecutive
    mindspore.ops.unsorted_segment_max
    mindspore.ops.unsorted_segment_min
    mindspore.ops.unsorted_segment_prod
    mindspore.ops.unsorted_segment_sum
    mindspore.ops.unsqueeze
    mindspore.ops.unstack
    mindspore.ops.view_as_real
    mindspore.ops.vsplit
    mindspore.ops.vstack
    mindspore.ops.where
    mindspore.ops.cross
    mindspore.ops.renorm

类型转换
^^^^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.cast
    mindspore.ops.scalar_to_tensor
    mindspore.ops.tuple_to_array

梯度剪裁
^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.clip_by_value

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
    mindspore.ops.atleast_1d
    mindspore.ops.atleast_2d
    mindspore.ops.atleast_3d
    mindspore.ops.bessel_i0
    mindspore.ops.bessel_i0e
    mindspore.ops.bessel_i1
    mindspore.ops.bessel_i1e
    mindspore.ops.bessel_j0
    mindspore.ops.bessel_j1
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
    mindspore.ops.combinations
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
    mindspore.ops.floor_divide
    mindspore.ops.floor_mod
    mindspore.ops.float_power
    mindspore.ops.fmax
    mindspore.ops.fmod
    mindspore.ops.frac
    mindspore.ops.gcd
    mindspore.ops.hypot
    mindspore.ops.igamma
    mindspore.ops.igammac
    mindspore.ops.imag
    mindspore.ops.i0
    mindspore.ops.inv
    mindspore.ops.invert
    mindspore.ops.lcm
    mindspore.ops.ldexp
    mindspore.ops.lerp
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
    mindspore.ops.mul
    mindspore.ops.multiply
    mindspore.ops.mvlgamma
    mindspore.ops.neg
    mindspore.ops.negative
    mindspore.ops.nextafter
    mindspore.ops.polar
    mindspore.ops.polygamma
    mindspore.ops.positive
    mindspore.ops.pow
    mindspore.ops.rad2deg
    mindspore.ops.ravel
    mindspore.ops.real
    mindspore.ops.reciprocal
    mindspore.ops.remainder
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
    mindspore.ops.tril_indices
    mindspore.ops.triu_indices
    mindspore.ops.true_divide
    mindspore.ops.trunc
    mindspore.ops.truncate_mod
    mindspore.ops.xdivy
    mindspore.ops.xlogy
    mindspore.ops.zeta

Reduction函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.all
    mindspore.ops.amax
    mindspore.ops.amin
    mindspore.ops.aminmax
    mindspore.ops.any
    mindspore.ops.argmax
    mindspore.ops.argmin
    mindspore.ops.cummax
    mindspore.ops.cummin
    mindspore.ops.cumprod
    mindspore.ops.cumsum
    mindspore.ops.histc
    mindspore.ops.logcumsumexp
    mindspore.ops.logsumexp
    mindspore.ops.max
    mindspore.ops.mean
    mindspore.ops.median
    mindspore.ops.min
    mindspore.ops.norm
    mindspore.ops.prod
    mindspore.ops.std
    mindspore.ops.std_mean
    mindspore.ops.var
    mindspore.ops.var_mean

比较函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.argsort
    mindspore.ops.bucketize
    mindspore.ops.eq
    mindspore.ops.equal
    mindspore.ops.ge
    mindspore.ops.greater
    mindspore.ops.greater_equal
    mindspore.ops.gt
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
    mindspore.ops.msort
    mindspore.ops.ne
    mindspore.ops.not_equal
    mindspore.ops.searchsorted
    mindspore.ops.topk

线性代数函数
^^^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.addbmm
    mindspore.ops.addmm
    mindspore.ops.addr
    mindspore.ops.adjoint
    mindspore.ops.baddbmm
    mindspore.ops.bmm
    mindspore.ops.cholesky
    mindspore.ops.cholesky_solve
    mindspore.ops.cond
    mindspore.ops.dot
    mindspore.ops.eigvals
    mindspore.ops.geqrf
    mindspore.ops.ger
    mindspore.ops.inner
    mindspore.ops.inverse
    mindspore.ops.kron
    mindspore.ops.logdet
    mindspore.ops.lu_solve
    mindspore.ops.lu_unpack
    mindspore.ops.matmul
    mindspore.ops.matrix_band_part
    mindspore.ops.matrix_solve
    mindspore.ops.mm
    mindspore.ops.mv
    mindspore.ops.outer
    mindspore.ops.orgqr
    mindspore.ops.ormqr
    mindspore.ops.pinv
    mindspore.ops.svd
    mindspore.ops.slogdet
    mindspore.ops.trace
    mindspore.ops.tensor_dot
    mindspore.ops.vander
    mindspore.ops.vecdot

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
    mindspore.ops.avg_pool1d
    mindspore.ops.avg_pool2d
    mindspore.ops.avg_pool3d
    mindspore.ops.batch_norm
    mindspore.ops.bias_add
    mindspore.ops.conv1d
    mindspore.ops.conv2d
    mindspore.ops.conv3d
    mindspore.ops.dense
    mindspore.ops.dropout
    mindspore.ops.dropout1d
    mindspore.ops.dropout2d
    mindspore.ops.dropout3d
    mindspore.ops.embedding
    mindspore.ops.flash_attention_score
    mindspore.ops.flatten
    mindspore.ops.fold
    mindspore.ops.fractional_max_pool3d
    mindspore.ops.fused_infer_attention_score
    mindspore.ops.group_norm
    mindspore.ops.incre_flash_attention
    mindspore.ops.layer_norm
    mindspore.ops.lightning_indexer
    mindspore.ops.lp_pool1d
    mindspore.ops.lp_pool2d
    mindspore.ops.max_pool2d
    mindspore.ops.max_pool3d
    mindspore.ops.max_unpool1d
    mindspore.ops.max_unpool2d
    mindspore.ops.max_unpool3d
    mindspore.ops.moe_token_permute
    mindspore.ops.moe_token_unpermute
    mindspore.ops.nsa_select_attention
    mindspore.ops.prompt_flash_attention
    mindspore.ops.ring_attention_update
    mindspore.ops.rms_norm
    mindspore.ops.sparse_flash_attention
    mindspore.ops.speed_fusion_attention
    mindspore.ops.unfold

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
    mindspore.ops.ctc_loss
    mindspore.ops.gaussian_nll_loss
    mindspore.ops.hinge_embedding_loss
    mindspore.ops.huber_loss
    mindspore.ops.kl_div
    mindspore.ops.l1_loss
    mindspore.ops.margin_ranking_loss
    mindspore.ops.mse_loss
    mindspore.ops.multi_margin_loss
    mindspore.ops.multilabel_margin_loss
    mindspore.ops.multilabel_soft_margin_loss
    mindspore.ops.nll_loss
    mindspore.ops.smooth_l1_loss
    mindspore.ops.soft_margin_loss
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
    mindspore.ops.hardsigmoid
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
    mindspore.ops.softmax
    mindspore.ops.softmin
    mindspore.ops.softshrink
    mindspore.ops.softsign
    mindspore.ops.swiglu
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

    mindspore.ops.random_categorical

图像函数
^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.affine_grid
    mindspore.ops.grid_sample
    mindspore.ops.interpolate
    mindspore.ops.pad
    mindspore.ops.padding
    mindspore.ops.pixel_shuffle
    mindspore.ops.pixel_unshuffle
    mindspore.ops.rotary_position_embedding
    mindspore.ops.upsample

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
    mindspore.ops.tensordump

稀疏函数
----------------

.. note::
    本版本已移除基于公开 COO/CSR 稀疏 Tensor 对象的函数及稀疏 Tensor 转换入口。
    原有代码需要按具体算子的签名、平台支持和语义进行迁移。
    接收普通 Tensor 分量的共享稀疏原语继续保留；私有稀疏辅助函数不作为稳定的公开替代接口。

MC2函数
----------------

.. warning::
    这些是实验性API，后续可能修改或删除。

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.all_gather_matmul
    mindspore.ops.matmul_reduce_scatter

通信函数
--------------------------------

.. mscnplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.communication.all_gather
    mindspore.ops.communication.all_gather_into_tensor
    mindspore.ops.communication.all_gather_into_tensor_uneven
    mindspore.ops.communication.all_gather_object
    mindspore.ops.communication.all_reduce
    mindspore.ops.communication.all_to_all
    mindspore.ops.communication.all_to_all_single
    mindspore.ops.communication.all_to_all_v_c
    mindspore.ops.communication.barrier
    mindspore.ops.communication.batch_isend_irecv
    mindspore.ops.communication.broadcast
    mindspore.ops.communication.broadcast_object_list
    mindspore.ops.communication.destroy_process_group
    mindspore.ops.communication.gather
    mindspore.ops.communication.gather_into_tensor
    mindspore.ops.communication.gather_object
    mindspore.ops.communication.get_backend
    mindspore.ops.communication.get_global_rank
    mindspore.ops.communication.get_group_rank
    mindspore.ops.communication.get_process_group_ranks
    mindspore.ops.communication.get_rank
    mindspore.ops.communication.get_world_size
    mindspore.ops.communication.init_process_group
    mindspore.ops.communication.irecv
    mindspore.ops.communication.isend
    mindspore.ops.communication.is_available
    mindspore.ops.communication.is_initialized
    mindspore.ops.communication.new_group
    mindspore.ops.communication.P2POp
    mindspore.ops.communication.recv
    mindspore.ops.communication.recv_object_list
    mindspore.ops.communication.reduce
    mindspore.ops.communication.reduce_scatter
    mindspore.ops.communication.reduce_scatter_tensor
    mindspore.ops.communication.reduce_scatter_tensor_uneven
    mindspore.ops.communication.scatter
    mindspore.ops.communication.scatter_object_list
    mindspore.ops.communication.scatter_tensor
    mindspore.ops.communication.send
    mindspore.ops.communication.send_object_list
    mindspore.ops.communication.set_comm_ops_inplace
    mindspore.ops.communication.TCPStore
