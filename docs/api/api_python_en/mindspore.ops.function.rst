mindspore.ops.function
=============================

Neural Network Layer Functions
------------------------------

Neural Network
^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.adaptive_avg_pool1d
    mindspore.ops.adaptive_avg_pool2d
    mindspore.ops.adaptive_avg_pool3d
    mindspore.ops.adaptive_max_pool1d
    mindspore.ops.adaptive_max_pool3d
    mindspore.ops.avg_pool1d
    mindspore.ops.avg_pool2d
    mindspore.ops.avg_pool3d
    mindspore.ops.batch_norm
    mindspore.ops.bias_add
    mindspore.ops.conv2d
    mindspore.ops.conv3d
    mindspore.ops.ctc_greedy_decoder
    mindspore.ops.crop_and_resize
    mindspore.ops.deformable_conv2d
    mindspore.ops.dropout
    mindspore.ops.dropout2d
    mindspore.ops.dropout3d
    mindspore.ops.flatten
    mindspore.ops.interpolate
    mindspore.ops.lrn
    mindspore.ops.max_pool3d
    mindspore.ops.multi_margin_loss
    mindspore.ops.multi_label_margin_loss
    mindspore.ops.kl_div
    mindspore.ops.pad
    mindspore.ops.padding
    mindspore.ops.pdist
    mindspore.ops.prelu
    mindspore.ops.relu
    mindspore.ops.relu6


Loss Functions
^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.binary_cross_entropy
    mindspore.ops.binary_cross_entropy_with_logits
    mindspore.ops.cross_entropy
    mindspore.ops.nll_loss
    mindspore.ops.smooth_l1_loss

Activation Functions
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.celu
    mindspore.ops.dropout
    mindspore.ops.elu
    mindspore.ops.fast_gelu
    mindspore.ops.gelu
    mindspore.ops.glu    
    mindspore.ops.gumbel_softmax
    mindspore.ops.hardshrink
    mindspore.ops.hardswish
    mindspore.ops.log_softmax
    mindspore.ops.mish
    mindspore.ops.selu
    mindspore.ops.softsign
    mindspore.ops.soft_shrink
    mindspore.ops.softmax
    mindspore.ops.tanh

Sampling Functions
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.grid_sample
    mindspore.ops.uniform_candidate_sampler

Distance Functions
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.cdist

Mathematical Functions
^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bmm
    mindspore.ops.cholesky
    mindspore.ops.cholesky_inverse
    mindspore.ops.conj
    mindspore.ops.cross
    mindspore.ops.erfinv
    mindspore.ops.less_equal

Element-by-Element Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.abs
    mindspore.ops.accumulate_n
    mindspore.ops.acos
    mindspore.ops.acosh
    mindspore.ops.add
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
    mindspore.ops.logit
    mindspore.ops.log_matrix_determinant
    mindspore.ops.matrix_determinant
    mindspore.ops.mul
    mindspore.ops.neg
    mindspore.ops.pow
    mindspore.ops.round
    mindspore.ops.sin
    mindspore.ops.sinh
    mindspore.ops.sqrt
    mindspore.ops.square
    mindspore.ops.sub
    mindspore.ops.svd
    mindspore.ops.tan
    mindspore.ops.trunc
    mindspore.ops.truncate_div
    mindspore.ops.truncate_mod
    mindspore.ops.xdivy
    mindspore.ops.xlogy

Reduction Functions
^^^^^^^^^^^^^^^^^^^
.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.amax
    mindspore.ops.amin
    mindspore.ops.argmax
    mindspore.ops.argmin
    mindspore.ops.cummax
    mindspore.ops.cummin
    mindspore.ops.logsumexp
    mindspore.ops.max
    mindspore.ops.mean
    mindspore.ops.median
    mindspore.ops.min
    mindspore.ops.norm
    mindspore.ops.prod
    mindspore.ops.std

Comparison Functions
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
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

Linear Algebraic Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
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

Tensor Operation Functions
--------------------------

Tensor Building
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.eye
    mindspore.ops.fill
    mindspore.ops.fills
    mindspore.ops.linspace
    mindspore.ops.narrow
    mindspore.ops.one_hot
    mindspore.ops.ones
    mindspore.ops.ones_like
    mindspore.ops.zeros_like

Randomly Generating Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.choice_with_mask
    mindspore.ops.gamma
    mindspore.ops.laplace
    mindspore.ops.multinomial
    mindspore.ops.random_poisson
    mindspore.ops.random_categorical
    mindspore.ops.random_gamma
    mindspore.ops.shuffle
    mindspore.ops.standard_laplace
    mindspore.ops.standard_normal
    mindspore.ops.uniform

Array Operation
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.adaptive_max_pool2d
    mindspore.ops.affine_grid
    mindspore.ops.batch_to_space_nd
    mindspore.ops.broadcast_to
    mindspore.ops.col2im
    mindspore.ops.concat
    mindspore.ops.count_nonzero
    mindspore.ops.diag
    mindspore.ops.diagonal
    mindspore.ops.dyn_shape
    mindspore.ops.expand
    mindspore.ops.expand_dims
    mindspore.ops.fold
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
    mindspore.ops.strided_slice
    mindspore.ops.tensor_scatter_add
    mindspore.ops.tensor_scatter_min
    mindspore.ops.tensor_scatter_max
    mindspore.ops.tensor_scatter_div
    mindspore.ops.tensor_scatter_mul
    mindspore.ops.tensor_scatter_sub
    mindspore.ops.tensor_scatter_elements
    mindspore.ops.tile
    mindspore.ops.top_k
    mindspore.ops.transpose
    mindspore.ops.unfold
    mindspore.ops.unique
    mindspore.ops.unique_consecutive
    mindspore.ops.unique_with_pad
    mindspore.ops.unsorted_segment_max
    mindspore.ops.unsorted_segment_min
    mindspore.ops.unsorted_segment_prod
    mindspore.ops.unsorted_segment_sum
    mindspore.ops.unstack

Type Conversion
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.scalar_cast
    mindspore.ops.scalar_to_array
    mindspore.ops.scalar_to_tensor
    mindspore.ops.tuple_to_array

Sparse Functions
^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.dense_to_sparse_coo
    mindspore.ops.dense_to_sparse_csr
    mindspore.ops.csr_add
    mindspore.ops.csr_softmax
    mindspore.ops.csr_to_coo
    mindspore.ops.sparse_add

Gradient Clipping
^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.clip_by_global_norm
    mindspore.ops.clip_by_value

Parameter Operation Functions
-----------------------------

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.assign
    mindspore.ops.assign_add
    mindspore.ops.assign_sub
    mindspore.ops.scatter_add
    mindspore.ops.scatter_div
    mindspore.ops.scatter_min
    mindspore.ops.scatter_max
    mindspore.ops.scatter_mul
    mindspore.ops.scatter_nd_add
    mindspore.ops.scatter_nd_div
    mindspore.ops.scatter_nd_max
    mindspore.ops.scatter_nd_min
    mindspore.ops.scatter_nd_mul
    mindspore.ops.scatter_nd_sub
    mindspore.ops.scatter_update

Differential Functions
----------------------

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.derivative
    mindspore.ops.jet
    mindspore.ops.vmap

Debugging Functions
-------------------

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.print_

Image Functions
---------------

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.bounding_box_decode
    mindspore.ops.bounding_box_encode
    mindspore.ops.check_valid
    mindspore.ops.iou

Spectral Functions
------------------

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.blackman_window

Other Functions
---------------

.. msplatformautosummary::
    :toctree: ops
    :nosignatures:
    :template: classtemplate.rst

    mindspore.ops.core
