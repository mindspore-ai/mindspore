mindspore.mint
===============

mindspore.mint provides a large number of functional, nn, optimizer interfaces. The API usages and functions are consistent with the mainstream usage in the industry for easy reference.
The mint interface is currently an experimental interface and performs better than ops in graph mode of O0 and PyNative mode. Currently, the Ascend GE backend and CPU/GPU backend are not supported, and it will be gradually improved in the future.

The module import method is as follows:

.. code-block::

    from mindspore import mint

Compared with the previous version, the added, deleted and supported platforms change information of `mindspore.mint` operators in MindSpore, please refer to the link `mindspore.mint API Interface Change <https://gitee.com/mindspore/docs/blob/master/resource/api_updates/mint_api_updates_en.md>`_ .

Tensor
---------------

Creation Operations
^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.empty
    mindspore.mint.empty_like
    mindspore.mint.eye
    mindspore.mint.full
    mindspore.mint.full_like
    mindspore.mint.linspace
    mindspore.mint.ones
    mindspore.mint.ones_like
    mindspore.mint.polar
    mindspore.mint.zeros
    mindspore.mint.zeros_like

Indexing, Slicing, Joining, Mutating Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.cat
    mindspore.mint.chunk
    mindspore.mint.concat
    mindspore.mint.gather
    mindspore.mint.index_add
    mindspore.mint.index_select
    mindspore.mint.masked_select
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.permute
    mindspore.mint.reshape
    mindspore.mint.scatter
    mindspore.mint.scatter_add
    mindspore.mint.select
    mindspore.mint.split
    mindspore.mint.squeeze
    mindspore.mint.stack
    mindspore.mint.swapaxes
    mindspore.mint.tile
    mindspore.mint.transpose
    mindspore.mint.unbind
    mindspore.mint.unsqueeze
    mindspore.mint.where

Random Sampling
-----------------

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.bernoulli
    mindspore.mint.multinomial
    mindspore.mint.normal
    mindspore.mint.rand
    mindspore.mint.rand_like
    mindspore.mint.randint
    mindspore.mint.randint_like
    mindspore.mint.randn
    mindspore.mint.randn_like
    mindspore.mint.randperm

Math Operations
------------------

Pointwise Operations
^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.acos
    mindspore.mint.acosh
    mindspore.mint.add
    mindspore.mint.arccos
    mindspore.mint.arccosh
    mindspore.mint.arcsin
    mindspore.mint.arcsinh
    mindspore.mint.arctan
    mindspore.mint.arctan2
    mindspore.mint.arctanh
    mindspore.mint.asin
    mindspore.mint.asinh
    mindspore.mint.atan
    mindspore.mint.atan2
    mindspore.mint.atanh
    mindspore.mint.bitwise_and
    mindspore.mint.bitwise_or
    mindspore.mint.bitwise_xor
    mindspore.mint.ceil
    mindspore.mint.clamp
    mindspore.mint.cos
    mindspore.mint.cosh
    mindspore.mint.div
    mindspore.mint.divide
    mindspore.mint.erf
    mindspore.mint.erfc
    mindspore.mint.erfinv
    mindspore.mint.exp
    mindspore.mint.exp2
    mindspore.mint.expm1
    mindspore.mint.fix
    mindspore.mint.float_power
    mindspore.mint.floor
    mindspore.mint.floor_divide
    mindspore.mint.fmod
    mindspore.mint.frac
    mindspore.mint.imag
    mindspore.mint.lerp
    mindspore.mint.log
    mindspore.mint.log10
    mindspore.mint.log1p
    mindspore.mint.log2
    mindspore.mint.logaddexp
    mindspore.mint.logaddexp2
    mindspore.mint.logical_and
    mindspore.mint.logical_not
    mindspore.mint.logical_or
    mindspore.mint.logical_xor
    mindspore.mint.mul
    mindspore.mint.nan_to_num
    mindspore.mint.neg
    mindspore.mint.negative
    mindspore.mint.pow
    mindspore.mint.real
    mindspore.mint.reciprocal
    mindspore.mint.remainder
    mindspore.mint.round
    mindspore.mint.rsqrt
    mindspore.mint.sigmoid
    mindspore.mint.sign
    mindspore.mint.sin
    mindspore.mint.sinc
    mindspore.mint.sinh
    mindspore.mint.softmax
    mindspore.mint.sqrt
    mindspore.mint.square
    mindspore.mint.sub
    mindspore.mint.t
    mindspore.mint.tan
    mindspore.mint.tanh
    mindspore.mint.trunc
    mindspore.mint.xlogy

Reduction Operations
^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.all
    mindspore.mint.amax
    mindspore.mint.amin
    mindspore.mint.any
    mindspore.mint.argmax
    mindspore.mint.argmin
    mindspore.mint.count_nonzero
    mindspore.mint.logsumexp
    mindspore.mint.max
    mindspore.mint.mean
    mindspore.mint.median
    mindspore.mint.min
    mindspore.mint.nansum
    mindspore.mint.norm
    mindspore.mint.prod
    mindspore.mint.std
    mindspore.mint.std_mean
    mindspore.mint.sum
    mindspore.mint.unique
    mindspore.mint.unique_consecutive
    mindspore.mint.var
    mindspore.mint.var_mean

Comparison Operations
^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.allclose
    mindspore.mint.argsort
    mindspore.mint.eq
    mindspore.mint.equal
    mindspore.mint.greater
    mindspore.mint.greater_equal
    mindspore.mint.gt
    mindspore.mint.isclose
    mindspore.mint.isfinite
    mindspore.mint.isinf
    mindspore.mint.isneginf
    mindspore.mint.le
    mindspore.mint.less
    mindspore.mint.less_equal
    mindspore.mint.lt
    mindspore.mint.maximum
    mindspore.mint.minimum
    mindspore.mint.ne
    mindspore.mint.not_equal
    mindspore.mint.sort
    mindspore.mint.topk

BLAS and LAPACK Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.addbmm
    mindspore.mint.addmm
    mindspore.mint.addmv
    mindspore.mint.baddbmm
    mindspore.mint.bmm
    mindspore.mint.dot
    mindspore.mint.inverse
    mindspore.mint.matmul
    mindspore.mint.mm
    mindspore.mint.mv
    mindspore.mint.outer
    mindspore.mint.triangular_solve

Other Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.bincount
    mindspore.mint.broadcast_to
    mindspore.mint.cdist
    mindspore.mint.clone
    mindspore.mint.cross
    mindspore.mint.cummax
    mindspore.mint.cummin
    mindspore.mint.cumprod
    mindspore.mint.cumsum
    mindspore.mint.diag
    mindspore.mint.diff
    mindspore.mint.einsum
    mindspore.mint.flatten
    mindspore.mint.flip
    mindspore.mint.histc
    mindspore.mint.meshgrid
    mindspore.mint.ravel
    mindspore.mint.repeat_interleave
    mindspore.mint.roll
    mindspore.mint.searchsorted
    mindspore.mint.trace
    mindspore.mint.tril
    mindspore.mint.triu

mindspore.mint.nn
------------------

Convolution Layers
^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Conv1d
    mindspore.mint.nn.Conv2d
    mindspore.mint.nn.Conv3d
    mindspore.mint.nn.ConvTranspose2d
    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold

Pooling Layers
^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.AdaptiveAvgPool1d
    mindspore.mint.nn.AdaptiveAvgPool2d
    mindspore.mint.nn.AdaptiveAvgPool3d
    mindspore.mint.nn.AdaptiveMaxPool1d
    mindspore.mint.nn.AdaptiveMaxPool2d
    mindspore.mint.nn.AvgPool2d
    mindspore.mint.nn.AvgPool3d
    mindspore.mint.nn.MaxUnpool2d

Padding Layers
^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.ConstantPad1d
    mindspore.mint.nn.ConstantPad2d
    mindspore.mint.nn.ConstantPad3d
    mindspore.mint.nn.ReflectionPad1d
    mindspore.mint.nn.ReflectionPad2d
    mindspore.mint.nn.ReflectionPad3d
    mindspore.mint.nn.ReplicationPad1d
    mindspore.mint.nn.ReplicationPad2d
    mindspore.mint.nn.ReplicationPad3d
    mindspore.mint.nn.ZeroPad1d
    mindspore.mint.nn.ZeroPad2d
    mindspore.mint.nn.ZeroPad3d

Non-linear Activations (weighted sum, nonlinearity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.ELU
    mindspore.mint.nn.GELU
    mindspore.mint.nn.GLU
    mindspore.mint.nn.Hardshrink
    mindspore.mint.nn.Hardsigmoid
    mindspore.mint.nn.Hardswish
    mindspore.mint.nn.LogSigmoid
    mindspore.mint.nn.LogSoftmax
    mindspore.mint.nn.Mish
    mindspore.mint.nn.PReLU
    mindspore.mint.nn.ReLU
    mindspore.mint.nn.ReLU6
    mindspore.mint.nn.SELU
    mindspore.mint.nn.Sigmoid
    mindspore.mint.nn.SiLU
    mindspore.mint.nn.Softmax
    mindspore.mint.nn.Softshrink
    mindspore.mint.nn.Tanh
    mindspore.mint.nn.Threshold

Normalization Layers
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BatchNorm1d
    mindspore.mint.nn.BatchNorm2d
    mindspore.mint.nn.BatchNorm3d
    mindspore.mint.nn.GroupNorm
    mindspore.mint.nn.LayerNorm
    mindspore.mint.nn.SyncBatchNorm

Linear Layers
^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Identity
    mindspore.mint.nn.Linear

Dropout Layers
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout
    mindspore.mint.nn.Dropout2d

Sparse Layers
^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Embedding

Loss Functions
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCELoss
    mindspore.mint.nn.BCEWithLogitsLoss
    mindspore.mint.nn.CosineEmbeddingLoss
    mindspore.mint.nn.CrossEntropyLoss
    mindspore.mint.nn.KLDivLoss
    mindspore.mint.nn.L1Loss
    mindspore.mint.nn.MSELoss
    mindspore.mint.nn.NLLLoss
    mindspore.mint.nn.SmoothL1Loss

Vision Layers
^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.PixelShuffle
    mindspore.mint.nn.Upsample

mindspore.mint.nn.functional
-----------------------------

Convolution functions
^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.conv1d
    mindspore.mint.nn.functional.conv2d
    mindspore.mint.nn.functional.conv3d
    mindspore.mint.nn.functional.conv_transpose2d
    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold

Pooling functions
^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.adaptive_avg_pool1d
    mindspore.mint.nn.functional.adaptive_avg_pool2d
    mindspore.mint.nn.functional.adaptive_avg_pool3d
    mindspore.mint.nn.functional.adaptive_max_pool1d
    mindspore.mint.nn.functional.adaptive_max_pool2d
    mindspore.mint.nn.functional.avg_pool1d
    mindspore.mint.nn.functional.avg_pool2d
    mindspore.mint.nn.functional.avg_pool3d
    mindspore.mint.nn.functional.max_pool2d
    mindspore.mint.nn.functional.max_unpool2d

Non-linear activation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.elu_
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.glu
    mindspore.mint.nn.functional.group_norm
    mindspore.mint.nn.functional.hardshrink
    mindspore.mint.nn.functional.hardsigmoid
    mindspore.mint.nn.functional.hardswish
    mindspore.mint.nn.functional.layer_norm
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.log_softmax
    mindspore.mint.nn.functional.logsigmoid
    mindspore.mint.nn.functional.mish
    mindspore.mint.nn.functional.normalize
    mindspore.mint.nn.functional.prelu
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.relu_
    mindspore.mint.nn.functional.relu6
    mindspore.mint.nn.functional.selu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.softshrink
    mindspore.mint.nn.functional.tanh
    mindspore.mint.nn.functional.threshold
    mindspore.mint.nn.functional.threshold_

Linear functions
^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.linear

Dropout functions
^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.dropout
    mindspore.mint.nn.functional.dropout2d

Sparse functions
^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.embedding
    mindspore.mint.nn.functional.one_hot

Loss Functions
^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.binary_cross_entropy_with_logits
    mindspore.mint.nn.functional.cosine_embedding_loss
    mindspore.mint.nn.functional.cross_entropy
    mindspore.mint.nn.functional.kl_div
    mindspore.mint.nn.functional.l1_loss
    mindspore.mint.nn.functional.mse_loss
    mindspore.mint.nn.functional.nll_loss
    mindspore.mint.nn.functional.smooth_l1_loss

Vision functions
^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.interpolate
    mindspore.mint.nn.functional.pad
    mindspore.mint.nn.functional.pixel_shuffle

mindspore.mint.optim
---------------------

Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.optim.Adam
    mindspore.mint.optim.AdamW
    mindspore.mint.optim.SGD

mindspore.mint.linalg
----------------------

Matrix Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.matrix_norm
    mindspore.mint.linalg.norm
    mindspore.mint.linalg.vector_norm

Decompositions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.qr

Inverses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.inv

mindspore.mint.special
----------------------

Pointwise Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.special.erfc
    mindspore.mint.special.exp2
    mindspore.mint.special.expm1
    mindspore.mint.special.log1p
    mindspore.mint.special.log_softmax
    mindspore.mint.special.round
    mindspore.mint.special.sinc

mindspore.mint.distributed
--------------------------------

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.distributed.all_gather
    mindspore.mint.distributed.all_gather_into_tensor
    mindspore.mint.distributed.all_gather_into_tensor_uneven
    mindspore.mint.distributed.all_gather_object
    mindspore.mint.distributed.all_reduce
    mindspore.mint.distributed.all_to_all
    mindspore.mint.distributed.all_to_all_single
    mindspore.mint.distributed.barrier
    mindspore.mint.distributed.batch_isend_irecv
    mindspore.mint.distributed.broadcast
    mindspore.mint.distributed.broadcast_object_list
    mindspore.mint.distributed.destroy_process_group
    mindspore.mint.distributed.gather
    mindspore.mint.distributed.gather_object
    mindspore.mint.distributed.get_backend
    mindspore.mint.distributed.get_global_rank
    mindspore.mint.distributed.get_group_rank
    mindspore.mint.distributed.get_process_group_ranks
    mindspore.mint.distributed.get_rank
    mindspore.mint.distributed.get_world_size
    mindspore.mint.distributed.init_process_group
    mindspore.mint.distributed.irecv
    mindspore.mint.distributed.isend
    mindspore.mint.distributed.is_available
    mindspore.mint.distributed.is_initialized
    mindspore.mint.distributed.new_group
    mindspore.mint.distributed.P2POp
    mindspore.mint.distributed.recv
    mindspore.mint.distributed.reduce
    mindspore.mint.distributed.reduce_scatter
    mindspore.mint.distributed.reduce_scatter_tensor
    mindspore.mint.distributed.reduce_scatter_tensor_uneven
    mindspore.mint.distributed.scatter
    mindspore.mint.distributed.scatter_object_list
    mindspore.mint.distributed.send
    mindspore.mint.distributed.TCPStore
