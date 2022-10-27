.. role:: hidden
    :class: hidden-section

.. currentmodule:: {{ module }}

{% if objname in ["AdaSumByDeltaWeightWrapCell", "AdaSumByGradWrapCell", "DistributedGradReducer"] %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: infer_value, infer_shape, infer_dtype, construct
    :members:

{% elif fullname in ["mindspore.nn.Adam","mindspore.nn.AdamWeightDecay","mindspore.nn.FTRL","mindspore.nn.LazyAdam","mindspore.nn.ProximalAdagrad"] %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: infer_value, infer_shape, infer_dtype, target
    :members:

{% elif fullname=="mindspore.Tensor" %}
{{ fullname | underline }}

.. autoclass:: {{ name }}

Neural Network Layer Methods
----------------------------

Neural Network
^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.flatten

Activation Function
^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.hardshrink
    mindspore.Tensor.soft_shrink

Mathematical Methods
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
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
    mindspore.Tensor.igamma
    mindspore.Tensor.igammac

Element-wise Methods
^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.abs
    mindspore.Tensor.add
    mindspore.Tensor.addcdiv
    mindspore.Tensor.addcmul
    mindspore.Tensor.addr
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
    mindspore.Tensor.negative
    mindspore.Tensor.round
    mindspore.Tensor.sigmoid
    mindspore.Tensor.sqrt
    mindspore.Tensor.std
    mindspore.Tensor.sub
    mindspore.Tensor.svd
    mindspore.Tensor.square
    mindspore.Tensor.tan
    mindspore.Tensor.tanh
    mindspore.Tensor.var
    mindspore.Tensor.xdivy
    mindspore.Tensor.xlogy

Reduction Methods
^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
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

Comparison Methods
^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.all
    mindspore.Tensor.any
    mindspore.Tensor.approximate_equal
    mindspore.Tensor.ge
    mindspore.Tensor.greater
    mindspore.Tensor.greater_equal
    mindspore.Tensor.gt
    mindspore.Tensor.has_init
    mindspore.Tensor.isclose
    mindspore.Tensor.isfinite
    mindspore.Tensor.top_k

Linear Algebraic Methods
^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.ger
    mindspore.Tensor.log_matrix_determinant
    mindspore.Tensor.matrix_determinant
    mindspore.Tensor.det

Tensor Operation Methods
------------------------

Tensor Construction
^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.choose
    mindspore.Tensor.fill
    mindspore.Tensor.fills
    mindspore.Tensor.view

Random Generation Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.random_categorical

Array Methods
^^^^^^^^^^^^^

.. msplatformautosummary::
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
    mindspore.Tensor.index_add
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
    mindspore.Tensor.reverse
    mindspore.Tensor.reverse_sequence
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

Type Conversion
^^^^^^^^^^^^^^^

.. msplatformautosummary::
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

Gradient Clipping
^^^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.clip

Parameter Operation Methods
---------------------------

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.assign_value

Other Methods
--------------------

.. msplatformautosummary::
    :toctree: Tensor
    :nosignatures:

    mindspore.Tensor.flush_from_cache
    mindspore.Tensor.set_const_arg

{% elif objname[0].istitle() %}
{{ fullname | underline }}

.. autoclass:: {{ name }}
    :exclude-members: infer_value, infer_shape, infer_dtype
    :members:

{% else %}
{{ fullname | underline }}

.. autofunction:: {{ fullname }}

{% endif %}

..
  autogenerated from _templates/classtemplate.rst
  note it does not have :inherited-members:
