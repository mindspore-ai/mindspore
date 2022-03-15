mindspore.numpy
===============

Numpy-like interfaces in mindspore.

**Examples**

.. code-block::

    import mindspore.numpy as np

.. note::
    array_ops.py defines all the array operation interfaces.

    array_creations.py defines all the array generation interfaces.

    math_ops.py defines all the math operations on tensors.

    logic_ops.py defines all the logical operations on tensors.

    dtypes.py defines all the mindspore.numpy dtypes (mainly redirected from mindspore)

Array Generation
----------------

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.arange
    mindspore.numpy.array
    mindspore.numpy.asarray
    mindspore.numpy.asfarray
    mindspore.numpy.bartlett
    mindspore.numpy.blackman
    mindspore.numpy.copy
    mindspore.numpy.diag
    mindspore.numpy.diag_indices
    mindspore.numpy.diagflat
    mindspore.numpy.diagonal
    mindspore.numpy.empty
    mindspore.numpy.empty_like
    mindspore.numpy.eye
    mindspore.numpy.full
    mindspore.numpy.full_like
    mindspore.numpy.geomspace
    mindspore.numpy.hamming
    mindspore.numpy.hanning
    mindspore.numpy.histogram_bin_edges
    mindspore.numpy.identity
    mindspore.numpy.indices
    mindspore.numpy.ix_
    mindspore.numpy.linspace
    mindspore.numpy.logspace
    mindspore.numpy.meshgrid
    mindspore.numpy.mgrid
    mindspore.numpy.ogrid
    mindspore.numpy.ones
    mindspore.numpy.ones_like
    mindspore.numpy.pad
    mindspore.numpy.rand
    mindspore.numpy.randint
    mindspore.numpy.randn
    mindspore.numpy.trace
    mindspore.numpy.tri
    mindspore.numpy.tril
    mindspore.numpy.tril_indices
    mindspore.numpy.tril_indices_from
    mindspore.numpy.triu
    mindspore.numpy.triu_indices
    mindspore.numpy.triu_indices_from
    mindspore.numpy.vander
    mindspore.numpy.zeros
    mindspore.numpy.zeros_like

Array Operation
---------------

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.append
    mindspore.numpy.apply_along_axis
    mindspore.numpy.apply_over_axes
    mindspore.numpy.array_split
    mindspore.numpy.array_str
    mindspore.numpy.atleast_1d
    mindspore.numpy.atleast_2d
    mindspore.numpy.atleast_3d
    mindspore.numpy.broadcast_arrays
    mindspore.numpy.broadcast_to
    mindspore.numpy.choose
    mindspore.numpy.column_stack
    mindspore.numpy.concatenate
    mindspore.numpy.dsplit
    mindspore.numpy.dstack
    mindspore.numpy.expand_dims
    mindspore.numpy.flip
    mindspore.numpy.fliplr
    mindspore.numpy.flipud
    mindspore.numpy.hsplit
    mindspore.numpy.hstack
    mindspore.numpy.moveaxis
    mindspore.numpy.piecewise
    mindspore.numpy.ravel
    mindspore.numpy.repeat
    mindspore.numpy.reshape
    mindspore.numpy.roll
    mindspore.numpy.rollaxis
    mindspore.numpy.rot90
    mindspore.numpy.select
    mindspore.numpy.size
    mindspore.numpy.split
    mindspore.numpy.squeeze
    mindspore.numpy.stack
    mindspore.numpy.swapaxes
    mindspore.numpy.take
    mindspore.numpy.take_along_axis
    mindspore.numpy.tile
    mindspore.numpy.transpose
    mindspore.numpy.unique
    mindspore.numpy.unravel_index
    mindspore.numpy.vsplit
    mindspore.numpy.vstack
    mindspore.numpy.where

Logic
-----

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.array_equal
    mindspore.numpy.array_equiv
    mindspore.numpy.equal
    mindspore.numpy.greater
    mindspore.numpy.greater_equal
    mindspore.numpy.in1d
    mindspore.numpy.isclose
    mindspore.numpy.isfinite
    mindspore.numpy.isin
    mindspore.numpy.isinf
    mindspore.numpy.isnan
    mindspore.numpy.isneginf
    mindspore.numpy.isposinf
    mindspore.numpy.isscalar
    mindspore.numpy.less
    mindspore.numpy.less_equal
    mindspore.numpy.logical_and
    mindspore.numpy.logical_not
    mindspore.numpy.logical_or
    mindspore.numpy.logical_xor
    mindspore.numpy.not_equal
    mindspore.numpy.signbit
    mindspore.numpy.sometrue

Math
----

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.absolute
    mindspore.numpy.add
    mindspore.numpy.amax
    mindspore.numpy.amin
    mindspore.numpy.arccos
    mindspore.numpy.arccosh
    mindspore.numpy.arcsin
    mindspore.numpy.arcsinh
    mindspore.numpy.arctan
    mindspore.numpy.arctan2
    mindspore.numpy.arctanh
    mindspore.numpy.argmax
    mindspore.numpy.argmin
    mindspore.numpy.around
    mindspore.numpy.average
    mindspore.numpy.bincount
    mindspore.numpy.bitwise_and
    mindspore.numpy.bitwise_or
    mindspore.numpy.bitwise_xor
    mindspore.numpy.cbrt
    mindspore.numpy.ceil
    mindspore.numpy.clip
    mindspore.numpy.convolve
    mindspore.numpy.copysign
    mindspore.numpy.corrcoef
    mindspore.numpy.correlate
    mindspore.numpy.cos
    mindspore.numpy.cosh
    mindspore.numpy.count_nonzero
    mindspore.numpy.cov
    mindspore.numpy.cross
    mindspore.numpy.cumprod
    mindspore.numpy.cumsum
    mindspore.numpy.deg2rad
    mindspore.numpy.diff
    mindspore.numpy.digitize
    mindspore.numpy.divide
    mindspore.numpy.divmod
    mindspore.numpy.dot
    mindspore.numpy.ediff1d
    mindspore.numpy.exp
    mindspore.numpy.exp2
    mindspore.numpy.expm1
    mindspore.numpy.fix
    mindspore.numpy.float_power
    mindspore.numpy.floor
    mindspore.numpy.floor_divide
    mindspore.numpy.fmod
    mindspore.numpy.gcd
    mindspore.numpy.gradient
    mindspore.numpy.heaviside
    mindspore.numpy.histogram
    mindspore.numpy.histogram2d
    mindspore.numpy.histogramdd
    mindspore.numpy.hypot
    mindspore.numpy.inner
    mindspore.numpy.interp
    mindspore.numpy.invert
    mindspore.numpy.kron
    mindspore.numpy.lcm
    mindspore.numpy.log
    mindspore.numpy.log10
    mindspore.numpy.log1p
    mindspore.numpy.log2
    mindspore.numpy.logaddexp
    mindspore.numpy.logaddexp2
    mindspore.numpy.matmul
    mindspore.numpy.matrix_power
    mindspore.numpy.maximum
    mindspore.numpy.mean
    mindspore.numpy.minimum
    mindspore.numpy.multi_dot
    mindspore.numpy.multiply
    mindspore.numpy.nancumsum
    mindspore.numpy.nanmax
    mindspore.numpy.nanmean
    mindspore.numpy.nanmin
    mindspore.numpy.nanstd
    mindspore.numpy.nansum
    mindspore.numpy.nanvar
    mindspore.numpy.negative
    mindspore.numpy.norm
    mindspore.numpy.outer
    mindspore.numpy.polyadd
    mindspore.numpy.polyder
    mindspore.numpy.polyint
    mindspore.numpy.polymul
    mindspore.numpy.polysub
    mindspore.numpy.polyval
    mindspore.numpy.positive
    mindspore.numpy.power
    mindspore.numpy.promote_types
    mindspore.numpy.ptp
    mindspore.numpy.rad2deg
    mindspore.numpy.radians
    mindspore.numpy.ravel_multi_index
    mindspore.numpy.reciprocal
    mindspore.numpy.remainder
    mindspore.numpy.result_type
    mindspore.numpy.rint
    mindspore.numpy.searchsorted
    mindspore.numpy.sign
    mindspore.numpy.sin
    mindspore.numpy.sinh
    mindspore.numpy.sqrt
    mindspore.numpy.square
    mindspore.numpy.std
    mindspore.numpy.subtract
    mindspore.numpy.sum
    mindspore.numpy.tan
    mindspore.numpy.tanh
    mindspore.numpy.tensordot
    mindspore.numpy.trapz
    mindspore.numpy.true_divide
    mindspore.numpy.trunc
    mindspore.numpy.unwrap
    mindspore.numpy.var