mindspore.mint
===============

mindpsore.mint provides a large number of functional, nn, optimizer interfaces. The API usages and functions are consistent with the mainstream usage in the industry for easy reference.

The module import method is as follows:

.. code-block::

    from mindspore import mint

Tensor
---------------


Creation Operations
^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.eye
    mindspore.mint.ones
    mindspore.mint.ones_like
    mindspore.mint.zeros
    mindspore.mint.zeros_like

Indexing, Slicing, Joining, Mutating Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.broadcast_to
    mindspore.mint.cat
    mindspore.mint.flip
    mindspore.mint.index_select
    mindspore.mint.max
    mindspore.mint.min
    mindspore.mint.scatter_add
    mindspore.mint.split
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.normal
    mindspore.mint.tile
    mindspore.mint.topk
    mindspore.mint.sort
    mindspore.mint.stack
    mindspore.mint.where


Random Sampling
-----------------

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.rand_like
    mindspore.mint.rand


Math Operations
-----------------

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.gather
    mindspore.mint.permute
    mindspore.mint.repeat_interleave

Pointwise Operations
^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.add
    mindspore.mint.clamp
    mindspore.mint.cumsum
    mindspore.mint.atan2
    mindspore.mint.arctan2
    mindspore.mint.ceil
    mindspore.mint.cos
    mindspore.mint.unique
    mindspore.mint.div
    mindspore.mint.divide
    mindspore.mint.erf
    mindspore.mint.erfinv
    mindspore.mint.exp
    mindspore.mint.floor
    mindspore.mint.isfinite
    mindspore.mint.log
    mindspore.mint.logical_and
    mindspore.mint.logical_not
    mindspore.mint.logical_or
    mindspore.mint.mul
    mindspore.mint.neg
    mindspore.mint.negative
    mindspore.mint.pow
    mindspore.mint.reciprocal
    mindspore.mint.rsqrt
    mindspore.mint.sigmoid
    mindspore.mint.silu
    mindspore.mint.sin
    mindspore.mint.sqrt
    mindspore.mint.square
    mindspore.mint.sub

    mindspore.mint.tanh

Linear Algebraic Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.bmm
    mindspore.mint.matmul

Reduction Operations
^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.all
    mindspore.mint.any
    mindspore.mint.mean
    mindspore.mint.prod
    mindspore.mint.sum

Comparison Operations
^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.eq
    mindspore.mint.ne
    mindspore.mint.greater
    mindspore.mint.gt
    mindspore.mint.greater_equal
    mindspore.mint.isclose
    mindspore.mint.le
    mindspore.mint.less
    mindspore.mint.less_equal
    mindspore.mint.lt
    mindspore.mint.maximum
    mindspore.mint.minimum

BLAS and LAPACK Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.inverse

Other Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.searchsorted

Reduction Functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.argmax


mindspore.mint.nn
------------------

Dropout Layers
^^^^^^^^^^^^^^^

.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout




Convolution Layers
^^^^^^^^^^^^^^^^^^
.. msplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold



Loss Functions
^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCEWithLogitsLoss

mindspore.mint.nn.functional
-----------------------------

Neural Network Layer Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.dropout
    mindspore.mint.nn.functional.embedding
    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.group_norm
    mindspore.mint.nn.functional.layer_norm
    mindspore.mint.nn.functional.linear

Convolution functions
^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold


Tensor Creation
^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.one_hot



Pooling functions
^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.max_pool2d





Attention Mechanisms
^^^^^^^^^^^^^^^^^^^^^^^







Non-linear activation functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.tanh





Linear functions
^^^^^^^^^^^^^^^^^^^







Dropout functions
^^^^^^^^^^^^^^^^^^^







Distance functions
^^^^^^^^^^^^^^^^^^^






Loss Functions
^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy_with_logits





Vision functions
^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.pad






mindspore.mint.optim
---------------------

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.optim.AdamW

mindspore.mint.linalg
----------------------

Inverses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. msplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.inv
