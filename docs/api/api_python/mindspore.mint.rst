mindspore.mint
===============

张量
---------------

创建运算
^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.eye
    mindspore.mint.ones_like
    mindspore.mint.zeros_like

索引、切分、连接、突变运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.cat
    mindspore.mint.index_select
    mindspore.mint.nonzero
    mindspore.mint.scatter_add
    mindspore.mint.stack
    mindspore.mint.tile
    mindspore.mint.where

随机采样
---------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.rand_like
    mindspore.mint.rand

数学运算
---------------

逐元素运算
^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.add
    mindspore.mint.arctan2
    mindspore.mint.atan2
    mindspore.mint.ceil
    mindspore.mint.clamp
    mindspore.mint.cos
    mindspore.mint.div
    mindspore.mint.divide
    mindspore.mint.erf
    mindspore.mint.floor
    mindspore.mint.log
    mindspore.mint.tanh
    mindspore.mint.square
    mindspore.mint.sub

Reduction运算
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.mint.all
    mindspore.mint.any
    mindspore.mint.argmax
    mindspore.mint.max
    mindspore.mint.mean
    mindspore.mint.min
    mindspore.mint.prod
    mindspore.mint.sum
    mindspore.mint.unique

比较运算
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.greater_equal
    mindspore.mint.isclose
    mindspore.mint.isfinite
    mindspore.mint.maximum
    mindspore.mint.minimum
    mindspore.mint.sort
    mindspore.mint.topk

BLAS和LAPACK运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.bmm
    mindspore.mint.matmul

其他运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.broadcast_to
    mindspore.mint.flip

mindspore.mint.nn
-----------------

卷积层
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold

Dropout层
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout

mindspore.mint.nn.functional
--------------------------------

卷积函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold

池化函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.max_pool2d

非线性激活函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.group_norm
    mindspore.mint.nn.functional.layer_norm
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.tanh

线性函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst
    
    mindspore.mint.nn.functional.linear

Dropout函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.dropout

稀疏函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.one_hot

Vision函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.pad

mindspore.mint.optim
-----------------------


