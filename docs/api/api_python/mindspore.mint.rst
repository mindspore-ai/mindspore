mindspore.mint
===============

mindspore.mint提供了大量的functional、nn、优化器接口，API用法及功能等与业界主流用法一致，方便用户参考使用。
mint接口当前是实验性接口，在图编译模式为O0和PyNative模式下性能比ops更优。当前暂不支持图下沉模式及CPU、GPU后端，后续会逐步完善。

模块导入方法如下：

.. code-block::

    from mindspore import mint

Tensor
---------------

创建运算
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.eye
    mindspore.mint.full
    mindspore.mint.linspace
    mindspore.mint.ones
    mindspore.mint.ones_like
    mindspore.mint.zeros
    mindspore.mint.zeros_like

索引、切分、连接、突变运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.cat
    mindspore.mint.gather
    mindspore.mint.index_select
    mindspore.mint.permute
    mindspore.mint.scatter_add
    mindspore.mint.split
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.tile
    mindspore.mint.stack
    mindspore.mint.where

随机采样
------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.normal
    mindspore.mint.rand_like
    mindspore.mint.rand

数学运算
------------------

逐元素运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.add
    mindspore.mint.acos
    mindspore.mint.acosh
    mindspore.mint.arccos
    mindspore.mint.arccosh
    mindspore.mint.arcsin
    mindspore.mint.arcsinh
    mindspore.mint.arctan2
    mindspore.mint.asin
    mindspore.mint.asinh
    mindspore.mint.atan2
    mindspore.mint.bitwise_and
    mindspore.mint.bitwise_or
    mindspore.mint.bitwise_xor
    mindspore.mint.ceil
    mindspore.mint.clamp
    mindspore.mint.cos
    mindspore.mint.cosh
    mindspore.mint.cross
    mindspore.mint.div
    mindspore.mint.divide
    mindspore.mint.erf
    mindspore.mint.erfinv
    mindspore.mint.exp
    mindspore.mint.floor
    mindspore.mint.log
    mindspore.mint.logical_and
    mindspore.mint.logical_not
    mindspore.mint.logical_or
    mindspore.mint.mul
    mindspore.mint.neg
    mindspore.mint.negative
    mindspore.mint.pow
    mindspore.mint.reciprocal
    mindspore.mint.roll
    mindspore.mint.rsqrt
    mindspore.mint.sigmoid
    mindspore.mint.sin
    mindspore.mint.sinc
    mindspore.mint.sinh
    mindspore.mint.sqrt
    mindspore.mint.square
    mindspore.mint.sub
    mindspore.mint.tanh

Reduction运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.argmax
    mindspore.mint.argmin
    mindspore.mint.all
    mindspore.mint.any
    mindspore.mint.max
    mindspore.mint.mean
    mindspore.mint.median
    mindspore.mint.min
    mindspore.mint.prod
    mindspore.mint.sum
    mindspore.mint.unique

比较运算
^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.eq
    mindspore.mint.greater
    mindspore.mint.greater_equal
    mindspore.mint.gt
    mindspore.mint.isclose
    mindspore.mint.isfinite
    mindspore.mint.le
    mindspore.mint.less
    mindspore.mint.less_equal
    mindspore.mint.lt
    mindspore.mint.maximum
    mindspore.mint.minimum
    mindspore.mint.ne
    mindspore.mint.topk
    mindspore.mint.sort

BLAS和LAPACK运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.bmm
    mindspore.mint.inverse
    mindspore.mint.matmul

其他运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.broadcast_to
    mindspore.mint.cummax
    mindspore.mint.cummin
    mindspore.mint.cumsum
    mindspore.mint.flatten
    mindspore.mint.flip
    mindspore.mint.repeat_interleave
    mindspore.mint.searchsorted

mindspore.mint.nn
------------------

损失函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.L1Loss

卷积层
^^^^^^^^^^^^^^^^^^
.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold

非线性激活层 (加权和，非线性)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Hardshrink

线性层
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Linear

Dropout层
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout

池化层
^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.AvgPool2d

损失函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCEWithLogitsLoss

mindspore.mint.nn.functional
-----------------------------

卷积函数
^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.fold
    mindspore.mint.nn.functional.unfold

池化函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.avg_pool2d
    mindspore.mint.nn.functional.max_pool2d

非线性激活函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.group_norm
    mindspore.mint.nn.functional.hardshrink
    mindspore.mint.nn.functional.hardsigmoid
    mindspore.mint.nn.functional.hardswish
    mindspore.mint.nn.functional.layer_norm
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.relu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.tanh

线性函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.linear

Dropout函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.dropout

稀疏函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.embedding
    mindspore.mint.nn.functional.one_hot

损失函数
^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.binary_cross_entropy_with_logits
    mindspore.mint.nn.functional.l1_loss

Vision函数
^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.pad

mindspore.mint.optim
---------------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.optim.AdamW

mindspore.mint.linalg
----------------------

逆数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.linalg.inv

mindspore.mint.special
----------------------

三角函数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.special.sinc
