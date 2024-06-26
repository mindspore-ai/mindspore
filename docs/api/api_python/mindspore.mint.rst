mindspore.mint
===============

mindspore.mint提供了大量的functional、nn、优化器接口，API用法及功能等与业界主流用法一致，方便用户参考使用。

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

    mindspore.mint.eye
    mindspore.mint.ones
    mindspore.mint.ones_like
    mindspore.mint.one_hot
    mindspore.mint.zeros
    mindspore.mint.zeros_like

索引、切分、连接、突变运算
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.arange
    mindspore.mint.broadcast_to
    mindspore.mint.cat
    mindspore.mint.index_select
    mindspore.mint.scatter_add
    mindspore.mint.split
    mindspore.mint.narrow
    mindspore.mint.nonzero
    mindspore.mint.normal
    mindspore.mint.topk
    mindspore.mint.sort
    mindspore.mint.stack

随机采样
-----------------
.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.rand_like
    mindspore.mint.rand


数学运算
-----------------

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.repeat_interleave

逐元素运算
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.abs
    mindspore.mint.add
    mindspore.mint.cumsum
    mindspore.mint.atan2
    mindspore.mint.arctan2
    mindspore.mint.unique
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
    mindspore.mint.rsqrt
    mindspore.mint.sin
    mindspore.mint.sqrt
    mindspore.mint.sub
    mindspore.mint.tanh

线性函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.bmm
    mindspore.mint.matmul

Reduction 函数
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.any

比较函数
^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.greater_equal


Reduction运算
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.all
    mindspore.mint.mean
    mindspore.mint.prod
    mindspore.mint.sum


比较运算
^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.eq
    mindspore.mint.greater
    mindspore.mint.gt
    mindspore.mint.isclose
    mindspore.mint.le
    mindspore.mint.less
    mindspore.mint.less_equal
    mindspore.mint.lt

BLAS和LAPACK运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.inverse

其他运算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.searchsorted

Reduction函数
^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.argmax

mindspore.mint.nn
------------------

Dropout层
^^^^^^^^^^^

.. mscnplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Dropout





卷积层
^^^^^^

.. mscnplatformautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.Fold
    mindspore.mint.nn.Unfold


损失函数
^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.BCEWithLogitsLoss


mindspore.mint.nn.functional
-----------------------------

神经网络层函数
^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.batch_norm
    mindspore.mint.nn.functional.dropout
    mindspore.mint.nn.functional.grid_sample
    mindspore.mint.nn.functional.linear

卷积函数
^^^^^^^^^^

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

    mindspore.mint.nn.functional.max_pool2d






注意力机制
^^^^^^^^^^^^^^^^^^^







非线性激活函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy
    mindspore.mint.nn.functional.elu
    mindspore.mint.nn.functional.gelu
    mindspore.mint.nn.functional.leaky_relu
    mindspore.mint.nn.functional.sigmoid
    mindspore.mint.nn.functional.silu
    mindspore.mint.nn.functional.softmax
    mindspore.mint.nn.functional.softplus
    mindspore.mint.nn.functional.tanh




线性函数
^^^^^^^^^^^^^^^^^^^







Dropout函数
^^^^^^^^^^^^^^^^^^^







距离函数
^^^^^^^^^^^^^^^^^^^





损失函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

    mindspore.mint.nn.functional.binary_cross_entropy_with_logits






视觉函数
^^^^^^^^^^^^^^^^^^^

.. mscnplatwarnautosummary::
    :toctree: mint
    :nosignatures:
    :template: classtemplate.rst

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
