mindspore.ops.vecdot
====================

.. py:function:: mindspore.ops.vecdot(x, y, *, axis=-1)

    按指定轴，计算两批向量的点积。

    .. warning::
        - 从2.9.0（不含）之后版本开始，关键字参数 `axis` 将重命名为 `dim`。
        - 这是一个实验性API，后续可能修改或删除。

    支持广播。

    计算公式如下，
    如果 `x` 是复数向量，:math:`\bar{x_{i}}` 表示向量中元素的共轭；如果 `x` 是实数向量，:math:`\bar{x_{i}}` 表示向量中元素本身。

    .. math::

        \sum_{i=1}^{n} \bar{x_{i}}{y_{i}}

    参数：
        - **x** (Tensor) - 第一批tensor。
        - **y** (Tensor) - 第二批tensor。

    关键字参数：
        - **axis** (int) - 指定轴。默认 ``-1`` 。

    返回：
        Tensor

    .. note::
        当前在GPU上不支持复数。
