mindspore.ops.vecdot
====================

.. py:function:: mindspore.ops.vecdot(x, y, *, axis=-1)

    在指定维度上，计算两批向量的点积。

    计算公式如下，
    如果 `x` 是复数向量，:math:`\bar{x_{i}}` 表示向量中元素的共轭；如果 `x` 是实数向量，:math:`\bar{x_{i}}` 表示向量中元素本身。

    .. math::

        \sum_{i=1}^{n} \bar{x_{i}}{y_{i}}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (Tensor) - 进行点积运算的第一批向量。它的shape是 :math:`(*,N)` ，其中 :math:`*` 表示任意额外的维度。它支持广播。
        - **y** (Tensor) - 进行点积运算的第二批向量。它的shape是 :math:`(*,N)` ，其中 :math:`*` 表示任意额外的维度。它支持广播。
        - **axis** (int) - 进行点积运算的维度。默认值： ``-1`` 。

    返回：
        Tensor，它的shape与广播后得到的Tensor的shape几乎相同，但删掉了指定维度 `axis` 。

    异常：
        - **TypeError** - `x` 或 `y` 不是Tensor。
        - **TypeError** - `axis` 的数据类型不是int。
        - **ValueError** - `axis` 超出范围。

    .. note::
        当前在GPU上不支持复数。
