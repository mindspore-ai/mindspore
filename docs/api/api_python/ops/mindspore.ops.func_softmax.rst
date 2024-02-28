mindspore.ops.softmax
=====================

.. py:function:: mindspore.ops.softmax(input, axis=-1, *, dtype=None)

    在指定轴上对输入Tensor执行Softmax激活函数做归一化操作。假设指定轴 :math:`axis` 上有切片，那么每个元素 :math:`input_i` 所对应的Softmax函数如下所示：

    .. math::
        \text{output}(input_i) = \frac{\exp(input_i)}{\sum_{j = 0}^{N-1}\exp(input_j)},

    其中 :math:`N` 代表Tensor的长度。

    参数：
        - **input** (Tensor) - Softmax的输入，shape为 :math:`(N, *)` ，其中 :math:`*` 为任意额外维度。
        - **axis** (int，可选) - 指定Softmax操作的轴。默认值： ``-1`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 如果设置此参数，则会在执行之前将 `input` 转换为指定的类型，返回的Tensor类型也将为指定类型 `dtype`。默认值： ``None`` 。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `axis` 不是int。
