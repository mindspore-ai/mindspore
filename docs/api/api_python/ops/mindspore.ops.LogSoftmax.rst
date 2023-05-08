mindspore.ops.LogSoftmax
=========================

.. py:class:: mindspore.ops.LogSoftmax(axis=-1)

    LogSoftmax激活函数。

    更多参考详见 :func:`mindspore.ops.log_softmax`。

    参数：
        - **axis** (int，可选) - 指定进行运算的轴。默认值： ``-1`` 。

    输入：
        - **logits** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，其数据类型为float16或float32。

    输出：
        Tensor，数据类型和shape与 `logits` 相同。
