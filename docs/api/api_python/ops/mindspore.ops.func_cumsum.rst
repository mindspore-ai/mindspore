mindspore.ops.cumsum
====================

.. py:function:: mindspore.ops.cumsum(x, axis, dtype=None)

    返回tensor在指定轴上累积的元素和。

    .. math::
        y_i = x_1 + x_2 + x_3 + ... + x_i

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `x` 将重命名为 `input` 。
        - 参数 `axis` 将重命名为 `dim` 。

        变更后的接口签名为 ``mindspore.ops.cumsum(input, dim, dtype=None)`` 。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **axis** (int) - 指定计算的轴。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型。默认 ``None`` 。
    返回：
        Tensor
