mindspore.ops.cummax
====================

.. py:function:: mindspore.ops.cummax(input, axis)

    返回tensor在指定轴上的累积最大值及其索引。

    .. math::
        \begin{array}{ll} \\
            y_{i} = \max(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `axis` 将重命名为 `dim` 。

        变更后的接口签名为 ``mindspore.ops.cummax(input, dim)`` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定计算的轴。

    返回：
        两个tensor组成的tuple(max, max_indices)
