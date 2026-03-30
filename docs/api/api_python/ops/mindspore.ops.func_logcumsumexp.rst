mindspore.ops.logcumsumexp
==========================

.. py:function:: mindspore.ops.logcumsumexp(input, axis)

    计算tensor在指定维度上的累积指数和的对数。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：参数 `axis` 将重命名为 `dim` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定计算轴。

    返回：
        Tensor
