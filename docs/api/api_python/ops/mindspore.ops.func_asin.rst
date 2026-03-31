mindspore.ops.asin
===================

.. py:function:: mindspore.ops.asin(input)

    逐元素计算输入tensor的反正弦。

    .. warning::
        从2.9.0（不含）之后版本开始，参数命名将统一为 `input`，历史别名 `x` 将不再作为文档入口。

    .. math::
        out_i = \sin^{-1}(input_i)

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
