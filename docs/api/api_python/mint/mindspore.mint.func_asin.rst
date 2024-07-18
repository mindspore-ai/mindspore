mindspore.mint.asin
===================

.. py:function:: mindspore.mint.asin(input)

    逐元素计算输入Tensor的反正弦。

    .. math::
        out_i = \sin^{-1}(input_i)

    参数：
        - **input** (Tensor) - Tensor的shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
