mindspore.Tensor.asinh
======================

.. py:method:: mindspore.Tensor.asinh()

    逐元素计算输入张量的反双曲正弦值。

    .. math::
        out_i = \sinh^{-1}(input_i)

    返回：
        Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。