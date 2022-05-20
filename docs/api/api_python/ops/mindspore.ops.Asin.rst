mindspore.ops.Asin
==================

.. py:class:: mindspore.ops.Asin

    逐元素计算输入Tensor的反正弦值。

    .. math::
        out_i = sin^{-1}(x_i)

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N,*)` 的输入Tensor，其中 :math:`*` 表示任何数量的附加维度。数据类型可为以下类型：float16、float32或float64。

    **输出：**

    Tensor，shape与输入 `x` 相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。
    - **TypeError** - `x` 的数据类型不是float16、float32或float64。
