mindspore.ops.Asinh
===================

.. py:class:: mindspore.ops.Asinh

    逐元素计算输入Tensor的反双曲正弦值。

    .. math::
        out_i = \sinh^{-1}(x_i)

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N,*)` 的输入Tensor，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。

    **输出：**

    Tensor，shape和数据类型与输入 `x` 相同。

    **异常：**
    
    - **TypeError** - `x` 不是Tensor。
