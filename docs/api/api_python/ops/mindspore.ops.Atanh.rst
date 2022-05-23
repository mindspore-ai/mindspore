mindspore.ops.Atanh
===================

.. py:class:: mindspore.ops.Atanh

    逐元素计算输入Tensor的反双曲正切值。

    .. math::
        out_i = \tanh^{-1}(x_{i})

    **输入：**

    - **x** (Tensor): 输入Tensor，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其轶应小于8。数据类型可为以下类型：float16或float32。

    **输出：**

    Tensor，shape和数据类型与输入 `x` 相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。
    - **TypeError** - `x` 的数据类型非float16或float32。
