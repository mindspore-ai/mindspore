mindspore.ops.Log1p
===================

.. py:class:: mindspore.ops.Log1p

    对输入Tensor逐元素加一后计算自然对数。

    .. math::
        out_i = {log_e}(x_i + 1)

    **输入：**
    
    - **x** (Tensor) - 输入Tensor。数据类型为float16或float32。
      该值必须大于-1。
      shape： :math:`(N,*)` 其中 :math:`*` 表示任何数量的附加维度，其轶应小于8。

    **输出：**
    
    Tensor，与 `x` 的shape相同。

    **异常：**
    
    - **TypeError** - `x` 不是Tensor。
    - **TypeError** - `x` 的数据类型非float16或float32。
