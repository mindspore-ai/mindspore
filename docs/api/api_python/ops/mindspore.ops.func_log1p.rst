mindspore.ops.log1p
===================

.. py:function:: mindspore.ops.log1p(input)

    对输入Tensor逐元素加一后计算自然对数。

    .. math::
        out_i = {log_e}(x_i + 1)

    参数：
        - **input** (Tensor) - 输入Tensor。数据类型为float16或float32。
          该值必须大于-1。
          shape： :math:`(N,*)` 其中 :math:`*` 表示任何数量的附加维度，其秩应小于8。

    返回：
        Tensor，与 `input` 的shape相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型非float16或float32。
