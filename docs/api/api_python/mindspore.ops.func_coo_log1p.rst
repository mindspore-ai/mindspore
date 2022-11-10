mindspore.ops.coo_log1p
========================

.. py:function:: mindspore.ops.coo_log1p(x: COOTensor)

    对输入COOTensor逐元素加一后计算自然对数。

    .. math::
        out_i = {log_e}(x_i + 1)

    参数：
        - **x** (COOTensor) - 输入COOTensor。数据类型为float16或float32。
          该值必须大于-1。
          shape： :math:`(N,*)` 其中 :math:`*` 表示任何数量的附加维度，其秩应小于8。

    返回：
        COOTensor，与 `x` 的shape相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
        - **TypeError** - `x` 的数据类型非float16或float32。
