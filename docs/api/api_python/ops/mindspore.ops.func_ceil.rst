mindspore.ops.ceil
===================

.. py:function:: mindspore.ops.ceil(x)

    向上取整函数。

    .. math::
        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    参数：
        - **x** (Tensor) - Ceil的输入。其数据类型为float16或float32。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。

    返回：
        Tensor，shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
