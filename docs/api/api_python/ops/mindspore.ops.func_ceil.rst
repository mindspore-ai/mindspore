mindspore.ops.ceil
===================

.. py:function:: mindspore.ops.ceil(input)

    向上取整函数。

    .. math::
        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    参数：
        - **input** (Tensor) - Ceil的输入。其数据类型为float16或float32。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。秩应小于8。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型既不是float16也不是float32。
