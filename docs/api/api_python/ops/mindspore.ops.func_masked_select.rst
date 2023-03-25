mindspore.ops.masked_select
===========================

.. py:function:: mindspore.ops.masked_select(input, mask)

    返回一个一维Tensor，其中的内容是 `input` 中对应于 `mask` 中True位置的值。`mask` 的shape与 `input` 的shape不需要一样，但必须符合广播规则。

    参数：
        - **input** (Tensor) - 它的shape是 :math:`(x_1, x_2, ..., x_R)`。
        - **mask** (Tensor[bool]) - 它的shape是 :math:`(x_1, x_2, ..., x_R)`。

    返回：
        一个一维Tensor，类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 或 `mask` 不是Tensor。
        - **TypeError** - `mask` 不是bool类型的Tensor。