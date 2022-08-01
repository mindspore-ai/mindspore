mindspore.Tensor.masked_select
==============================

.. py:method:: mindspore.Tensor.masked_select(mask)

    返回一个一维张量，其中的内容是此张量中对应于 `mask` 张量中True位置的值。`mask` 张量的shape与此张量的shape不需要一样，但必须符合广播规则。

    参数：
        - **mask** (Tensor[bool]) - 值为bool类型的张量。

    返回：
        一个一维张量，类型与此张量相同。

    异常：
        - **TypeError** - `mask` 不是bool类型的Tensor。