mindspore.ops.masked_select
===========================

.. py:function:: mindspore.ops.masked_select(x, mask)

    返回一个一维张量，其中的内容是input张量中对应于mask张量中True位置的值。mask的shape与input的shape不需要一样，但必须符合广播规则。

    **参数：**

    - **x** (Tensor) - 它的shape是 :math:`(x_1, x_2, ..., x_R)`。
    - **mask** (Tensor[bool]) - 它的shape是 :math:`(x_1, x_2, ..., x_R)`。

    **返回：**

    一个一维Tensor, 类型与input相同。

    **异常：**

    - **TypeError** - mask不是bool类型的Tensor。