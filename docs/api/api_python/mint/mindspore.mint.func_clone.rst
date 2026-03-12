mindspore.mint.clone
====================

.. py:function:: mindspore.mint.clone(input)

    返回一个输入tensor的副本。

    .. note::
        该函数是可微分的，梯度将直接从该函数的计算结果流向 `input`。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor，其数据、shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
