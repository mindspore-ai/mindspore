mindspore.mint.exp2
=============================

.. py:function:: mindspore.mint.exp2(input)

    逐元素计算tensor `input` 以2为底的指数。

    .. math::
        out_i = 2^{\text{input}_i}

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor，其shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
