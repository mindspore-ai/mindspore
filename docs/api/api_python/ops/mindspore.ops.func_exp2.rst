mindspore.ops.exp2
==================

.. py:function:: mindspore.ops.exp2(input)

    逐元素计算Tensor `input` 以2为底的指数。

    .. math::
        out_i = 2^{input_i}

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，具有与 `input` 相同的数据类型和shape。

    异常：
        - **TypeError** - `input` 不是Tensor。
