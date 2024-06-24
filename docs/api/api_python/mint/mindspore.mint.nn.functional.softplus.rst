mindspore.mint.nn.functional.softplus
=====================================

.. py:function:: mindspore.mint.nn.functional.softplus(input, beta=1, threshold=20)

    将softplus函数作用于 `input` 的每个元素上。

    softplus函数如下所示，其中x是 `input` 中的元素：

    .. math::

        \text{output} = \frac{1}{beta}\log(1 + \exp(\text{beta * x}))

    当 :math:`input * beta > threshold` 时，为保证数值稳定性，softplus的实现被转换为线性函数。

    参数：
        - **input** (Tensor) - 任意维度的输入Tensor。支持数据类型：

          - Ascend：float16、float32、bfloat16。

        - **beta** (number.Number，可选) - softplus函数中的缩放参数。默认值：``1`` 。
        - **threshold** (number.Number，可选) - 为了数值稳定性，softplus函数转换为线性函数的阈值参数。默认值：``20`` 。

    返回：
        Tensor，其数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型非float16、float32、bfloat16。
