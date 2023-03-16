mindspore.ops.ger
=================

.. py:function:: mindspore.ops.ger(input, vec2)

    计算输入一维Tensor `input` 与 `vec2` 的外积。如果 `input` shape为 :math:`(m,)` ，`vec2` shape为 :math:`(n,)` ，则输出的shape为 :math:`(m, n)` 。

    .. note::
        Ascend不支持float64数据格式的输入。

    参数：
        - **input** (Tensor) - 输入1-D Tensor，数据类型为float16、float32或float64。
        - **vec2** (Tensor) - 输入1-D Tensor，数据类型为float16、float32或float64，输入数据类型需和 `input` 保持一致。

    返回：
        Tensor，与 `input` 数据类型相同的输出Tensor。如果 `input` shape为 :math:`(m,)` ，`vec2` shape为 :math:`(n,)` ，则输出的shape为 :math:`(m, n)` 。

    异常：
        - **TypeError** - `input` 或 `vec2` 不是一维Tensor。
        - **TypeError** - 输入的 `input` 与 `vec2` 数据类型不是float16、float32或float64。
        - **TypeError** - 输入的 `input` 与 `vec2` 数据类型不一致。
