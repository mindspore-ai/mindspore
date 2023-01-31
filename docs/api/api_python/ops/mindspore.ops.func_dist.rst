mindspore.ops.dist
====================

.. py:function:: mindspore.ops.dist(input_x, input_y, p=2)

    计算输入中每对行向量之间的p-范数距离。

    .. note::
        在MindSpore中只支持计算整数 :math:`p`-norm形式的范数，如果 :math:`p` 不是整数会引发类型错误。

    参数：
        - **input_x** (Tensor) - 第一个输入Tensor，数据类型需为float16或float32。
        - **input_y** (float) - 第二个输入Tensor，数据类型需为float1或float32。
        - **p** (int，可选) - 范数的次数。 `p` 大于或等于0。默认值：2。

    返回：
        Tensor，具有与 `input_x` 相同的dtype，其shape为：:math:`(1)` 。

    异常：
        - **TypeError** -  `input_x` 或 `input_y` 不是Tensor。
        - **TypeError** - `input_x` 和 `input_y` 数据类型不是float16或float32。
        - **TypeError** - `p` 不是非负整数。
