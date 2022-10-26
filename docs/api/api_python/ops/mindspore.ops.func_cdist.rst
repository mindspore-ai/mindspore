mindspore.ops.cdist
===================

.. py:function:: mindspore.ops.cdist(x, y, p=2.0)

    批量计算两个Tensor每一批次所有向量两两之间的p-范数距离。

    参数：
        - **x** (Tensor) - 输入tensor x，输入shape [B, P, M]，B维度可以为0，即shape为 [P, M]。
        - **y** (Tensor) - 输入tensor y，输入shape [B, R, M]。
        - **p** (float) - P -范数距离的P值，P∈[0，∞]。默认值:2.0。

    返回：
        Tensor，p-范数距离，shape为[B, P, R]。

    异常：
        - **TypeError** - `input_x` 或 `input_x` 不是Tensor。
        - **TypeError** - `input_x` 或 `input_y` 的数据类型不是float16，也不是float32。
        - **TypeError** - `p` 不是float32。
        - **ValueError** - `p` 是负数。
        - **ValueError** - `input_x` 与 `input_y` 维度不同。
        - **ValueError** - `input_x` 与 `input_y` 的维度不是2，也不是3。
        - **ValueError** - 单批次训练下 `x` 和 `y` 的shape不一样。
        - **ValueError** - `x` 和 `y` 的列数不一样。
