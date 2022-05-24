mindspore.ops.cdist
===================

.. py:function:: mindspore.ops.cdist(x, y, p=2.0)

    计算两个tensor的p-范数距离。

    **参数：**

    - **x** (tensor) - 输入tensor x，输入shape [B, P, M]。
    - **y** (tensor) - 输入tensor y，输入shape [B, R, M]。
    - **p** (float) - P -范数距离的P值，P∈[0，∞]。默认值:2.0。

    **返回：**

    Tensor，p-范数距离, shape为[B, R, M]。

    **异常：**

    - **TypeError** - `input_x` 或 `input_x` 不是tensor。
    - **TypeError** - `input_x` 或 `input_y` 的数据类型不是float16，也不是float32。
    - **TypeError** - `p` 不是float32。
    - **ValueError** - `p` 是负数。
    - **ValueError** - `input_x` 与 `input_y` 维度不同。
    - **ValueError** - `input_x` 与 `input_y` 不是2，，也不是3。
