mindspore.ops.argmin
====================

.. py:function:: mindspore.ops.argmin(x, axis=-1)

    返回输入Tensor在指定轴上的最小值索引。

    如果输入Tensor的shape为 :math:`(x_1, ..., x_N)` ，则输出Tensor的shape为 :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。

    参数：
        - **x** (Tensor) - shape非空，任意维度的Tensor。
        - **axis** (int) - 指定计算轴。默认值：-1。

    返回：
        Tensor，输出为指定轴上输入Tensor最小值的索引。

    异常：
        - **TypeError** - `axis` 不是int。
