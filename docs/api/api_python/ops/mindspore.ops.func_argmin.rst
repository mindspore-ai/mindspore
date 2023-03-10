mindspore.ops.argmin
====================

.. py:function:: mindspore.ops.argmin(input, axis=None, keepdims=False)

    返回输入Tensor在指定轴上的最小值索引。

    如果输入Tensor的shape为 :math:`(x_1, ..., x_N)` ，则输出Tensor的shape为 :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **axis** (Union[int, None]，可选) - 指定计算轴。如果是None，将会返回扁平化Tensor在指定轴上的最小值索引。默认值：None。
        - **keepdims** (bool，可选) - 输出Tensor是否保留指定轴。如果 `axis` 是None，忽略该选项。默认值： False。

    返回：
        Tensor，输出为指定轴上输入Tensor最小值的索引。

    异常：
        - **TypeError** - `axis` 不是int。
