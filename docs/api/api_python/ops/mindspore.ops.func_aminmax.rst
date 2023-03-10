mindspore.ops.aminmax
======================

.. py:function:: mindspore.ops.aminmax(input, *, axis=0, keepdims=False)

    返回输入Tensor在指定轴上的最小值和最大值。

    参数：
        - **input** (Tensor) - 输入Tensor，可以是任意维度。设输入Tensor的shape为 :math:`(x_1, x_2, ..., x_N)` 。

    关键字参数：
        - **axis** (int，可选) - 要进行规约计算的维度。 `axis` 必须在[-rank, rank)范围内，其中 “rank” 是 `input` 的维度。默认值：0。
        - **keepdims** (bool，可选) - 是否保留维度。如果为True，则输出shape与输入shape一致，否则移除规约计算的维度 `axis` 。默认值：False。

    返回：
        tuple (Tensor)，包含输入Tensor在指定轴上的最小值和最大值。

        - `keepdims` 为True，输出shape为： :math:`(x_1, x_2, ..., x_{axis-1}, 1, x_{axis+1}, ..., x_N)` 。
        - `keepdims` 为False，输出shape为： :math:`(x_1, x_2, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。

    异常：
        - **TypeError** - `keepdims` 不是bool类型。
        - **TypeError** - `axis` 不是int类型。
        - **ValueError** - `axis` 不在[-rank, rank)范围内。
