mindspore.ops.hardtanh
======================

.. py:function:: mindspore.ops.hardtanh(input, min_val=-1.0, max_val=1.0)

    逐元素元素计算hardtanh激活函数。hardtanh函数定义为：

    .. math::
        \text{hardtanh}(input) = \begin{cases}
            max\_val, & \text{ if } input > max\_val \\
            min\_val, & \text{ if } input < min\_val \\
            input, & \text{ otherwise. }
        \end{cases}

    线性区域范围 :math:`[min\_val, max\_val]` 可以使用 `min_val` 和 `max_val` 进行调整。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **min_val** (Union[int, float]) - 线性区域范围的最小值。默认值： ``-1.0`` 。
        - **max_val** (Union[int, float]) - 线性区域范围的最大值。默认值： ``1.0`` 。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `min_val` 的数据类型既不是int也不是float。
        - **TypeError** - `max_val` 的数据类型既不是int也不是float。
