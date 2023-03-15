mindspore.ops.threshold
=======================

.. py:function:: mindspore.ops.threshold(input, thr, value)

    将使用 `thr` 参数对 `input` 逐元素阈值化后的结果作为Tensor返回。

    threshold定义为：

    .. math::
        y =
        \begin{cases}
        input, &\text{ if } input > \text{thr} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    参数：
        - **input** (Tensor) - 输入Tensor，数据类型为float16或float32。
        - **thr** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 输入Tensor中element小于阈值时的填充值。

    返回：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `thr` 不是浮点数或整数。
        - **TypeError** - `value` 不是浮点数或整数。
