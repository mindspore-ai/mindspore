mindspore.mint.nn.functional.threshold\_
========================================

.. py:function:: mindspore.mint.nn.functional.threshold_(input, threshold, value)

    通过逐元素计算 Threshold 激活函数，原地更新 `input` Tensor。

    Threshold定义为：

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{otherwise}
        \end{cases}

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **threshold** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 小于阈值时的填充值。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `threshold` 不是浮点数或整数。
        - **TypeError** - `value` 不是浮点数或整数。
