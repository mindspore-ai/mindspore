mindspore.nn.Threshold
=============================

.. py:class:: mindspore.nn.Threshold(threshold, value)

    逐元素计算Threshold激活函数。

    Threshold定义为：

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    .. warning::
        从2.9.0（不含）之后版本开始，前向输入 `input_x` 将改为 `input`，并新增可选构造参数 `inplace`。
        接口签名将变更为 ``mindspore.nn.Threshold(threshold, value, inplace=False)`` 。

    参数：
        - **threshold** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 输入Tensor中元素小于阈值时的填充值。

    输入：
        - **input_x** (Tensor) - 输入Tensor，数据类型为float16或float32。

    输出：
        Tensor，数据类型和shape与 `input_x` 的相同。

    异常：
        - **TypeError** - `threshold` 不是浮点数或整数。
        - **TypeError** - `value` 不是浮点数或整数。
