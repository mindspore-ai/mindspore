mindspore.mint.nn.Threshold
===========================

.. py:class:: mindspore.mint.nn.Threshold(threshold, value, inplace=False)

    逐元素计算Threshold激活函数。

    Threshold定义为：

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    参数：
        - **threshold** (Union[int, float]) - 阈值。
        - **value** (Union[int, float]) - 小于阈值时的填充值。
        - **inplace** (bool, 可选) - 是否启用原地更新功能。默认 ``False`` 。

    输入：
        - **input** (Tensor) - 输入tensor。

    输出：
        Tensor。
