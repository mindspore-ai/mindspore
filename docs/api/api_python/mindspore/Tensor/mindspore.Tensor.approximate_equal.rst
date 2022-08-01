mindspore.Tensor.approximate_equal
==================================

.. py:method:: mindspore.Tensor.approximate_equal(y, tolerance=1e-5)

    逐元素计算abs(x-y)，如果小于tolerance则为True，否则为False。
    
    .. math::
        out_i = \begin{cases}
        & \text{ if } \left | x_{i} - y_{i} \right | < \text{tolerance},\ \ True  \\
        & \text{ if } \left | x_{i} - y_{i} \right | \ge \text{tolerance},\ \  False
        \end{cases}

    `x` 为当前Tensor。
    `tolerance` 为相等的两元素间最大偏差。
    `x` 和 `y` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被自动转换到高精度的数据类型。

    参数：
        - **y** (Tensor) - 输入Tensor，shape与数据类型和当前Tensor相同。
        - **tolerance** (float) - 两元素可被视为相等的最大偏差。默认值：1e-05。

    返回：
        Tensor，shape与当前Tensor相同，bool类型。

    异常：
        - **TypeError** - `tolerance` 不是float类型。
        - **RuntimeError** - `variable` 与 `value` 之间的类型转换不被支持。