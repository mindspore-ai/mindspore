mindspore.ops.approximate_equal
===============================

.. py:class:: mindspore.ops.approximate_equal(x, y, tolerance=1e-05)

    逐元素计算abs(x-y)，如果小于tolerance则为True，否则为False。
    
    .. math::
        out_i = \begin{cases}
        & \text{ if } \left | x_{i} - y_{i} \right | < \text{tolerance},\ \ True  \\
        & \text{ if } \left | x_{i} - y_{i} \right | \ge \text{tolerance},\ \  False
        \end{cases}

    `tolerance` 为相等的两元素间最大偏差。
    输入 `x` 和 `y` 会通过隐式数据类型转换使数据类型保持一致。如果数据类型不同，低精度的数据类型会被自动转换到高精度的数据类型。

    参数：
        - **x** (Tensor) - 输入Tensor，需为以下数据类型：float16，float32。shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其秩应小于8。
        - **y** (Tensor) - 输入Tensor，shape与数据类型与 `x` 相同。
        - **tolerance** (float) - 两元素可被视为相等的最大偏差。默认值：1e-05。

    输出：
        Tensor，shape与 `x` 相同，bool类型。

    异常：
        - **TypeError** - `tolerance` 不是float类型。
        - **RuntimeError** - `x` 与 `y` 之间的类型转换不被支持。
