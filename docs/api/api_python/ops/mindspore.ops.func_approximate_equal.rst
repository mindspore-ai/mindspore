mindspore.ops.approximate_equal
===============================

.. py:function:: mindspore.ops.approximate_equal(x, y, tolerance=1e-5)

    `ops.approximate_equal` is deprecated from version 2.8.0 and will be removed in a future version,
    please use :func:`mindspore.ops.allclose` instead.

    返回一个布尔型tensor，表示两个tensor在容忍度内是否逐元素相等。

    支持隐式类型转换、类型提升。

    数学公式为：

    .. math::
        out_i = \begin{cases}
        & \text{ if } \left | x_{i} - y_{i} \right | < \text{tolerance},\ \ True  \\
        & \text{ if } \left | x_{i} - y_{i} \right | \ge \text{tolerance},\ \  False
        \end{cases}

    两个inf值和两个NaN值均不被认为相等。

    参数：
        - **x** (Tensor) - 第一个输入tensor。
        - **y** (Tensor) - 第二个输入tensor。
        - **tolerance** (float) - 两个元素被视为相等的最大偏差。默认 ``1e-5`` 。

    返回：
        Tensor
