mindspore.ops.clamp
====================

.. py:function:: mindspore.ops.clamp(input, min=None, max=None)

    将输入tensor的所有元素限制在范围 [min, max] 内。

    .. math::
        out_i= \left\{
        \begin{array}{ll}
            max & \text{ if } input_i\ge max \\
            input_i & \text{ if } min \lt input_i \lt max \\
            min & \text{ if } input_i \le min \\
        \end{array}\right.

    .. note::
        - `min` 和 `max` 不能同时为None；
        - 当 `min` 为 ``None`` 时，无最小值限制；
        - 当 `max` 为 ``None`` 时，无最大值限制；
        - 当 `min` 大于 `max` 时，Tensor中所有元素的值会被置为 `max`；

    参数：
        - **input** (Tensor) - 输入tensor。
        - **min** (Union(Tensor, float, int)，可选) - 指定最小值。默认 ``None`` 。
        - **max** (Union(Tensor, float, int)，可选) - 指定最大值。默认 ``None`` 。

    返回：
        Tensor