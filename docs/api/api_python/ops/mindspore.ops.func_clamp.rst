mindspore.ops.clamp
====================

.. py:function:: mindspore.ops.clamp(input, min=None, max=None)

    将输入Tensor的值裁剪到指定的最小值和最大值之间。

    限制 :math:`input` 的范围，其最小值为 `min` ，最大值为 `max` 。

    .. math::
        out_i= \left\{
        \begin{array}{align}
            max & \text{ if } x_i\ge max \\
            x_i & \text{ if } min \lt x_i \lt max \\
            min & \text{ if } x_i \le min \\
        \end{array}\right.

    .. note::
        - `min` 和 `max` 不能同时为None；
        - 当 `min` 为None，`max` 不为None时，Tensor中大于 `max` 的元素会变为 `max`；
        - 当 `min` 不为None，`max` 为None时，Tensor中小于 `min` 的元素会变为 `min`；
        - 当 `min` 大于 `max` 时，Tensor中所有元素的值会被置为 `max`；
        - `input`，`min` 和 `max` 的数据类型需支持隐式类型转换，且不能为布尔型。

    参数：
        - **input** (Union(Tensor, list[Tensor], tuple[Tensor])) - `clamp` 的输入，类型为Tensor、Tensor的列表或元组。支持任意维度的Tensor。
        - **min** (Union(Tensor, float, int)) - 指定最小值。默认值为None。
        - **max** (Union(Tensor, float, int)) - 指定最大值。默认值为None。

    返回：
        Tensor、Tensor的列表或元组，表示裁剪后的Tensor。其shape和数据类型和 `input` 相同。

    异常：
        - **ValueError** - 如果 `min` 和 `max` 都为None。
        - **TypeError** - 如果 `input` 的数据类型不在Tensor、list[Tensor]或tuple[Tensor]中。
        - **TypeError** - 如果 `min` 的数据类型不为None、Tensor、float或int。
        - **TypeError** - 如果 `max` 的数据类型不为None、Tensor、float或int。
