mindspore.ops.clip_by_value
============================

.. py:function:: mindspore.ops.clip_by_value(x, clip_value_min=None, clip_value_max=None)

    将输入Tensor值裁剪到指定的最小值和最大值之间。

    限制 :math:`x` 的范围，其最小值为 `clip_value_min` ，最大值为 `clip_value_max` 。

    .. math::
        out_i= \left\{
        \begin{array}{align}
            clip\_value\_max & \text{ if } x_i\ge  clip\_value\_max \\
            x_i & \text{ if } clip\_value\_min \lt x_i \lt clip\_value\_max \\
            clip\_value\_min & \text{ if } x_i \le clip\_value\_min \\
        \end{array}\right.

    .. note::
        - `clip_value_min` 和 `clip_value_max` 不能同时为None；
        - 当 `clip_value_min` 为None，`clip_value_max` 不为None时，Tensor中大于 `clip_value_max` 的元素会变为 `clip_value_max`；
        - 当 `clip_value_min` 不为None，`clip_value_max` 为None时，Tensor中小于 `clip_value_min` 的元素会变为 `clip_value_min`；
        - 当 `clip_value_min` 大于 `clip_value_max` 时，Tensor中所有元素的值会被置为 `clip_value_max`；
        - `x`，`clip_value_min` 和 `clip_value_max` 的数据类型需支持隐式类型转换，且不能为布尔型。

    参数：
        - **x** (Union(Tensor, list[Tensor], tuple[Tensor])) - `clip_by_value` 的输入，类型为Tensor、Tensor的列表或元组。支持任意维度的Tensor。
        - **clip_value_min** (Union(Tensor, float, int)) - 指定最小值。默认值为 ``None`` 。
        - **clip_value_max** (Union(Tensor, float, int)) - 指定最大值。默认值为 ``None`` 。

    返回：
        Tensor、Tensor的列表或元组，表示裁剪后的Tensor。其shape和数据类型和 `x` 相同。
    
    异常：
        - **ValueError** - 如果 `clip_value_min` 和 `clip_value_max` 都为None。
        - **TypeError** - 如果 `x` 的数据类型不在Tensor、list[Tensor]或tuple[Tensor]中。
        - **TypeError** - 如果 `clip_value_min` 的数据类型不为None、Tensor、float或int。
        - **TypeError** - 如果 `clip_value_max` 的数据类型不为None、Tensor、float或int。
