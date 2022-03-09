mindspore.ops.clip_by_value
============================

.. py:function:: mindspore.ops.clip_by_value(x, clip_value_min, clip_value_max)

    将输入Tensor值裁剪到指定的最小值和最大值之间。

    限制 :math:`x` 的范围，其 :math:`x` 的最小值为'clip_value_min'，最大值为'clip_value_max'。

    .. math::
        out_i= \left\{
        \begin{array}{align}
            clip\_value_{max} & \text{ if } x_i\ge  clip\_value_{max} \\
            x_i & \text{ if } clip\_value_{min} \lt x_i \lt clip\_value_{max} \\
            clip\_value_{min} & \text{ if } x_i \le clip\_value_{min} \\
        \end{array}\right.

    .. note::
        'clip_value_min'必须小于或等于'clip_value_max'。

    **参数：**

    - **x** (Tensor) - clip_by_value的输入，任意维度的Tensor。
    - **clip_value_min** (Tensor) - 指定最小值。
    - **clip_value_max** (Tensor) - 指定最大值。

    **返回：**

    Tensor，表示裁剪后的Tensor。其shape和数据类型和 `x` 相同。