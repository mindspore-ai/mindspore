mindspore.Tensor.cauchy
========================

.. py:method:: mindspore.Tensor.cauchy(median=0.0, sigma=1.0)

    使用柯西分布生成的数值填充当前Tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. math::
        f(x)= \frac{1}{\pi} \frac{\sigma}{(x-median)^2 +\sigma^2}

    参数：
        - **median** (float, 可选) - 柯西分布中定位分布峰值位置的位置参数。默认值： ``0.0``。
        - **sigma** (float, 可选) - 柯西分布中最大值一半处的一半宽度的尺度参数。默认值： ``1.0``。

    返回：
        Tensor，具有与当前Tensor相同的shape和dtype。
