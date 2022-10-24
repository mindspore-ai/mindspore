mindspore.Tensor.cumprod
========================

.. py:method:: mindspore.Tensor.cumprod(dim, dtype=None)

    返回输入的元素在 `dim` 维度上的累积乘积。
    例如，如果当前输入是大小为N的向量，则结果也将是大小为N的向量（带有元素）。

    .. math::
        y_i = x_1 * x_2 * x_3 * ... * x_i

    参数：
        - **dim** (int) - 计算累积乘积的尺寸。只允许常量值。
        - **dtype** - 输出的数据类型。默认值：None。

    返回：
        Tensor，数据类型和shape与当前Tensor相同，除非指定了 `dtype`。

    异常：
        - **TypeError** -  如果 `dim` 不是int。
        - **TypeError** -  如果 `dtype` 无法进行转换。
        - **ValueError** - 如果 `dim` 是None。
