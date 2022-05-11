mindspore.nn.Hardtanh
=============================

.. py:class:: mindspore.nn.Hardtanh

    Hardtanh激活函数。

    按元素计算Hardtanh函数。Hardtanh函数定义为：

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1, & \text{ if } x > 1; \\
            -1, & \text{ if } x < -1; \\
            x, & \text{ otherwise. }
        \end{cases}

    线性区域范围 :math:`[-1, 1]` 可以使用 `min_val` 和 `max_val` 进行调整。

    **参数：**
    
    - **min_val** (Union[int, float]) - 线性区域范围的最小值。 默认值：-1.0。
    - **max_val** (Union[int, float]) - 线性区域范围的最大值。 默认值：1.0。

    **输入：**
    
    - **x** (Tensor) - 任意维度的Tensor，数据类型为float16或float32。shape是 :math:`(N,*)` ， :math:`*` 表示任意的附加维度数。

    **输出：**
    
    Tensor，数据类型和shape与 `x` 的相同。

    **异常：**
    
    **TypeError** - `x` 的数据类型既不是float16也不是float32。
    **TypeError** - `min_val` 的数据类型既不是int也不是float。
    **TypeError** - `max_val` 的数据类型既不是int也不是float。
    **ValueError** - `max_val` 小于 `min_val` 。
