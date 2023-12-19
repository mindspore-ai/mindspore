mindspore.nn.Hardtanh
=============================

.. py:class:: mindspore.nn.Hardtanh(min_val=-1.0, max_val=1.0)

    逐元素计算Hardtanh函数。Hardtanh函数定义为：

    .. math::
        \text{Hardtanh}(x) = \begin{cases}
            1, & \text{ if } x > 1; \\
            -1, & \text{ if } x < -1; \\
            x, & \text{ otherwise. }
        \end{cases}

    线性区域范围 :math:`[-1, 1]` 可以使用 `min_val` 和 `max_val` 进行调整。

    Hardtanh函数图：

    .. image:: ../images/Hardtanh.png
        :align: center

    .. note::
        在Ascend硬件上，float16数据类型场景下会有偶现的精度误差较大的问题。

    参数：
        - **min_val** (Union[int, float]) - 线性区域范围的最小值。默认值： ``-1.0`` 。
        - **max_val** (Union[int, float]) - 线性区域范围的最大值。默认值： ``1.0`` 。

    输入：
        - **x** (Tensor) - 数据类型为float16或float32的Tensor。在CPU和Ascend平台上支持零到七维。在GPU平台上支持零到四维。

    输出：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **TypeError** - `min_val` 的数据类型既不是int也不是float。
        - **TypeError** - `max_val` 的数据类型既不是int也不是float。
        - **ValueError** - `min_val` 不小于 `max_val` 。
