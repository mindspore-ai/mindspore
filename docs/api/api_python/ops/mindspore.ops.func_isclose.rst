mindspore.ops.isclose
=====================

.. py:function:: mindspore.ops.isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回一个布尔型Tensor，表示 `x1` 的每个元素与 `x2` 的对应元素在给定容忍度内是否“接近”。其中“接近”的数学公式为：

    .. math::
        ∣x1−x2∣  ≤  atol + rtol × ∣x2∣

    参数：
        - **x1** (Tensor) - 对比的第一个输入，支持的类型有float32，float16，int32。
        - **x2** (Tensor) - 对比的第二个输入，支持的类型有float32，float16，int32。
        - **rtol** (float, 可选) - 相对容忍度。默认值：1e-05。
        - **atol** (float, 可选) - 绝对容忍度。默认值：1e-08。
        - **equal_nan** (bool, 可选) - 若为True，则两个NaN被视为相同。默认值：False。

    返回：
        Tensor，shape与广播后的shape相同，数据类型是布尔型。

    异常：
        - **TypeError** - `x1` 和 `x2` 中的任何一个不是Tensor。
        - **TypeError** - `x1` 和 `x2` 的数据类型不是float16、float32或int32之一。
        - **TypeError** - `atol` 和 `rtol` 中的任何一个不是float。
        - **TypeError** - `equal_nan` 不是bool。
        - **TypeError** - `x1` 和 `x2` 的数据类型不同。
        - **ValueError** - `x1` 和 `x2` 无法广播。
        - **ValueError** - `atol` 和 `rtol` 中的任何一个小于零。
