mindspore.ops.isclose
=====================

.. py:function:: mindspore.ops.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回一个布尔型Tensor，表示 `input` 的每个元素与 `other` 的对应元素在给定容忍度内是否“接近”。其中“接近”的数学公式为：

    .. math::
        ∣input−other∣  ≤  atol + rtol × ∣other∣

    参数：
        - **input** (Tensor) - 对比的第一个输入，支持的类型有float32，float16，int32。
        - **other** (Tensor) - 对比的第二个输入，支持的类型有float32，float16，int32。
        - **rtol** (float, 可选) - 相对容忍度。默认值： ``1e-05`` 。
        - **atol** (float, 可选) - 绝对容忍度。默认值： ``1e-08`` 。
        - **equal_nan** (bool, 可选) - 若为True，则两个NaN被视为相同。默认值： ``False`` 。

    返回：
        Tensor，shape与广播后的shape相同，数据类型是布尔型。

    异常：
        - **TypeError** - `input` 和 `other` 中的任何一个不是Tensor。
        - **TypeError** - `input` 和 `other` 的数据类型不是float16、float32或int32之一。
        - **TypeError** - `atol` 和 `rtol` 中的任何一个不是float。
        - **TypeError** - `equal_nan` 不是bool。
        - **TypeError** - `input` 和 `other` 的数据类型不同。
        - **ValueError** - `input` 和 `other` 无法广播。
        - **ValueError** - `atol` 和 `rtol` 中的任何一个小于零。
